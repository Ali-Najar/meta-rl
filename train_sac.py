import os
import csv
import time
import warnings
import random

import gymnasium as gym
import metaworld
import numpy as np
import torch
import torch.optim as optim

from ttt import TTTConfig
from agent import TTTEpisodePolicy
from arguments import get_args
from sac_meta import (
    SquashedGaussianActor,
    Critic,
    KStepForecaster,
    TrialReplayBuffer,
    sac_update,
)
from utils import (
    MetaWorldTaskSamplerEnv,
    RunningMeanStd,
    get_agent_input,
    get_success_array,
)
from train import set_seed, make_run_dir, make_sampler_env, build_metaworld

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _cached_context_step(context_model, inp_t, cache_params):
    """Advance the current-episode TTT context by exactly one token.

    Rollout/eval must always use cached full-prefix context.  The
    --context_seq_len option is reserved for replay/training-time context
    reconstruction, where we intentionally train on shorter windows.
    """
    x = context_model.input_encoder(inp_t[:, None, :])
    out = context_model.model(
        inputs_embeds=x,
        cache_params=cache_params,
        use_cache=True,
        return_dict=True,
    )
    current_hidden = out.last_hidden_state[:, -1, :]
    return current_hidden, out.cache_params


@torch.no_grad()
def evaluate_meta_learning_sac(context_model, actor, meta_learning, obs_normalizer, args, device=device):
    """Evaluate SAC actor using the same prefix-report scheme as PPO eval."""
    classes = meta_learning.test_classes
    tasks = meta_learning.test_tasks
    n = args.eval_num_tasks
    eval_trial_length = args.eval_trial_length
    report_lengths = tuple(args.eval_report_lengths)

    if context_model.aggregator_type == "concat" and eval_trial_length != context_model.num_episodes:
        raise ValueError(
            "Evaluation trial length different from training trial_length is only supported "
            "with aggregator_type='mean' or 'ema'."
        )

    envs = gym.vector.SyncVectorEnv(
        [make_sampler_env(classes, tasks, args.seed + 20_000 + i) for i in range(n)]
    )
    action_dim = envs.single_action_space.shape[0]
    eval_class_names = list(classes.keys())
    if n < len(eval_class_names):
        raise ValueError(
            f"eval_num_tasks={n} is smaller than number of test classes={len(eval_class_names)}."
        )

    returns = np.zeros((args.eval_num_trials, n, eval_trial_length), dtype=np.float32)
    successes = np.zeros((args.eval_num_trials, n, eval_trial_length), dtype=bool)
    detailed_rows = []

    context_model.eval()
    actor.eval()

    for eval_round in range(args.eval_num_trials):
        assigned_classes = []
        while len(assigned_classes) < n:
            assigned_classes.extend(eval_class_names)
        assigned_classes = assigned_classes[:n]
        np.random.shuffle(assigned_classes)

        for env, env_name in zip(envs.envs, assigned_classes):
            env.sample_new_task(env_name=env_name)

        raw_obs, _ = envs.reset()
        obs = obs_normalizer.normalize(raw_obs)
        episode_memory = context_model.init_episode_memory(n, device=device, num_episodes=eval_trial_length)
        prev_action = np.zeros((n, action_dim), dtype=np.float32)
        prev_reward = np.zeros((n, 1), dtype=np.float32)
        prev_done = np.zeros((n, 1), dtype=np.float32)
        cumulative_any = np.zeros(n, dtype=bool)

        for ep in range(eval_trial_length):
            ep_rewards = np.zeros(n, dtype=np.float32)
            ep_success = np.zeros(n, dtype=bool)
            cache_params = None
            last_hidden = torch.zeros((n, context_model.hidden_size), device=device)

            for step in range(args.rollout_steps):
                agent_input = get_agent_input(obs, prev_action, prev_reward, prev_done, args.agent_mode)
                inp_t = torch.tensor(agent_input, dtype=torch.float32, device=device)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

                # Rollout/eval uses the full current-episode prefix via the TTT cache.
                # Do not apply --context_seq_len here; that option is for replay/training
                # context reconstruction only.
                current_hidden, cache_params = _cached_context_step(context_model, inp_t, cache_params)
                last_hidden = current_hidden
                z = context_model.aggregate_step(episode_memory, current_hidden, ep)
                action_t = actor.act(obs_t, z, deterministic=True)
                action_np = action_t.cpu().numpy()

                next_raw_obs, reward, terminated, truncated, info = envs.step(action_np)
                done = np.logical_or(terminated, truncated)
                ep_rewards += reward
                ep_success = np.logical_or(ep_success, get_success_array(info, n))

                prev_action = action_np
                prev_reward = reward[:, None]
                prev_done = done[:, None].astype(np.float32)
                obs = obs_normalizer.normalize(next_raw_obs)
                if done.all():
                    break

            episode_memory[:, ep, :] = last_hidden.detach()
            returns[eval_round, :, ep] = ep_rewards
            successes[eval_round, :, ep] = ep_success
            cumulative_any = np.logical_or(cumulative_any, ep_success)

            for env_idx in range(n):
                detailed_rows.append(
                    {
                        "eval_round": eval_round,
                        "eval_trial_id": eval_round * n + env_idx,
                        "env_index": env_idx,
                        "env_name": assigned_classes[env_idx],
                        "episode_index": ep + 1,
                        "episode_return": float(ep_rewards[env_idx]),
                        "episode_success": int(ep_success[env_idx]),
                        "cum_anysuccess": int(cumulative_any[env_idx]),
                    }
                )

            if ep < eval_trial_length - 1:
                raw_obs, _ = envs.reset()
                obs = obs_normalizer.normalize(raw_obs)
                prev_action[:] = 0.0
                prev_reward[:] = 0.0
                prev_done[:] = 0.0

    envs.close()

    summaries = {}
    print("Eval adaptation curve from one max-length rollout:")
    for ep in range(eval_trial_length):
        mean_ret = float(returns[:, :, ep].mean())
        mean_succ = float(successes[:, :, ep].mean())
        cum_any = float(successes[:, :, : ep + 1].any(axis=2).mean())
        print(f"  ep {ep + 1:02d}: return={mean_ret:.2f}, success={100 * mean_succ:.1f}%, any<=ep={100 * cum_any:.1f}%")

    for length in report_lengths:
        prefix_returns = returns[:, :, :length]
        prefix_successes = successes[:, :, :length]
        final_returns = returns[:, :, length - 1]
        final_successes = successes[:, :, length - 1]
        any_successes = prefix_successes.any(axis=2)
        prefix = f"eval_len_{length}"
        summaries[f"{prefix}_trial_return"] = float(prefix_returns.sum(axis=2).mean())
        summaries[f"{prefix}_final_return"] = float(final_returns.mean())
        summaries[f"{prefix}_final_success"] = float(final_successes.mean())
        summaries[f"{prefix}_anysuccess"] = float(any_successes.mean())
        summaries[f"{prefix}_mean_ep_success"] = float(prefix_successes.mean())
        print(
            f"Eval prefix {length}: trial_return={summaries[f'{prefix}_trial_return']:.2f}, "
            f"final_success={100 * summaries[f'{prefix}_final_success']:.1f}%, "
            f"anysuccess={100 * summaries[f'{prefix}_anysuccess']:.1f}%"
        )

    context_model.train()
    actor.train()
    return summaries, detailed_rows


def make_balanced_class_assignment(class_names, num_envs, rng):
    """Return a shuffled list of class names with near-equal counts.

    Example:
        ML10, num_envs=50 -> exactly 5 copies of each class.
        ML10, num_envs=53 -> 5 copies each + 3 random extra classes.
    """
    class_names = list(class_names)
    if len(class_names) == 0:
        raise ValueError("No task classes available.")

    base = num_envs // len(class_names)
    remainder = num_envs % len(class_names)

    assigned = []
    for name in class_names:
        assigned.extend([name] * base)

    if remainder > 0:
        extra = rng.choice(class_names, size=remainder, replace=False).tolist()
        assigned.extend(extra)

    rng.shuffle(assigned)
    return assigned

def train():
    args = get_args()
    set_seed(args.seed)
    print(f"Using device: {device}")
    print(
        f"SAC+TTT | task_set={args.task_set} | envs={args.num_envs} | "
        f"trial_length={args.trial_length} | ep_len={args.rollout_steps}"
    )

    if args.task_set not in ["ML1", "ML10", "ML45"]:
        raise ValueError("train_sac.py is for ML1/ML10/ML45 meta-learning.")
    if args.aggregator_type == "concat":
        warnings.warn(
            "SAC context sampling/detaching is designed for aggregator_type=mean or ema. "
            "concat can run for rollout but sampled previous-episode SAC context is not supported.",
            UserWarning,
        )
    if args.sac_update_ttt_with_sac:
        warnings.warn(
            "--sac_update_ttt_with_sac lets actor/critic gradients update TTT. "
            "This is less stable; default is to train TTT with forecasting and detach z for SAC.",
            UserWarning,
        )

    meta_learning = build_metaworld(args)
    envs = gym.vector.SyncVectorEnv(
        [make_sampler_env(meta_learning.train_classes, meta_learning.train_tasks, args.seed + i) for i in range(args.num_envs)]
    )

    
    train_class_names = list(meta_learning.train_classes.keys())
    task_assignment_rng = np.random.default_rng(args.seed + 12345)


    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    input_dim = obs_dim
    if args.agent_mode == "agent_v2":
        input_dim = obs_dim + action_dim + 1
    elif args.agent_mode == "agent_rl2":
        input_dim = obs_dim + action_dim + 2

    obs_normalizer = RunningMeanStd(shape=(obs_dim,))

    config = TTTConfig(
        vocab_size=1,
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 3,
        num_attention_heads=args.num_attention_heads,
        max_position_embeddings=max(2048, args.rollout_steps),
        num_hidden_layers=args.num_hidden_layers,
        ttt_layer_type=args.ttt_layer_type,
        rms_norm_eps=1e-5,
        use_cache=True,
        mini_batch_size=args.mini_batch_size,
        scan_checkpoint_group_size=0,
        tie_word_embeddings=False,
    )

    context_model = TTTEpisodePolicy(
        config,
        input_dim=input_dim,
        obs_dim=obs_dim,
        num_actions=action_dim,
        num_episodes=args.trial_length,
        continuous=True,
        policy_hidden_sizes=(),
        value_hidden_sizes=(),
        aggregator_type=args.aggregator_type,
        ema_beta=args.ema_beta,
        use_state_proj=False,
        init_type=args.init_type,
        context_seq_len=args.context_seq_len,
        prev_context_window_mode=args.prev_context_window_mode,
        min_std=args.min_std,
        max_std=args.max_std,
        init_std=args.init_std,
    ).to(device)

    actor = SquashedGaussianActor(obs_dim, args.hidden_size, action_dim, args.sac_hidden_sizes).to(device)
    q1 = Critic(obs_dim, args.hidden_size, action_dim, args.sac_hidden_sizes).to(device)
    q2 = Critic(obs_dim, args.hidden_size, action_dim, args.sac_hidden_sizes).to(device)
    q1_target = Critic(obs_dim, args.hidden_size, action_dim, args.sac_hidden_sizes).to(device)
    q2_target = Critic(obs_dim, args.hidden_size, action_dim, args.sac_hidden_sizes).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    forecaster = KStepForecaster(obs_dim, args.hidden_size, action_dim, args.sac_hidden_sizes).to(device)

    # Only the TTT input encoder and TTTModel are context parameters.
    # TTTEpisodePolicy also has PPO-style policy/value heads, but train_sac.py
    # does not use them. Keeping them out avoids unused optimizer state.
    context_params = list(context_model.input_encoder.parameters()) + list(context_model.model.parameters())
    optimizers = {
        "actor": optim.Adam(actor.parameters(), lr=args.sac_actor_lr),
        "critic": optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.sac_critic_lr),
        "context_forecast": optim.Adam(context_params + list(forecaster.parameters()), lr=args.sac_context_lr),
        "context_sac": optim.Adam(context_params, lr=args.sac_context_lr),
    }
    replay = TrialReplayBuffer(args.sac_replay_size)

    out_dir = make_run_dir(args)
    print(f"Run directory: {out_dir}")
    eval_metric_keys = []
    for length in args.eval_report_lengths:
        prefix = f"eval_len_{length}"
        eval_metric_keys.extend([
            f"{prefix}_trial_return",
            f"{prefix}_final_return",
            f"{prefix}_final_success",
            f"{prefix}_anysuccess",
            f"{prefix}_mean_ep_success",
        ])

    metrics_header = [
        "update", "timestep", "replay_trials", "rollout_trial_return", "rollout_final_return",
        "rollout_final_success", "rollout_anysuccess", *eval_metric_keys,
        "forecast_loss", "forecast_obs_loss", "forecast_reward_loss", "critic_loss", "actor_loss",
        "q1", "q2", "entropy", "sac_steps", "forecast_steps", "initial_forecast_steps",
        "sac_train_epochs", "sac_episode_batch_size", "sac_chunk_steps", "sac_effective_batch_size",
    ]
    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(metrics_header)

    eval_episode_csv_path = os.path.join(out_dir, "eval_episode_success.csv")
    with open(eval_episode_csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "update", "timestep", "eval_trial_length", "eval_round", "eval_trial_id",
            "env_index", "env_name", "episode_index", "episode_return", "episode_success", "cum_anysuccess",
        ])

    raw_obs, _ = envs.reset()
    obs_normalizer.update(raw_obs)
    obs = obs_normalizer.normalize(raw_obs)
    global_timesteps = 0
    global_episodes = 0
    did_initial_sac_training = False

    for update in range(args.num_updates):
        t0 = time.time()
        # Sample a new task variation once per trial, then keep it fixed across trial_length episodes.

        assigned_classes = make_balanced_class_assignment(
            train_class_names,
            args.num_envs,
            task_assignment_rng,
        )

        for env, env_name in zip(envs.envs, assigned_classes):
            env.sample_new_task(env_name=env_name)

        raw_obs, _ = envs.reset()
        obs_normalizer.update(raw_obs)
        obs = obs_normalizer.normalize(raw_obs)

        B, E, T = args.num_envs, args.trial_length, args.rollout_steps
        b_inputs = np.zeros((B, E, T, input_dim), dtype=np.float32)
        b_states = np.zeros((B, E, T, obs_dim), dtype=np.float32)
        b_next_states = np.zeros((B, E, T, obs_dim), dtype=np.float32)
        b_actions = np.zeros((B, E, T, action_dim), dtype=np.float32)
        b_rewards = np.zeros((B, E, T), dtype=np.float32)
        b_dones = np.zeros((B, E, T), dtype=np.float32)

        episode_memory = context_model.init_episode_memory(B, device=device)
        prev_action = np.zeros((B, action_dim), dtype=np.float32)
        prev_reward = np.zeros((B, 1), dtype=np.float32)
        prev_done = np.zeros((B, 1), dtype=np.float32)
        trial_ep_rewards = np.zeros((B, E), dtype=np.float32)
        trial_ep_successes = np.zeros((B, E), dtype=bool)

        context_model.eval()
        actor.eval()
        for ep in range(E):
            cache_params = None
            ep_rewards = np.zeros(B, dtype=np.float32)
            ep_success = np.zeros(B, dtype=bool)
            last_hidden = torch.zeros((B, args.hidden_size), device=device)

            for step in range(T):
                agent_input = get_agent_input(obs, prev_action, prev_reward, prev_done, args.agent_mode)
                inp_t = torch.tensor(agent_input, dtype=torch.float32, device=device)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

                with torch.no_grad():
                    # Collection always advances one token through the cached full-prefix
                    # TTT state.  --context_seq_len must not cause rollout-time window
                    # recomputation; it is used only inside SAC replay/training updates.
                    current_hidden, cache_params = _cached_context_step(context_model, inp_t, cache_params)
                    last_hidden = current_hidden
                    z = context_model.aggregate_step(episode_memory, current_hidden, ep)
                    if global_timesteps < args.sac_random_steps:
                        action_np = np.stack([envs.single_action_space.sample() for _ in range(B)], axis=0).astype(np.float32)
                    else:
                        action_np = actor.act(obs_t, z, deterministic=False).cpu().numpy()

                next_raw_obs, raw_reward, terminated, truncated, info = envs.step(action_np)
                global_timesteps += B
                done = np.logical_or(terminated, truncated)
                success = get_success_array(info, B)
                ep_rewards += raw_reward
                ep_success = np.logical_or(ep_success, success)

                obs_normalizer.update(next_raw_obs)
                next_obs_norm = obs_normalizer.normalize(next_raw_obs)

                b_inputs[:, ep, step, :] = agent_input
                b_states[:, ep, step, :] = obs
                b_next_states[:, ep, step, :] = next_obs_norm
                b_actions[:, ep, step, :] = action_np
                b_rewards[:, ep, step] = raw_reward
                b_dones[:, ep, step] = done.astype(np.float32)

                prev_action = action_np
                prev_reward = raw_reward[:, None]
                prev_done = done[:, None].astype(np.float32)
                obs = next_obs_norm
                if done.all():
                    # MetaWorld usually truncates exactly at T. We keep arrays fixed-size.
                    break

            episode_memory[:, ep, :] = last_hidden.detach()
            trial_ep_rewards[:, ep] = ep_rewards
            trial_ep_successes[:, ep] = ep_success
            global_episodes += B
            if ep < E - 1:
                raw_obs, _ = envs.reset()
                obs = obs_normalizer.normalize(raw_obs)
                prev_action[:] = 0.0
                prev_reward[:] = 0.0
                prev_done[:] = 0.0

        context_model.train()
        actor.train()
        data_time = time.time() - t0
        replay.add_trial_batch(b_inputs, b_states, b_actions, b_rewards, b_dones, b_next_states)

        update_t0 = time.time()
        if len(replay) > 0:
            stats = sac_update(
                context_model,
                actor,
                q1,
                q2,
                q1_target,
                q2_target,
                forecaster,
                optimizers,
                replay,
                args,
                device,
                first_update=not did_initial_sac_training,
            )
            did_initial_sac_training = True
        else:
            stats = {
                "forecast_loss": np.nan, "forecast_obs_loss": np.nan, "forecast_reward_loss": np.nan,
                "critic_loss": np.nan, "actor_loss": np.nan, "q1": np.nan, "q2": np.nan,
                "entropy": np.nan, "sac_steps": 0.0, "forecast_steps": 0.0,
                "initial_forecast_steps": 0.0, "sac_train_epochs": float(args.sac_train_epochs),
                "sac_episode_batch_size": float(args.sac_episode_batch_size),
                "sac_chunk_steps": float(args.sac_chunk_steps),
                "sac_effective_batch_size": float(args.sac_episode_batch_size * args.sac_chunk_steps),
            }
        update_time = time.time() - update_t0

        rollout_trial_return = float(trial_ep_rewards.sum(axis=1).mean())
        rollout_final_return = float(trial_ep_rewards[:, -1].mean())
        rollout_final_success = float(trial_ep_successes[:, -1].mean())
        rollout_anysuccess = float(trial_ep_successes.any(axis=1).mean())

        print(
            f"Update {update+1}/{args.num_updates} | steps={global_timesteps} | "
            f"Data={data_time:.2f}s SAC={update_time:.2f}s | "
            f"ret={rollout_trial_return:.2f} final_succ={100*rollout_final_success:.1f}% any={100*rollout_anysuccess:.1f}% | "
            f"critic={_safe_float(stats.get('critic_loss')):.4f} actor={_safe_float(stats.get('actor_loss')):.4f} "
            f"forecast={_safe_float(stats.get('forecast_loss')):.4f}"
        )

        eval_summaries = {key: np.nan for key in eval_metric_keys}
        if (update + 1) % args.eval_interval == 0:
            eval_summaries, eval_rows = evaluate_meta_learning_sac(context_model, actor, meta_learning, obs_normalizer, args, device=device)
            with open(eval_episode_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for row in eval_rows:
                    writer.writerow([
                        update + 1,
                        global_timesteps,
                        args.eval_trial_length,
                        row["eval_round"],
                        row["eval_trial_id"],
                        row["env_index"],
                        row["env_name"],
                        row["episode_index"],
                        row["episode_return"],
                        row["episode_success"],
                        row["cum_anysuccess"],
                    ])

        metrics_row = [
            update + 1,
            global_timesteps,
            len(replay),
            rollout_trial_return,
            rollout_final_return,
            rollout_final_success,
            rollout_anysuccess,
            *[eval_summaries.get(key, np.nan) for key in eval_metric_keys],
            stats.get("forecast_loss", np.nan),
            stats.get("forecast_obs_loss", np.nan),
            stats.get("forecast_reward_loss", np.nan),
            stats.get("critic_loss", np.nan),
            stats.get("actor_loss", np.nan),
            stats.get("q1", np.nan),
            stats.get("q2", np.nan),
            stats.get("entropy", np.nan),
            stats.get("sac_steps", 0.0),
            stats.get("forecast_steps", 0.0),
            stats.get("initial_forecast_steps", 0.0),
            stats.get("sac_train_epochs", float(args.sac_train_epochs)),
            stats.get("sac_episode_batch_size", float(args.sac_episode_batch_size)),
            stats.get("sac_chunk_steps", float(args.sac_chunk_steps)),
            stats.get("sac_effective_batch_size", float(args.sac_episode_batch_size * args.sac_chunk_steps)),
        ]
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(metrics_row)

    torch.save(
        {
            "context_model": context_model.state_dict(),
            "actor": actor.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "forecaster": forecaster.state_dict(),
        },
        os.path.join(out_dir, "sac_ttt_policy.pth"),
    )
    envs.close()


if __name__ == "__main__":
    train()
