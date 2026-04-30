import os
import csv
import time
import json
import sys
from datetime import datetime
import random
import warnings
import gymnasium as gym
import metaworld
import numpy as np
import torch
import torch.optim as optim

from ttt import TTTConfig
from agent import TTTEpisodePolicy
from arguments import get_args
from utils import (
    MetaWorldTaskSamplerEnv,
    RunningMeanStd,
    RewardNormalizer,
    create_scheduler,
    get_agent_input,
    get_success_array,
    compute_gae,
    train_ppo,
)

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)




def _json_safe(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def make_run_dir(args):
    """Create an incrementally numbered run directory and save config.json."""
    if args.run_name:
        base_name = args.run_name
    else:
        comment = f"_{args.comment}" if args.comment else ""
        base_name = (
            f"ttt_ecet_{args.task_set}_{args.env_name}"
            f"_{args.agent_mode}_tl{args.trial_length}_hs{args.hidden_size}"
            f"_layers{args.num_hidden_layers}_mb{args.mini_batch_size}{comment}_seed{args.seed}"
        )

    os.makedirs(args.run_root, exist_ok=True)
    run_idx = 0
    while True:
        run_dir = os.path.join(args.run_root, f"{base_name}_run{run_idx:03d}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=False)
            break
        run_idx += 1

    config = {k: _json_safe(v) for k, v in vars(args).items()}
    config["run_index"] = run_idx
    config["run_dir"] = run_dir
    config["created_at"] = datetime.now().isoformat(timespec="seconds")
    config["command"] = " ".join(sys.argv)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    return run_dir

def make_sampler_env(classes, tasks, seed):
    def thunk():
        return MetaWorldTaskSamplerEnv(classes, tasks, seed=seed, mask_goal=True)

    return thunk


def build_metaworld(args):
    if args.task_set == "ML10":
        return metaworld.ML10()
    if args.task_set == "ML45":
        return metaworld.ML45()
    if args.task_set == "ML1":
        return metaworld.ML1(args.env_name)
    raise ValueError("This refactor is intended for ML1/ML10/ML45 meta-learning.")


@torch.no_grad()
def evaluate_meta_learning(model, meta_learning, obs_normalizer, args, device=device):
    classes = meta_learning.test_classes
    tasks = meta_learning.test_tasks
    n = args.eval_num_tasks

    envs = gym.vector.SyncVectorEnv(
        [make_sampler_env(classes, tasks, args.seed + 10_000 + i) for i in range(n)]
    )

    action_dim = envs.single_action_space.shape[0]

    final_rewards = []
    final_successes = []
    any_successes = []

    adaptation_returns = np.zeros((args.trial_length,), dtype=np.float32)
    adaptation_success = np.zeros((args.trial_length,), dtype=np.float32)
    adaptation_count = np.zeros((args.trial_length,), dtype=np.float32)

    for _ in range(args.eval_num_trials):
        
        eval_class_names = list(classes.keys())

        if n < len(eval_class_names):
            raise ValueError(
                f"eval_num_tasks={n} is smaller than number of test classes={len(eval_class_names)}. "
                "Increase --eval_num_tasks if you want every test class represented."
            )

        # Repeat class names until we have n env assignments, then shuffle.
        assigned_classes = []
        while len(assigned_classes) < n:
            assigned_classes.extend(eval_class_names)
        assigned_classes = assigned_classes[:n]
        np.random.shuffle(assigned_classes)

        for env, env_name in zip(envs.envs, assigned_classes):
            env.sample_new_task(env_name=env_name)

        raw_obs, _ = envs.reset()
        obs = obs_normalizer.normalize(raw_obs)

        episode_memory = model.init_episode_memory(n, device=device)
        prev_action = np.zeros((n, action_dim), dtype=np.float32)
        prev_reward = np.zeros((n, 1), dtype=np.float32)
        prev_done = np.zeros((n, 1), dtype=np.float32)

        trial_any_success = np.zeros(n, dtype=bool)

        for ep in range(args.trial_length):
            ep_rewards = np.zeros(n, dtype=np.float32)
            ep_success = np.zeros(n, dtype=bool)
            cache_params = None

            for step in range(args.rollout_steps):
                agent_input = get_agent_input(obs, prev_action, prev_reward, prev_done, args.agent_mode)
                inp_t = torch.tensor(agent_input, dtype=torch.float32, device=device)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

                out = model.act_step(inp_t, obs_t, episode_memory, ep, cache_params)
                mean, _ = out.policy
                action_np = mean.cpu().numpy()

                next_raw_obs, reward, terminated, truncated, info = envs.step(action_np)
                done = np.logical_or(terminated, truncated)

                ep_rewards += reward
                ep_success = np.logical_or(ep_success, get_success_array(info, n))

                cache_params = out.cache_params
                prev_action = action_np
                prev_reward = reward[:, None]
                prev_done = done[:, None].astype(np.float32)
                obs = obs_normalizer.normalize(next_raw_obs)

                if done.all():
                    break

            episode_memory[:, ep, :] = out.hidden_states.detach()

            adaptation_returns[ep] += ep_rewards.mean()
            adaptation_success[ep] += ep_success.mean()
            adaptation_count[ep] += 1

            trial_any_success = np.logical_or(trial_any_success, ep_success)

            if ep == args.trial_length - 1:
                final_rewards.extend(ep_rewards.tolist())
                final_successes.extend(ep_success.tolist())

            if ep < args.trial_length - 1:
                raw_obs, _ = envs.reset()
                obs = obs_normalizer.normalize(raw_obs)
                prev_action[:] = 0.0
                prev_reward[:] = 0.0
                prev_done[:] = 0.0

        any_successes.extend(trial_any_success.tolist())

    envs.close()

    adaptation_returns /= np.maximum(adaptation_count, 1)
    adaptation_success /= np.maximum(adaptation_count, 1)

    print("Eval adaptation curve:")
    for ep in range(args.trial_length):
        print(
            f"  ep {ep + 1:02d}: "
            f"return={adaptation_returns[ep]:.2f}, "
            f"success={100 * adaptation_success[ep]:.1f}%"
        )

    eval_final_return = float(np.mean(final_rewards)) if final_rewards else float("nan")
    eval_final_success = float(np.mean(final_successes)) if final_successes else float("nan")
    eval_anysuccess = float(np.mean(any_successes)) if any_successes else float("nan")

    print(f"Eval anysuccess across trial: {100 * eval_anysuccess:.1f}%")

    return eval_final_return, eval_final_success, eval_anysuccess


def train():
    args = get_args()
    set_seed(args.seed)

    print(f"Using device: {device}")
    print(
        f"Task set: {args.task_set} | "
        f"envs={args.num_envs} | "
        f"trial_length={args.trial_length} | "
        f"ep_len={args.rollout_steps}"
    )

    if args.task_set not in ["ML1", "ML10", "ML45"]:
        raise ValueError("This refactored trainer is for ML1/ML10/ML45.")

    meta_learning = build_metaworld(args)

    envs = gym.vector.SyncVectorEnv(
        [
            make_sampler_env(meta_learning.train_classes, meta_learning.train_tasks, args.seed + i)
            for i in range(args.num_envs)
        ]
    )

    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    input_dim = obs_dim
    if args.agent_mode == "agent_v2":
        input_dim = obs_dim + action_dim + 1
    elif args.agent_mode == "agent_rl2":
        input_dim = obs_dim + action_dim + 2

    obs_normalizer = RunningMeanStd(shape=(obs_dim,))
    reward_normalizer = RewardNormalizer(gamma=args.gamma)

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

    model = TTTEpisodePolicy(
        config,
        input_dim=input_dim,
        obs_dim=obs_dim,
        num_actions=action_dim,
        num_episodes=args.trial_length,
        continuous=True,
        policy_hidden_sizes=args.policy_hidden_sizes,
        value_hidden_sizes=args.value_hidden_sizes,
        aggregator_type=args.aggregator_type,
        use_state_proj=args.use_state_proj,
        init_type=args.init_type,
        min_std=args.min_std,
        max_std=args.max_std,
        init_std=args.init_std,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = create_scheduler(optimizer, args)

    out_dir = make_run_dir(args)
    print(f"Run directory: {out_dir}")

    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "update",
                "timestep",
                "rollout_trial_return",
                "rollout_final_return",
                "rollout_final_success",
                "rollout_anysuccess",
                "eval_final_return",
                "eval_final_success",
                "eval_anysuccess",
                "loss",
                "policy_loss",
                "value_loss",
                "entropy",
            ]
        )

    raw_obs, _ = envs.reset()
    obs_normalizer.update(raw_obs)
    obs = obs_normalizer.normalize(raw_obs)

    global_episodes = 0
    global_timesteps = 0

    for update in range(args.num_updates):
        t0 = time.time()

        for env in envs.envs:
            env.sample_new_task()

        raw_obs, _ = envs.reset()
        obs_normalizer.update(raw_obs)
        obs = obs_normalizer.normalize(raw_obs)

        B, E, T = args.num_envs, args.trial_length, args.rollout_steps

        b_inputs = torch.zeros((B, E, T, input_dim), device=device)
        b_states = torch.zeros((B, E, T, obs_dim), device=device)
        b_actions = torch.zeros((B, E, T, action_dim), device=device)
        b_logprobs = torch.zeros((B, E, T), device=device)
        b_rewards = torch.zeros((B, E, T), device=device)
        b_dones = torch.zeros((B, E, T), device=device)
        b_values = torch.zeros((B, E, T), device=device)

        episode_memory = model.init_episode_memory(B, device=device)

        prev_action = np.zeros((B, action_dim), dtype=np.float32)
        prev_reward = np.zeros((B, 1), dtype=np.float32)
        prev_done = np.zeros((B, 1), dtype=np.float32)

        trial_ep_rewards = np.zeros((B, E), dtype=np.float32)
        trial_ep_successes = np.zeros((B, E), dtype=bool)

        for ep in range(E):
            cache_params = None
            ep_rewards = np.zeros(B, dtype=np.float32)
            ep_success = np.zeros(B, dtype=bool)

            for step in range(T):
                agent_input = get_agent_input(obs, prev_action, prev_reward, prev_done, args.agent_mode)
                inp_t = torch.tensor(agent_input, dtype=torch.float32, device=device)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

                with torch.no_grad():
                    out = model.act_step(inp_t, obs_t, episode_memory, ep, cache_params)
                    mean, log_std = out.policy
                    dist = torch.distributions.Normal(mean, log_std.exp())
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                    value = out.value

                action_np = action.cpu().numpy()

                next_raw_obs, raw_reward, terminated, truncated, info = envs.step(action_np)
                global_timesteps += B
                done = np.logical_or(terminated, truncated)
                success = get_success_array(info, B)

                ep_rewards += raw_reward
                ep_success = np.logical_or(ep_success, success)

                obs_normalizer.update(next_raw_obs)
                norm_reward = reward_normalizer.normalize(raw_reward, done.astype(np.float32))

                b_inputs[:, ep, step, :] = inp_t
                b_states[:, ep, step, :] = obs_t
                b_actions[:, ep, step, :] = action
                b_logprobs[:, ep, step] = log_prob
                b_rewards[:, ep, step] = torch.tensor(norm_reward, dtype=torch.float32, device=device)
                b_dones[:, ep, step] = torch.tensor(done, dtype=torch.float32, device=device)
                b_values[:, ep, step] = value

                cache_params = out.cache_params
                prev_action = action_np
                prev_reward = raw_reward[:, None]
                prev_done = done[:, None].astype(np.float32)
                obs = obs_normalizer.normalize(next_raw_obs)

                if done.all():
                    break

            episode_memory[:, ep, :] = out.hidden_states.detach()
            trial_ep_rewards[:, ep] = ep_rewards
            trial_ep_successes[:, ep] = ep_success
            global_episodes += B

            if ep < E - 1:
                raw_obs, _ = envs.reset()
                obs = obs_normalizer.normalize(raw_obs)
                prev_action[:] = 0.0
                prev_reward[:] = 0.0
                prev_done[:] = 0.0

        data_time = time.time() - t0

        with torch.no_grad():
            next_value = torch.zeros(B, device=device)

        b_adv, b_returns = compute_gae(
            b_rewards,
            b_values,
            b_dones,
            next_value,
            args.gamma,
            args.gae_lambda,
        )

        ppo_t0 = time.time()
        rollouts = (b_inputs, b_states, b_actions, b_logprobs, b_adv, b_returns)
        loss, p_loss, v_loss, ent = train_ppo(model, optimizer, rollouts, args, device)
        ppo_time = time.time() - ppo_t0

        rollout_trial_return = float(trial_ep_rewards.sum(axis=1).mean())
        rollout_final_return = float(trial_ep_rewards[:, -1].mean())
        rollout_final_success = float(trial_ep_successes[:, -1].mean())
        rollout_anysuccess = float(trial_ep_successes.any(axis=1).mean())

        print(
            f"Update {update + 1}/{args.num_updates} | global_ep={global_episodes} | "
            f"timesteps={global_timesteps} | "
            f"Data={data_time:.2f}s PPO={ppo_time:.2f}s | "
            f"loss={loss:.4f} p={p_loss:.4f} v={v_loss:.4f} ent={ent:.4f} | "
            f"rollout_trial_return={rollout_trial_return:.2f} "
            f"final_return={rollout_final_return:.2f} "
            f"final_success={100 * rollout_final_success:.1f}% "
            f"anysuccess={100 * rollout_anysuccess:.1f}%"
        )

        eval_ret = np.nan
        eval_succ = np.nan
        eval_anysuccess = np.nan

        if (update + 1) % args.eval_interval == 0:
            print("policy std:", model.get_policy_std().detach().cpu().numpy())

            eval_ret, eval_succ, eval_anysuccess = evaluate_meta_learning(
                model,
                meta_learning,
                obs_normalizer,
                args,
                device=device,
            )

            print(
                f"Eval final return={eval_ret:.2f}, "
                f"final_success={100 * eval_succ:.1f}%, "
                f"anysuccess={100 * eval_anysuccess:.1f}%"
            )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    update + 1,
                    global_timesteps,
                    rollout_trial_return,
                    rollout_final_return,
                    rollout_final_success,
                    rollout_anysuccess,
                    eval_ret,
                    eval_succ,
                    eval_anysuccess,
                    loss,
                    p_loss,
                    v_loss,
                    ent,
                ]
            )

        if scheduler is not None:
            scheduler.step()

    torch.save({"model": model.state_dict()}, os.path.join(out_dir, "ttt_ecet_policy.pth"))
    envs.close()


if __name__ == "__main__":
    train()
