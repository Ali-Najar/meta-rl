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
import warnings

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

def should_mask_goal(args):
    """Mask goal coordinates only for ML1.

    ML10 and ML45 expose goal coordinates in the observation, matching the
    standard meta-learning benchmark setup.
    """
    return args.task_set == "ML1"


def make_sampler_env(classes, tasks, seed, mask_goal=True):
    def thunk():
        return MetaWorldTaskSamplerEnv(classes, tasks, seed=seed, mask_goal=mask_goal)

    return thunk

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




def summarize_rollout_by_task_class(assigned_classes, trial_ep_rewards, trial_ep_successes):
    """Aggregate rollout returns/successes by Meta-World task class.

    assigned_classes: list[str] length B, one class name per vector env.
    trial_ep_rewards: (B, E) raw episode returns for the collected rollout.
    trial_ep_successes: (B, E) boolean episode success flags.

    Returns a list of dictionaries sorted by task class name.
    """
    assigned = np.asarray(assigned_classes, dtype=object)
    rows = []

    for env_name in sorted(set(assigned_classes)):
        mask = assigned == env_name
        rewards = trial_ep_rewards[mask]
        successes = trial_ep_successes[mask]
        if rewards.shape[0] == 0:
            continue

        rows.append(
            {
                "env_name": env_name,
                "num_envs": int(rewards.shape[0]),
                "trial_return": float(rewards.sum(axis=1).mean()),
                "final_return": float(rewards[:, -1].mean()),
                "final_success": float(successes[:, -1].mean()),
                "anysuccess": float(successes.any(axis=1).mean()),
                "mean_ep_success": float(successes.mean()),
                "episode_returns": rewards.mean(axis=0).astype(float).tolist(),
                "episode_successes": successes.mean(axis=0).astype(float).tolist(),
            }
        )

    return rows

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
    """Run one max-length eval trial and summarize arbitrary prefixes.

    Evaluation runs for args.eval_trial_length episodes once. Metrics for every
    length in args.eval_report_lengths are computed from prefixes of that same
    rollout, e.g. length 5 uses episodes 1..5 from a 25-episode eval.
    """
    classes = meta_learning.test_classes
    tasks = meta_learning.test_tasks
    n = args.eval_num_tasks
    eval_trial_length = args.eval_trial_length
    report_lengths = tuple(args.eval_report_lengths)

    if model.aggregator_type == "concat" and eval_trial_length != model.num_episodes:
        raise ValueError(
            "Evaluation trial length different from training trial_length is only "
            "supported with aggregator_type='mean' or 'ema'. The concat/slot aggregator has "
            "a fixed number of learned episode slots."
        )

    envs = gym.vector.SyncVectorEnv(
        [make_sampler_env(classes, tasks, args.seed + 10_000 + i, mask_goal=should_mask_goal(args)) for i in range(n)]
    )

    action_dim = envs.single_action_space.shape[0]
    eval_class_names = list(classes.keys())
    if n < len(eval_class_names):
        raise ValueError(
            f"eval_num_tasks={n} is smaller than number of test classes={len(eval_class_names)}. "
            "Increase --eval_num_tasks if you want every test class represented."
        )

    # Store raw per-trial, per-episode data so arbitrary prefixes can be computed.
    # Shapes: (R, N, L), where R=eval_num_trials, N=eval_num_tasks, L=eval_trial_length.
    returns = np.zeros((args.eval_num_trials, n, eval_trial_length), dtype=np.float32)
    successes = np.zeros((args.eval_num_trials, n, eval_trial_length), dtype=bool)
    env_names = np.empty((args.eval_num_trials, n), dtype=object)

    detailed_rows = []

    for eval_round in range(args.eval_num_trials):
        # Balance classes across eval workers, then sample one variation per worker.
        assigned_classes = []
        while len(assigned_classes) < n:
            assigned_classes.extend(eval_class_names)
        assigned_classes = assigned_classes[:n]
        np.random.shuffle(assigned_classes)
        env_names[eval_round, :] = assigned_classes

        for env, env_name in zip(envs.envs, assigned_classes):
            env.sample_new_task(env_name=env_name)

        raw_obs, _ = envs.reset()
        obs = obs_normalizer.normalize(raw_obs)

        episode_memory = model.init_episode_memory(
            n,
            device=device,
            num_episodes=eval_trial_length,
        )
        prev_action = np.zeros((n, action_dim), dtype=np.float32)
        prev_reward = np.zeros((n, 1), dtype=np.float32)
        prev_done = np.zeros((n, 1), dtype=np.float32)
        cumulative_any = np.zeros(n, dtype=bool)

        for ep in range(eval_trial_length):
            ep_rewards = np.zeros(n, dtype=np.float32)
            ep_success = np.zeros(n, dtype=bool)
            cache_params = None
            
            rollout_uses_context_window = (args.context_seq_len > 0 and not args.no_rollout_context_seq_len)
            current_episode_inputs = None
            if rollout_uses_context_window:
                current_episode_inputs = torch.zeros(
                    (n, args.rollout_steps, model.input_dim),
                    device=device,
                )

            for step in range(args.rollout_steps):
                agent_input = get_agent_input(obs, prev_action, prev_reward, prev_done, args.agent_mode)
                inp_t = torch.tensor(agent_input, dtype=torch.float32, device=device)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

                context_window = None
                if rollout_uses_context_window:
                    current_episode_inputs[:, step, :] = inp_t
                    win_start = max(0, step + 1 - args.context_seq_len)
                    context_window = current_episode_inputs[:, win_start : step + 1, :]

                out = model.act_step(
                    inp_t,
                    obs_t,
                    episode_memory,
                    ep,
                    cache_params,
                    context_window_inputs=context_window,
                )
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
        print(
            f"  ep {ep + 1:02d}: return={mean_ret:.2f}, "
            f"success={100 * mean_succ:.1f}%, any<=ep={100 * cum_any:.1f}%"
        )

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
            f"Eval prefix {length}: "
            f"trial_return={summaries[f'{prefix}_trial_return']:.2f}, "
            f"final_return={summaries[f'{prefix}_final_return']:.2f}, "
            f"final_success={100 * summaries[f'{prefix}_final_success']:.1f}%, "
            f"anysuccess={100 * summaries[f'{prefix}_anysuccess']:.1f}%, "
            f"mean_ep_success={100 * summaries[f'{prefix}_mean_ep_success']:.1f}%"
        )

    return summaries, detailed_rows

def print_advantage_diagnostics_by_class(
    b_adv,
    assigned_classes,
    update,
    ppo_epoch=None,
    prefix="ADV_DIAG",
):
    """Print raw and globally-normalized advantage stats per task class.

    b_adv: torch.Tensor with shape (B, E, T)
    assigned_classes: list[str] of length B, one task class per vector env
    """
    with torch.no_grad():
        adv = b_adv.detach().float().cpu()  # (B, E, T)
        B = adv.shape[0]

        if len(assigned_classes) != B:
            print(
                f"{prefix} update={update}: skipped; "
                f"len(assigned_classes)={len(assigned_classes)} but B={B}"
            )
            return

        flat = adv.reshape(-1)
        global_mean = flat.mean()
        global_std = flat.std(unbiased=False).clamp_min(1e-8)

        adv_norm = (adv - global_mean) / global_std

        epoch_str = "" if ppo_epoch is None else f" epoch={ppo_epoch}"
        print(
            f"\n{prefix} update={update}{epoch_str} | "
            f"GLOBAL raw_mean={global_mean.item():+.5f} "
            f"raw_std={global_std.item():.5f} "
            f"raw_abs_mean={flat.abs().mean().item():.5f}"
        )

        unique_classes = sorted(set(assigned_classes))
        rows = []

        for cls in unique_classes:
            env_idx = [i for i, name in enumerate(assigned_classes) if name == cls]
            cls_raw = adv[env_idx].reshape(-1)
            cls_norm = adv_norm[env_idx].reshape(-1)

            raw_mean = cls_raw.mean().item()
            raw_std = cls_raw.std(unbiased=False).item()
            raw_abs = cls_raw.abs().mean().item()

            norm_mean = cls_norm.mean().item()
            norm_std = cls_norm.std(unbiased=False).item()
            norm_abs = cls_norm.abs().mean().item()

            # This ratio is useful: <1 means the class has lower raw advantage
            # scale than the global batch; >1 means it has larger scale.
            scale_vs_global = raw_std / global_std.item()

            rows.append(
                (
                    cls,
                    len(env_idx),
                    raw_mean,
                    raw_std,
                    raw_abs,
                    scale_vs_global,
                    norm_mean,
                    norm_std,
                    norm_abs,
                )
            )

        print(
            f"{'class':32s} {'n_env':>5s} "
            f"{'raw_mean':>10s} {'raw_std':>10s} {'raw_abs':>10s} "
            f"{'std/global':>10s} "
            f"{'norm_mean':>10s} {'norm_std':>10s} {'norm_abs':>10s}"
        )

        for row in rows:
            cls, n_env, raw_mean, raw_std, raw_abs, scale_vs_global, norm_mean, norm_std, norm_abs = row
            print(
                f"{cls:32s} {n_env:5d} "
                f"{raw_mean:+10.4f} {raw_std:10.4f} {raw_abs:10.4f} "
                f"{scale_vs_global:10.4f} "
                f"{norm_mean:+10.4f} {norm_std:10.4f} {norm_abs:10.4f}"
            )
        print("")

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

    if args.aggregator_type == "concat" and args.eval_trial_length != args.trial_length:
        warnings.warn(
            "--eval_trial_length different from --trial_length is only supported with "
            "--aggregator_type=mean, ema, or attn. The concat/slot aggregator has a fixed number "
            "of learned episode slots and will error if evaluated with a different length.",
            UserWarning,
        )
    
    if args.ppo_context_episode_sample > 0 and args.ppo_sequential_loss_scope == "prefix":
        warnings.warn(
            "--ppo_sequential_loss_scope=prefix with "
            "--ppo_context_episode_sample > 0 only computes prefix loss over the "
            "current episode prefix, while sampled previous episodes are used as "
            "context only. This is not equivalent to full-trial prefix loss. "
            "Use --ppo_sequential_loss_scope=chunk for the cleaner sampled-context setting.",
            UserWarning,
        )

    if args.ppo_context_episode_sample > 0:
        if args.aggregator_type not in ["mean", "ema", "attn"]:
            warnings.warn(
                "--ppo_context_episode_sample is only supported by --aggregator_type=mean/ema/attn. "
                "The PPO update will raise an error for concat.",
                UserWarning,
            )
        if args.aggregator_type == "ema" and args.context_episode_sample_mode == "uniform":
            warnings.warn(
                "For --aggregator_type=ema, --context_episode_sample_mode=uniform is valid but noisy. "
                "recent or last usually matches recency-weighted EMA better.",
                UserWarning,
            )

    if args.context_seq_len > 0:
        if args.ppo_update_mode != "sequential":
            warnings.warn(
                "--context_seq_len affects rollout/eval, but PPO random mode still uses "
                "the full forward path unless you use --ppo_update_mode=sequential. "
                "For consistent efficient training, prefer --ppo_update_mode=sequential.",
                UserWarning,
            )
        effective_chunk_steps = args.ppo_minibatch_steps if args.ppo_minibatch_steps > 0 else args.rollout_steps
        if args.context_seq_len < effective_chunk_steps:
            warnings.warn(
                "--context_seq_len is smaller than the effective PPO chunk length. In sequential "
                "chunk mode, only transitions inside the available context window are trained for "
                "each chunk; earlier positions in the chunk are dropped from that gradient step. "
                "For dense chunk losses, set --ppo_minibatch_steps <= --context_seq_len.",
                UserWarning,
            )
        if args.ppo_sequential_loss_scope == "prefix":
            warnings.warn(
                "With --context_seq_len > 0, sequential prefix loss is over the current "
                "episode context window/prefix only. Previous episodes are used as context "
                "summaries, not as loss positions.",
                UserWarning,
            )

    if args.prev_context_window_mode == "random":
        if args.context_seq_len <= 0:
            warnings.warn(
                "--prev_context_window_mode=random has no effect when --context_seq_len=0, "
                "because previous episodes are encoded as full episodes.",
                UserWarning,
            )
        if args.ppo_update_mode != "sequential":
            warnings.warn(
                "--prev_context_window_mode=random is mainly used by sequential PPO/windowed "
                "previous-episode encoding. In random PPO mode, it may have little or no effect.",
                UserWarning,
            )
        warnings.warn(
            "--prev_context_window_mode=random samples random subsequences for previous "
            "episodes during PPO only. Rollout/eval still store and use the observed final "
            "episode embedding as memory.",
            UserWarning,
        )

    if args.detach_context_episodes:
        if args.ppo_update_mode != "sequential":
            warnings.warn(
                "--detach_context_episodes only affects --ppo_update_mode=sequential. "
                "It will be ignored in random PPO mode.",
                UserWarning,
            )
        if args.aggregator_type not in ["mean", "ema", "attn"]:
            warnings.warn(
                "--detach_context_episodes is only implemented for --aggregator_type=mean/ema/attn. "
                "It will be ignored for the slot-specific concat/linear aggregator.",
                UserWarning,
            )
        if args.ppo_sequential_loss_scope == "prefix":
            warnings.warn(
                "--detach_context_episodes with --ppo_sequential_loss_scope=prefix computes "
                "prefix loss only over the current episode prefix. Previous episodes are reused "
                "as detached context features, not trained directly in that minibatch.",
                UserWarning,
            )

    meta_learning = build_metaworld(args)

    envs = gym.vector.SyncVectorEnv(
        [
            make_sampler_env(
                meta_learning.train_classes,
                meta_learning.train_tasks,
                args.seed + i,
                mask_goal=should_mask_goal(args),
            )
            for i in range(args.num_envs)
        ]
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
        ema_beta=args.ema_beta,
        use_state_proj=args.use_state_proj,
        init_type=args.init_type,
        context_seq_len=args.context_seq_len,
        prev_context_window_mode=args.prev_context_window_mode,
        use_context_gate=args.use_context_gate,
        context_gate_hidden_sizes=args.context_gate_hidden_sizes,
        context_gate_init_bias=args.context_gate_init_bias,
        episode_attn_heads=args.episode_attn_heads,
        min_std=args.min_std,
        max_std=args.max_std,
        init_std=args.init_std,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = create_scheduler(optimizer, args)

    out_dir = make_run_dir(args)
    print(f"Run directory: {out_dir}")

    csv_path = os.path.join(out_dir, "metrics.csv")
    eval_metric_keys = []
    for length in args.eval_report_lengths:
        prefix = f"eval_len_{length}"
        eval_metric_keys.extend(
            [
                f"{prefix}_trial_return",
                f"{prefix}_final_return",
                f"{prefix}_final_success",
                f"{prefix}_anysuccess",
                f"{prefix}_mean_ep_success",
            ]
        )

    metrics_header = [
        "update",
        "timestep",
        "rollout_trial_return",
        "rollout_final_return",
        "rollout_final_success",
        "rollout_anysuccess",
        *eval_metric_keys,
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
    ]

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(metrics_header)

    rollout_task_csv_path = os.path.join(out_dir, "rollout_task_class_metrics.csv")
    rollout_task_header = [
        "update",
        "timestep",
        "env_name",
        "num_envs",
        "trial_return",
        "final_return",
        "final_success",
        "anysuccess",
        "mean_ep_success",
    ]
    rollout_task_header.extend([f"episode_{ep + 1}_return" for ep in range(args.trial_length)])
    rollout_task_header.extend([f"episode_{ep + 1}_success" for ep in range(args.trial_length)])
    with open(rollout_task_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(rollout_task_header)

    eval_episode_csv_path = os.path.join(out_dir, "eval_episode_success.csv")
    with open(eval_episode_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "update",
                "timestep",
                "eval_trial_length",
                "eval_round",
                "eval_trial_id",
                "env_index",
                "env_name",
                "episode_index",
                "episode_return",
                "episode_success",
                "cum_anysuccess",
            ]
        )

    raw_obs, _ = envs.reset()
    obs_normalizer.update(raw_obs)
    obs = obs_normalizer.normalize(raw_obs)

    global_episodes = 0
    global_timesteps = 0

    for update in range(args.num_updates):
        t0 = time.time()


        if args.random_task_sample:
            assigned_classes = []
            for env in envs.envs:
                assigned_classes.append(env.sample_new_task())
        else:
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

        rollout_uses_context_window = (args.context_seq_len > 0 and not args.no_rollout_context_seq_len)

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
                b_inputs[:, ep, step, :] = inp_t
                b_states[:, ep, step, :] = obs_t

                context_window = None
                if rollout_uses_context_window:
                    win_start = max(0, step + 1 - args.context_seq_len)
                    context_window = b_inputs[:, ep, win_start : step + 1, :]

                with torch.no_grad():
                    out = model.act_step(
                        inp_t,
                        obs_t,
                        episode_memory,
                        ep,
                        cache_params,
                        context_window_inputs=context_window,
                    )
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

        #################################################################################
        # Diagnostic: check whether global advantage normalization would shrink
        # some task classes relative to others.
        print_advantage_diagnostics_by_class(
            b_adv=b_adv,
            assigned_classes=assigned_classes,
            update=update + 1,
            ppo_epoch=None,
            prefix="ADV_DIAG_PRE_PPO",
        )
        #################################################################################

        ppo_t0 = time.time()
        rollouts = (b_inputs, b_states, b_actions, b_logprobs, b_adv, b_returns)
        loss, p_loss, v_loss, ent = train_ppo(model, optimizer, rollouts, args, device)
        ppo_time = time.time() - ppo_t0

        rollout_trial_return = float(trial_ep_rewards.sum(axis=1).mean())
        rollout_final_return = float(trial_ep_rewards[:, -1].mean())
        rollout_final_success = float(trial_ep_successes[:, -1].mean())
        rollout_anysuccess = float(trial_ep_successes.any(axis=1).mean())

        rollout_task_rows = summarize_rollout_by_task_class(
            assigned_classes,
            trial_ep_rewards,
            trial_ep_successes,
        )
        with open(rollout_task_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for row in rollout_task_rows:
                writer.writerow(
                    [
                        update + 1,
                        global_timesteps,
                        row["env_name"],
                        row["num_envs"],
                        row["trial_return"],
                        row["final_return"],
                        row["final_success"],
                        row["anysuccess"],
                        row["mean_ep_success"],
                        *row["episode_returns"],
                        *row["episode_successes"],
                    ]
                )

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

        if len(rollout_task_rows) <= 12:
            class_summary = " | ".join(
                f"{row['env_name']}: ret={row['trial_return']:.1f}, "
                f"final_succ={100 * row['final_success']:.1f}%"
                for row in rollout_task_rows
            )
            print(f"Rollout by task class | {class_summary}")
        else:
            print(
                f"Rollout task-class metrics saved for {len(rollout_task_rows)} classes "
                f"to {rollout_task_csv_path}"
            )

        eval_summaries = {key: np.nan for key in eval_metric_keys}

        if (update + 1) % args.eval_interval == 0:
            print("policy std:", model.get_policy_std().detach().cpu().numpy())

            eval_summaries, eval_rows = evaluate_meta_learning(
                model,
                meta_learning,
                obs_normalizer,
                args,
                device=device,
            )

            with open(eval_episode_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for row in eval_rows:
                    writer.writerow(
                        [
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
                        ]
                    )

        metrics_row = [
            update + 1,
            global_timesteps,
            rollout_trial_return,
            rollout_final_return,
            rollout_final_success,
            rollout_anysuccess,
            *[eval_summaries.get(key, np.nan) for key in eval_metric_keys],
            loss,
            p_loss,
            v_loss,
            ent,
        ]

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(metrics_row)

        if scheduler is not None:
            scheduler.step()

    torch.save({"model": model.state_dict()}, os.path.join(out_dir, "ttt_ecet_policy.pth"))
    envs.close()


if __name__ == "__main__":
    train()
