import os
import csv
import time
import json
import sys
from datetime import datetime
import random
import warnings
import argparse

import gymnasium as gym
import metaworld
import numpy as np
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F

from ttt import TTTConfig
from agent import TTTEpisodePolicy
from arguments import get_args, parse_hidden_sizes
from utils import (
    MetaWorldTaskSamplerEnv,
    RunningMeanStd,
    RewardNormalizer,
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
    """Return a shuffled list of class names with near-equal counts."""
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
    """Aggregate rollout returns/successes by Meta-World task class."""
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



def parse_probe_args_and_strip_sys_argv():
    """Parse task-ID probe flags while leaving normal PPO flags for get_args().

    This keeps arguments.py unchanged. The script first consumes only probe-specific
    flags, then rewrites sys.argv so the original PPO parser sees the same options
    it already understands.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe_hidden_sizes", type=parse_hidden_sizes, default=(128,128))
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_epochs", type=int, default=20)
    parser.add_argument("--probe_batch_size", type=int, default=64)
    parser.add_argument("--probe_val_fraction", type=float, default=0.25)
    parser.add_argument(
        "--probe_buffer_updates",
        type=int,
        default=1,
        help=(
            "Number of recent PPO rollout batches of detached episode embeddings "
            "to keep for the task-ID probe. 1 means train/evaluate only on the "
            "current rollout representation."
        ),
    )
    parser.add_argument("--probe_interval", type=int, default=1, help="Train/evaluate probe every N PPO updates.")
    parser.add_argument("--probe_weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--probe_reset_each_update",
        action="store_true",
        help=(
            "Reinitialize the probe classifier at each probe interval before fitting on the "
            "current/FIFO detached embeddings. This is stricter but slower/noisier."
        ),
    )
    parser.add_argument(
        "--probe_grad_clip",
        type=float,
        default=10.0,
        help="Max grad norm for the probe classifier. Use <=0 to disable clipping.",
    )

    probe_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if probe_args.probe_buffer_updates <= 0:
        raise ValueError("--probe_buffer_updates must be positive.")
    if probe_args.probe_interval <= 0:
        raise ValueError("--probe_interval must be positive.")
    if not (0.0 < probe_args.probe_val_fraction < 1.0):
        raise ValueError("--probe_val_fraction must be in (0, 1).")
    return probe_args


class TaskIdProbe(nn.Module):
    """Small classifier trained on detached TTT episode embeddings."""

    def __init__(self, input_dim, num_classes, hidden_sizes=(128,)):
        super().__init__()
        layers = []
        prev = int(input_dim)
        for hidden in hidden_sizes:
            hidden = int(hidden)
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden
        layers.append(nn.Linear(prev, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ProbeBuffer:
    """FIFO buffer of detached episode embeddings from recent PPO rollouts."""

    def __init__(self, max_updates=1):
        self.max_updates = int(max_updates)
        self._chunks = []

    def add(self, episode_embeddings, task_ids):
        """Add one PPO rollout batch.

        episode_embeddings: (B,E,H) detached tensor.
        task_ids: (B,) long tensor giving class id per vector env.
        """
        x = episode_embeddings.detach().cpu().float()
        y_env = task_ids.detach().cpu().long()
        if x.ndim != 3:
            raise ValueError(f"episode_embeddings must be (B,E,H), got {tuple(x.shape)}")
        B, E, H = x.shape
        if y_env.shape != (B,):
            raise ValueError(f"task_ids must be (B,), got {tuple(y_env.shape)} for B={B}")
        y = y_env[:, None].expand(B, E).reshape(B * E)
        ep = torch.arange(E, dtype=torch.long)[None, :].expand(B, E).reshape(B * E)
        self._chunks.append((x.reshape(B * E, H), y, ep))
        if len(self._chunks) > self.max_updates:
            self._chunks = self._chunks[-self.max_updates :]

    def tensors(self):
        if not self._chunks:
            raise RuntimeError("ProbeBuffer is empty.")
        x = torch.cat([c[0] for c in self._chunks], dim=0)
        y = torch.cat([c[1] for c in self._chunks], dim=0)
        ep = torch.cat([c[2] for c in self._chunks], dim=0)
        return x, y, ep


def stratified_train_val_split(labels, val_fraction, seed):
    """Return train/val indices with each class represented in validation when possible."""
    labels = labels.detach().cpu().long()
    g = torch.Generator().manual_seed(int(seed))
    train_parts = []
    val_parts = []
    for cls in torch.unique(labels).tolist():
        idx = torch.nonzero(labels == int(cls), as_tuple=False).flatten()
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        if idx.numel() <= 1:
            train_parts.append(idx)
            continue
        n_val = max(1, int(round(float(idx.numel()) * float(val_fraction))))
        n_val = min(n_val, idx.numel() - 1)
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = torch.cat(train_parts, dim=0)
    val_idx = torch.cat(val_parts, dim=0) if val_parts else train_idx[:0]
    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)]
    if val_idx.numel() > 0:
        val_idx = val_idx[torch.randperm(val_idx.numel(), generator=g)]
    return train_idx, val_idx


@torch.no_grad()
def evaluate_probe(probe, x, y, ep_idx, indices, num_episodes, device):
    if indices.numel() == 0:
        return {"loss": float("nan"), "acc": float("nan"), "episode_acc": [float("nan")] * num_episodes}
    probe.eval()
    x_sel = x[indices].to(device=device, dtype=torch.float32)
    y_sel = y[indices].to(device=device, dtype=torch.long)
    logits = probe(x_sel)
    loss = F.cross_entropy(logits, y_sel).item()
    pred = logits.argmax(dim=-1).cpu()
    y_cpu = y_sel.cpu()
    acc = float((pred == y_cpu).float().mean().item())

    episode_acc = []
    ep_cpu = ep_idx[indices].cpu()
    for ep in range(num_episodes):
        mask = ep_cpu == ep
        if not mask.any():
            episode_acc.append(float("nan"))
        else:
            episode_acc.append(float((pred[mask] == y_cpu[mask]).float().mean().item()))
    return {"loss": loss, "acc": acc, "episode_acc": episode_acc}


def train_task_id_probe(probe, optimizer, buffer, probe_args, num_episodes, update, device):
    """Train/evaluate the detached task-ID probe on the current FIFO buffer."""
    x, y, ep_idx = buffer.tensors()
    train_idx, val_idx = stratified_train_val_split(
        y,
        val_fraction=probe_args.probe_val_fraction,
        seed=100_000 + int(update),
    )

    probe.train()
    batch_size = min(int(probe_args.probe_batch_size), max(1, train_idx.numel()))
    g = torch.Generator().manual_seed(200_000 + int(update))
    last_loss = float("nan")

    for _ in range(int(probe_args.probe_epochs)):
        perm = train_idx[torch.randperm(train_idx.numel(), generator=g)]
        for start in range(0, perm.numel(), batch_size):
            idx = perm[start : start + batch_size]
            xb = x[idx].to(device=device, dtype=torch.float32)
            yb = y[idx].to(device=device, dtype=torch.long)
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if probe_args.probe_grad_clip and probe_args.probe_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(probe.parameters(), probe_args.probe_grad_clip)
            optimizer.step()
            last_loss = float(loss.item())

    train_metrics = evaluate_probe(probe, x, y, ep_idx, train_idx, num_episodes, device)
    val_metrics = evaluate_probe(probe, x, y, ep_idx, val_idx, num_episodes, device)
    return {
        "loss": last_loss,
        "train_loss": train_metrics["loss"],
        "train_acc": train_metrics["acc"],
        "val_loss": val_metrics["loss"],
        "val_acc": val_metrics["acc"],
        "val_episode_acc": val_metrics["episode_acc"],
        "n_train": int(train_idx.numel()),
        "n_val": int(val_idx.numel()),
    }


def build_metaworld(args):
    if args.task_set == "ML10":
        return metaworld.ML10()
    if args.task_set == "ML45":
        return metaworld.ML45()
    if args.task_set == "ML1":
        return metaworld.ML1(args.env_name)
    raise ValueError("This trainer is for ML1/ML10/ML45.")


@torch.no_grad()
def evaluate_meta_learning(model, meta_learning, obs_normalizer, args, device=device):
    """Run one max-length eval trial and summarize arbitrary prefixes."""
    classes = meta_learning.test_classes
    tasks = meta_learning.test_tasks
    n = args.eval_num_tasks
    eval_trial_length = args.eval_trial_length
    report_lengths = tuple(args.eval_report_lengths)

    envs = gym.vector.SyncVectorEnv(
        [
            make_sampler_env(
                classes,
                tasks,
                args.seed + 10_000 + i,
                mask_goal=should_mask_goal(args),
            )
            for i in range(n)
        ]
    )

    action_dim = envs.single_action_space.shape[0]
    eval_class_names = list(classes.keys())
    if n < len(eval_class_names):
        raise ValueError(
            f"eval_num_tasks={n} is smaller than number of test classes={len(eval_class_names)}. "
            "Increase --eval_num_tasks if you want every test class represented."
        )

    returns = np.zeros((args.eval_num_trials, n, eval_trial_length), dtype=np.float32)
    successes = np.zeros((args.eval_num_trials, n, eval_trial_length), dtype=bool)
    detailed_rows = []

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
        episode_memory = model.init_episode_memory(n, device=device, num_episodes=eval_trial_length)
        prev_action = np.zeros((n, action_dim), dtype=np.float32)
        prev_reward = np.zeros((n, 1), dtype=np.float32)
        prev_done = np.zeros((n, 1), dtype=np.float32)
        cumulative_any = np.zeros(n, dtype=bool)

        for ep in range(eval_trial_length):
            ep_rewards = np.zeros(n, dtype=np.float32)
            ep_success = np.zeros(n, dtype=bool)
            cache_params = None
            out = None

            for _ in range(args.rollout_steps):
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

            if out is None:
                raise RuntimeError("Evaluation episode produced no rollout steps.")
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


def train():
    probe_args = parse_probe_args_and_strip_sys_argv()
    args = get_args()
    set_seed(args.seed)

    print(f"Using device: {device}")
    print(
        f"Task set: {args.task_set} | envs={args.num_envs} | "
        f"trial_length={args.trial_length} | ep_len={args.rollout_steps}"
    )

    if args.task_set not in ["ML1", "ML10", "ML45"]:
        raise ValueError("This trainer is for ML1/ML10/ML45.")
    if args.ppo_context_episode_sample > 0 and args.aggregator_type not in ["ema", "attn"]:
        warnings.warn(
            "--ppo_context_episode_sample is only supported by --aggregator_type=ema/attn.",
            UserWarning,
        )
    if args.aggregator_type == "ema" and args.context_episode_sample_mode == "uniform":
        warnings.warn(
            "For --aggregator_type=ema, --context_episode_sample_mode=uniform is valid but noisy. "
            "recent or last usually matches recency-weighted EMA better.",
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
    task_class_to_id = {name: idx for idx, name in enumerate(sorted(train_class_names))}

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
        episode_attn_heads=args.episode_attn_heads,
        min_std=args.min_std,
        max_std=args.max_std,
        init_std=args.init_std,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    probe = TaskIdProbe(
        input_dim=args.hidden_size,
        num_classes=len(train_class_names),
        hidden_sizes=probe_args.probe_hidden_sizes,
    ).to(device)
    probe_optimizer = optim.Adam(
        probe.parameters(),
        lr=probe_args.probe_lr,
        weight_decay=probe_args.probe_weight_decay,
    )
    probe_buffer = ProbeBuffer(max_updates=probe_args.probe_buffer_updates)

    out_dir = make_run_dir(args)
    print(f"Run directory: {out_dir}")
    with open(os.path.join(out_dir, "task_probe_config.json"), "w") as f:
        json.dump(vars(probe_args), f, indent=2, sort_keys=True)

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
    rollout_task_header.extend([f"episode_{i + 1}_return" for i in range(args.trial_length)])
    rollout_task_header.extend([f"episode_{i + 1}_success" for i in range(args.trial_length)])
    with open(rollout_task_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(rollout_task_header)

    task_probe_csv_path = os.path.join(out_dir, "task_probe_metrics.csv")
    task_probe_header = [
        "update",
        "timestep",
        "probe_loss",
        "probe_train_loss",
        "probe_train_acc",
        "probe_val_loss",
        "probe_val_acc",
        "probe_n_train",
        "probe_n_val",
    ]
    task_probe_header.extend([f"probe_val_acc_episode_{i + 1}" for i in range(args.trial_length)])
    with open(task_probe_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(task_probe_header)

    raw_obs, _ = envs.reset()
    obs_normalizer.update(raw_obs)
    obs = obs_normalizer.normalize(raw_obs)

    global_episodes = 0
    global_timesteps = 0

    for update in range(args.num_updates):
        t0 = time.time()

        if args.random_task_sample:
            assigned_classes = [env.sample_new_task() for env in envs.envs]
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
        b_inputs = torch.zeros((B, E, T, input_dim), device=device)
        b_states = torch.zeros((B, E, T, obs_dim), device=device)
        b_actions = torch.zeros((B, E, T, action_dim), device=device)
        b_logprobs = torch.zeros((B, E, T), device=device)
        b_rewards = torch.zeros((B, E, T), device=device)
        b_dones = torch.zeros((B, E, T), device=device)
        b_values = torch.zeros((B, E, T), device=device)
        b_valid = torch.zeros((B, E, T), device=device)
        b_task_ids = torch.tensor(
            [task_class_to_id[name] for name in assigned_classes],
            dtype=torch.long,
            device=device,
        )

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
            out = None

            for step in range(T):
                agent_input = get_agent_input(obs, prev_action, prev_reward, prev_done, args.agent_mode)
                inp_t = torch.tensor(agent_input, dtype=torch.float32, device=device)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                b_inputs[:, ep, step, :] = inp_t
                b_states[:, ep, step, :] = obs_t

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

                b_actions[:, ep, step, :] = action
                b_logprobs[:, ep, step] = log_prob
                b_rewards[:, ep, step] = torch.tensor(norm_reward, dtype=torch.float32, device=device)
                b_dones[:, ep, step] = torch.tensor(done, dtype=torch.float32, device=device)
                b_values[:, ep, step] = value
                b_valid[:, ep, step] = 1.0

                cache_params = out.cache_params
                prev_action = action_np
                prev_reward = raw_reward[:, None]
                prev_done = done[:, None].astype(np.float32)
                obs = obs_normalizer.normalize(next_raw_obs)

                if done.all():
                    break

            if out is None:
                raise RuntimeError("Training episode produced no rollout steps.")
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

        # Detached PPO-learned TTT episode representations for the task-ID probe.
        # Shape: (B, E, H), one final TTT hidden state per collected episode.
        probe_episode_embeddings = episode_memory[:, :E, :].detach().clone()

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
        rollouts = (
            b_inputs,
            b_states,
            b_actions,
            b_logprobs,
            b_adv,
            b_returns,
            b_valid,
            b_task_ids,
        )
        loss, p_loss, v_loss, ent = train_ppo(model, optimizer, rollouts, args, device)
        ppo_time = time.time() - ppo_t0

        probe_time = 0.0
        probe_metrics = None
        if (update + 1) % probe_args.probe_interval == 0:
            probe_t0 = time.time()
            if probe_args.probe_reset_each_update:
                probe = TaskIdProbe(
                    input_dim=args.hidden_size,
                    num_classes=len(train_class_names),
                    hidden_sizes=probe_args.probe_hidden_sizes,
                ).to(device)
                probe_optimizer = optim.Adam(
                    probe.parameters(),
                    lr=probe_args.probe_lr,
                    weight_decay=probe_args.probe_weight_decay,
                )
                probe_buffer = ProbeBuffer(max_updates=probe_args.probe_buffer_updates)
            probe_buffer.add(probe_episode_embeddings, b_task_ids)
            probe_metrics = train_task_id_probe(
                probe,
                probe_optimizer,
                probe_buffer,
                probe_args,
                num_episodes=E,
                update=update + 1,
                device=device,
            )
            probe_time = time.time() - probe_t0
            with open(task_probe_csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        update + 1,
                        global_timesteps,
                        probe_metrics["loss"],
                        probe_metrics["train_loss"],
                        probe_metrics["train_acc"],
                        probe_metrics["val_loss"],
                        probe_metrics["val_acc"],
                        probe_metrics["n_train"],
                        probe_metrics["n_val"],
                        *probe_metrics["val_episode_acc"],
                    ]
                )

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
            f"timesteps={global_timesteps} | Data={data_time:.2f}s PPO={ppo_time:.2f}s Probe={probe_time:.2f}s | "
            f"loss={loss:.4f} p={p_loss:.4f} v={v_loss:.4f} ent={ent:.4f} | "
            f"rollout_trial_return={rollout_trial_return:.2f} "
            f"final_return={rollout_final_return:.2f} "
            f"final_success={100 * rollout_final_success:.1f}% "
            f"anysuccess={100 * rollout_anysuccess:.1f}%"
        )

        if probe_metrics is not None:
            print(
                f"Task-ID probe | train_acc={100 * probe_metrics['train_acc']:.1f}% "
                f"val_acc={100 * probe_metrics['val_acc']:.1f}% "
                f"n_train={probe_metrics['n_train']} n_val={probe_metrics['n_val']}"
            )

        if len(rollout_task_rows) <= 12:
            class_summary = " | ".join(
                f"{row['env_name']}: ret={row['trial_return']:.1f}, final_succ={100 * row['final_success']:.1f}%"
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

    torch.save({"model": model.state_dict()}, os.path.join(out_dir, "ttt_ecet_policy.pth"))
    torch.save({"probe": probe.state_dict()}, os.path.join(out_dir, "task_id_probe.pth"))
    envs.close()


if __name__ == "__main__":
    train()
