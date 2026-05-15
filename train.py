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
    ClassRewardNormalizer,
    get_agent_input,
    get_success_array,
    get_qacc_bad_array,
    compute_gae,
    sample_tanh_squashed_action,
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




def _running_mean_std_state(rms):
    """Serialize RunningMeanStd in a checkpoint-friendly format."""
    return {
        "mean": np.asarray(rms.mean, dtype=np.float32).copy(),
        "var": np.asarray(rms.var, dtype=np.float32).copy(),
        "count": float(rms.count),
    }


def _reward_normalizer_state(normalizer):
    """Serialize RewardNormalizer or ClassRewardNormalizer."""
    if isinstance(normalizer, ClassRewardNormalizer):
        return {
            "type": "class",
            "gamma": float(normalizer.gamma),
            "normalizers": {
                str(name): _reward_normalizer_state(sub_normalizer)
                for name, sub_normalizer in normalizer.normalizers.items()
            },
        }

    state = {
        "type": "global",
        "gamma": float(normalizer.gamma),
        "return_rms": _running_mean_std_state(normalizer.return_rms),
        "returns": None,
    }
    if normalizer.returns is not None:
        state["returns"] = np.asarray(normalizer.returns, dtype=np.float32).copy()
    return state


def save_training_checkpoint(path, model, obs_normalizer, reward_normalizer, args):
    """Save model weights plus normalizer states.

    obs_normalizer is also saved as obs_normalizer.npz so analysis scripts can
    load it directly with --obs_norm_npz.
    """
    out_dir = os.path.dirname(path)
    obs_state = _running_mean_std_state(obs_normalizer)
    reward_state = _reward_normalizer_state(reward_normalizer)

    torch.save(
        {
            "model": model.state_dict(),
            "obs_normalizer": obs_state,
            "reward_normalizer": reward_state,
        },
        path,
    )

    np.savez(
        os.path.join(out_dir, "obs_normalizer.npz"),
        mean=obs_state["mean"],
        var=obs_state["var"],
        count=np.asarray(obs_state["count"], dtype=np.float64),
    )
    torch.save(reward_state, os.path.join(out_dir, "reward_normalizer.pt"))

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




def reset_vector_env_indices(envs, indices):
    """Reset selected vector-env indices and return {env_index: raw_obs}.

    For SyncVectorEnv we directly reset envs.envs[i]. For SubprocTaskVectorEnv
    this delegates to reset_indices(...). Reset observations are intentionally
    not used to update obs_normalizer when the reset was caused by QACC/QVEL.
    """
    indices = [int(i) for i in indices]
    if len(indices) == 0:
        return {}

    if hasattr(envs, "reset_indices"):
        return envs.reset_indices(indices)

    reset_obs = {}
    for idx in indices:
        raw_obs, _ = envs.envs[idx].reset()
        reset_obs[idx] = raw_obs
    return reset_obs


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
    """Run one max-length eval trial and summarize arbitrary prefixes.

    When --drop_qacc_trajectories is enabled, eval trajectories that hit
    QACC/QVEL/non-finite-observation instability are reset immediately and
    excluded from eval summaries. This prevents a single corrupted simulator
    state from looking like a sudden policy-performance collapse.
    """
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
    eval_valid = np.ones((args.eval_num_trials, n), dtype=bool)
    eval_qacc_bad_any = np.zeros((args.eval_num_trials, n), dtype=bool)
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
        eval_bad_envs = np.zeros(n, dtype=bool)

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
                if args.squash_actions:
                    action_np = (args.action_scale * torch.tanh(mean)).cpu().numpy().astype(np.float32)
                else:
                    action_np = mean.cpu().numpy()

                if args.drop_qacc_trajectories and eval_bad_envs.any():
                    action_np[eval_bad_envs] = 0.0
                active_before = ~eval_bad_envs if args.drop_qacc_trajectories else np.ones(n, dtype=bool)

                next_raw_obs, reward, terminated, truncated, info = envs.step(action_np)
                done = np.logical_or(terminated, truncated)
                success = get_success_array(info, n)

                step_valid_mask = np.ones(n, dtype=bool)
                qacc_bad = np.zeros(n, dtype=bool)
                bad_now = np.zeros(n, dtype=bool)
                if args.drop_qacc_trajectories:
                    qacc_bad, _qacc_max, _qacc_argmax = get_qacc_bad_array(
                        info,
                        n,
                        qacc_threshold=args.qacc_threshold,
                        qvel_threshold=args.qvel_threshold,
                    )
                    obs_bad = ~np.isfinite(next_raw_obs).all(axis=1)
                    bad_now = active_before & (qacc_bad | obs_bad)
                    step_valid_mask = active_before & ~bad_now

                    if bad_now.any():
                        bad_indices = np.nonzero(bad_now)[0].astype(int).tolist()
                        reset_obs = reset_vector_env_indices(envs, bad_indices)
                        next_raw_obs = np.asarray(next_raw_obs, dtype=np.float32).copy()
                        for env_idx, reset_raw_obs in reset_obs.items():
                            next_raw_obs[env_idx] = reset_raw_obs
                        eval_bad_envs = np.logical_or(eval_bad_envs, bad_now)
                        eval_valid[eval_round, bad_now] = False
                        eval_qacc_bad_any[eval_round, bad_now] = qacc_bad[bad_now]

                    reward = np.asarray(reward, dtype=np.float32).copy()
                    done = np.asarray(done, dtype=bool).copy()
                    success = np.asarray(success, dtype=bool).copy()
                    reward[~step_valid_mask] = 0.0
                    done[eval_bad_envs] = False
                    success[~step_valid_mask] = False

                ep_rewards += np.where(step_valid_mask, reward, 0.0)
                ep_success = np.logical_or(ep_success, success & step_valid_mask)

                cache_params = out.cache_params
                prev_action = action_np
                prev_reward = reward[:, None]
                prev_done = done[:, None].astype(np.float32)
                if args.drop_qacc_trajectories and eval_bad_envs.any():
                    prev_action[eval_bad_envs] = 0.0
                    prev_reward[eval_bad_envs] = 0.0
                    prev_done[eval_bad_envs] = 0.0
                obs = obs_normalizer.normalize(next_raw_obs)

                if args.drop_qacc_trajectories:
                    if np.logical_or(done, eval_bad_envs).all():
                        break
                elif done.all():
                    break

            if out is None:
                raise RuntimeError("Evaluation episode produced no rollout steps.")
            hidden_to_store = out.hidden_states.detach()
            if args.drop_qacc_trajectories and eval_bad_envs.any():
                hidden_to_store = hidden_to_store.clone()
                hidden_to_store[torch.tensor(eval_bad_envs, dtype=torch.bool, device=device)] = 0.0
            episode_memory[:, ep, :] = hidden_to_store
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
                        "eval_valid": int(eval_valid[eval_round, env_idx]),
                        "eval_qacc_bad": int(eval_qacc_bad_any[eval_round, env_idx]),
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
    valid_eval_mask = eval_valid
    invalid_count = int((~valid_eval_mask).sum())
    if args.drop_qacc_trajectories and invalid_count > 0:
        print(f"Eval: excluding {invalid_count}/{valid_eval_mask.size} trajectories with QACC/QVEL/non-finite obs.")

    print("Eval adaptation curve from one max-length rollout:")
    for ep in range(eval_trial_length):
        ep_returns = returns[:, :, ep][valid_eval_mask]
        ep_successes = successes[:, :, ep][valid_eval_mask]
        ep_any = successes[:, :, : ep + 1].any(axis=2)[valid_eval_mask]
        mean_ret = float(ep_returns.mean()) if ep_returns.size > 0 else float("nan")
        mean_succ = float(ep_successes.mean()) if ep_successes.size > 0 else float("nan")
        cum_any = float(ep_any.mean()) if ep_any.size > 0 else float("nan")
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
        trial_returns = prefix_returns.sum(axis=2)[valid_eval_mask]
        summaries[f"{prefix}_trial_return"] = float(trial_returns.mean()) if trial_returns.size > 0 else float("nan")
        valid_final_returns = final_returns[valid_eval_mask]
        summaries[f"{prefix}_final_return"] = (
            float(valid_final_returns.mean()) if valid_final_returns.size > 0 else float("nan")
        )
        valid_final_successes = final_successes[valid_eval_mask]
        summaries[f"{prefix}_final_success"] = (
            float(valid_final_successes.mean()) if valid_final_successes.size > 0 else float("nan")
        )
        valid_any_successes = any_successes[valid_eval_mask]
        summaries[f"{prefix}_anysuccess"] = (
            float(valid_any_successes.mean()) if valid_any_successes.size > 0 else float("nan")
        )
        valid_prefix_successes = prefix_successes[valid_eval_mask]
        summaries[f"{prefix}_mean_ep_success"] = (
            float(valid_prefix_successes.mean()) if valid_prefix_successes.size > 0 else float("nan")
        )

        print(
            f"Eval prefix {length}: "
            f"trial_return={summaries[f'{prefix}_trial_return']:.2f}, "
            f"final_return={summaries[f'{prefix}_final_return']:.2f}, "
            f"final_success={100 * summaries[f'{prefix}_final_success']:.1f}%, "
            f"anysuccess={100 * summaries[f'{prefix}_anysuccess']:.1f}%, "
            f"mean_ep_success={100 * summaries[f'{prefix}_mean_ep_success']:.1f}%"
        )

    return summaries, detailed_rows

def compute_learning_signal_by_class(
    assigned_classes,
    trial_ep_rewards,
    trial_ep_successes,
    b_adv,
    b_returns,
    b_values,
    b_logprobs,
    b_valid=None,
):
    """Summarize per-task PPO learning signal for one collected rollout.

    assigned_classes: list[str], length B
    trial_ep_rewards: np.ndarray (B, E)
    trial_ep_successes: np.ndarray (B, E)
    b_adv, b_returns, b_values, b_logprobs: torch tensors (B, E, T)
    b_valid: optional torch tensor (B, E, T), 1 for real rollout positions
    """
    rows = []

    with torch.no_grad():
        adv = b_adv.detach().float().cpu()
        ret = b_returns.detach().float().cpu()
        val = b_values.detach().float().cpu()
        logp = b_logprobs.detach().float().cpu()

        if b_valid is None:
            valid = torch.ones_like(adv, dtype=torch.bool)
        else:
            valid = b_valid.detach().cpu().bool()

        unique_classes = sorted(set(assigned_classes))

        for cls in unique_classes:
            env_idx = [i for i, name in enumerate(assigned_classes) if name == cls]
            if len(env_idx) == 0:
                continue

            cls_mask = valid[env_idx]
            cls_adv = adv[env_idx][cls_mask]
            cls_ret = ret[env_idx][cls_mask]
            cls_val = val[env_idx][cls_mask]
            cls_logp = logp[env_idx][cls_mask]

            if cls_adv.numel() == 0:
                continue

            value_err = cls_ret - cls_val
            value_mse = (value_err ** 2).mean()

            ret_var = cls_ret.var(unbiased=False)
            err_var = value_err.var(unbiased=False)
            if ret_var.item() > 1e-8:
                explained_var = 1.0 - (err_var / ret_var)
                explained_var = float(explained_var.item())
            else:
                explained_var = float("nan")

            rows.append(
                {
                    "env_name": cls,
                    "num_envs": len(env_idx),
                    "num_samples": int(cls_adv.numel()),
                    "trial_return": float(trial_ep_rewards[env_idx].sum(axis=1).mean()),
                    "final_return": float(trial_ep_rewards[env_idx, -1].mean()),
                    "final_success": float(trial_ep_successes[env_idx, -1].mean()),
                    "anysuccess": float(trial_ep_successes[env_idx].any(axis=1).mean()),
                    "mean_ep_success": float(trial_ep_successes[env_idx].mean()),
                    "adv_mean": float(cls_adv.mean().item()),
                    "adv_std": float(cls_adv.std(unbiased=False).item()),
                    "adv_abs_mean": float(cls_adv.abs().mean().item()),
                    "adv_min": float(cls_adv.min().item()),
                    "adv_max": float(cls_adv.max().item()),
                    "positive_adv_frac": float((cls_adv > 0).float().mean().item()),
                    "negative_adv_frac": float((cls_adv < 0).float().mean().item()),
                    "return_mean": float(cls_ret.mean().item()),
                    "return_std": float(cls_ret.std(unbiased=False).item()),
                    "value_mean": float(cls_val.mean().item()),
                    "value_std": float(cls_val.std(unbiased=False).item()),
                    "value_mse": float(value_mse.item()),
                    "value_abs_error": float(value_err.abs().mean().item()),
                    "explained_variance": explained_var,
                    "logprob_mean": float(cls_logp.mean().item()),
                    "logprob_std": float(cls_logp.std(unbiased=False).item()),
                }
            )

    return rows

def train():
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
    id_to_task_class = {idx: name for name, idx in task_class_to_id.items()}

    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    input_dim = obs_dim
    if args.agent_mode == "agent_v2":
        input_dim = obs_dim + action_dim + 1
    elif args.agent_mode == "agent_rl2":
        input_dim = obs_dim + action_dim + 2

    obs_normalizer = RunningMeanStd(shape=(obs_dim,))
    if args.normalize_reward_by_class:
        reward_normalizer = ClassRewardNormalizer(gamma=args.gamma)
    else:
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

    rollout_signal_csv_path = None
    if args.log_rollout_learning_signal:
        rollout_signal_csv_path = os.path.join(out_dir, "rollout_task_learning_signal.csv")
        rollout_signal_header = [
            "update",
            "timestep",
            "env_name",
            "num_envs",
            "num_samples",
            "trial_return",
            "final_return",
            "final_success",
            "anysuccess",
            "mean_ep_success",
            "adv_mean",
            "adv_std",
            "adv_abs_mean",
            "adv_min",
            "adv_max",
            "positive_adv_frac",
            "negative_adv_frac",
            "return_mean",
            "return_std",
            "value_mean",
            "value_std",
            "value_mse",
            "value_abs_error",
            "explained_variance",
            "logprob_mean",
            "logprob_std",
        ]

        with open(rollout_signal_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(rollout_signal_header)

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
                "eval_valid",
                "eval_qacc_bad",
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

    ppo_class_csv_path = None
    if args.log_ppo_class_diagnostics:
        ppo_class_csv_path = os.path.join(out_dir, "ppo_task_class_diagnostics.csv")
        ppo_class_header = [
            "update",
            "timestep",
            "env_name",
            "task_id",
            "num_records",
            "num_samples",
            "ppo_loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "grad_norm_mean",
            "grad_norm_max",
        ]
        with open(ppo_class_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(ppo_class_header)

    qacc_drop_csv_path = None
    if args.drop_qacc_trajectories:
        qacc_drop_csv_path = os.path.join(out_dir, "dropped_qacc_trajectories.csv")
        with open(qacc_drop_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "update",
                    "timestep",
                    "num_dropped",
                    "num_kept",
                    "dropped_env_indices",
                    "dropped_env_names",
                    "qacc_max_abs",
                    "qacc_argmax",
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
        qacc_bad_envs = np.zeros(B, dtype=bool)
        qacc_bad_max = np.zeros(B, dtype=np.float64)
        qacc_bad_argmax = np.full(B, -1, dtype=np.int64)

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
                    if args.squash_actions:
                        env_action, log_prob, action = sample_tanh_squashed_action(
                            mean,
                            log_std,
                            eps=args.squash_logprob_eps,
                            action_scale=args.action_scale,
                        )
                    else:
                        dist = torch.distributions.Normal(mean, log_std.exp())
                        action = dist.sample()
                        env_action = action
                        log_prob = dist.log_prob(action).sum(dim=-1)
                    value = out.value

                action_np = env_action.cpu().numpy().astype(np.float32)
                if args.drop_qacc_trajectories and qacc_bad_envs.any():
                    # Already-dropped envs are kept alive only to satisfy the vector env.
                    # Send zero actions so a corrupted/dropped env cannot keep slamming contacts.
                    action_np[qacc_bad_envs] = 0.0

                active_before = ~qacc_bad_envs if args.drop_qacc_trajectories else np.ones(B, dtype=bool)
                next_raw_obs, raw_reward, terminated, truncated, info = envs.step(action_np)
                global_timesteps += B
                done = np.logical_or(terminated, truncated)
                success = get_success_array(info, B)

                step_valid_mask = np.ones(B, dtype=bool)
                if args.drop_qacc_trajectories:
                    qacc_bad, qacc_max, qacc_argmax = get_qacc_bad_array(
                        info,
                        B,
                        qacc_threshold=args.qacc_threshold,
                        qvel_threshold=args.qvel_threshold,
                    )
                    obs_bad = ~np.isfinite(next_raw_obs).all(axis=1)
                    bad_now = active_before & (qacc_bad | obs_bad)
                    qacc_bad_envs = np.logical_or(qacc_bad_envs, bad_now)
                    step_valid_mask = active_before & ~bad_now

                    qacc_max_clean = np.nan_to_num(qacc_max, nan=0.0, posinf=np.inf, neginf=np.inf)
                    qacc_bad_max = np.maximum(qacc_bad_max, qacc_max_clean)
                    qacc_bad_argmax = np.where(qacc_bad | obs_bad, qacc_argmax, qacc_bad_argmax)

                    if bad_now.any():
                        bad_indices = np.nonzero(bad_now)[0].astype(int).tolist()
                        reset_obs = reset_vector_env_indices(envs, bad_indices)
                        next_raw_obs = np.asarray(next_raw_obs, dtype=np.float32).copy()
                        for env_idx, reset_raw_obs in reset_obs.items():
                            next_raw_obs[env_idx] = reset_raw_obs

                        # The bad transition is not a training sample and should not
                        # poison rewards, dones, success metrics, or prev_reward.
                        raw_reward = np.asarray(raw_reward, dtype=np.float32).copy()
                        done = np.asarray(done, dtype=bool).copy()
                        success = np.asarray(success, dtype=bool).copy()
                        raw_reward[bad_now] = 0.0
                        done[bad_now] = False
                        success[bad_now] = False

                ep_rewards += np.where(step_valid_mask, raw_reward, 0.0)
                ep_success = np.logical_or(ep_success, success & step_valid_mask)

                if args.drop_qacc_trajectories:
                    if step_valid_mask.any():
                        obs_normalizer.update(next_raw_obs[step_valid_mask])
                    if args.normalize_reward_by_class:
                        norm_reward = reward_normalizer.normalize_masked(
                            raw_reward,
                            done.astype(np.float32),
                            assigned_classes,
                            step_valid_mask,
                        )
                    else:
                        norm_reward = reward_normalizer.normalize_masked(
                            raw_reward,
                            done.astype(np.float32),
                            step_valid_mask,
                        )
                else:
                    obs_normalizer.update(next_raw_obs)
                    if args.normalize_reward_by_class:
                        norm_reward = reward_normalizer.normalize(
                            raw_reward,
                            done.astype(np.float32),
                            assigned_classes,
                        )
                    else:
                        norm_reward = reward_normalizer.normalize(raw_reward, done.astype(np.float32))

                b_actions[:, ep, step, :] = action
                b_logprobs[:, ep, step] = log_prob
                b_rewards[:, ep, step] = torch.tensor(norm_reward, dtype=torch.float32, device=device)
                b_dones[:, ep, step] = torch.tensor(done, dtype=torch.float32, device=device)
                b_values[:, ep, step] = value
                b_valid[:, ep, step] = torch.tensor(step_valid_mask.astype(np.float32), dtype=torch.float32, device=device)

                cache_params = out.cache_params
                prev_action = action_np
                prev_reward = raw_reward[:, None]
                prev_done = done[:, None].astype(np.float32)
                if args.drop_qacc_trajectories and qacc_bad_envs.any():
                    prev_action[qacc_bad_envs] = 0.0
                    prev_reward[qacc_bad_envs] = 0.0
                    prev_done[qacc_bad_envs] = 0.0
                obs = obs_normalizer.normalize(next_raw_obs)

                if args.drop_qacc_trajectories:
                    if np.logical_or(done, qacc_bad_envs).all():
                        break
                elif done.all():
                    break

            if out is None:
                raise RuntimeError("Training episode produced no rollout steps.")
            hidden_to_store = out.hidden_states.detach()
            if args.drop_qacc_trajectories and qacc_bad_envs.any():
                hidden_to_store = hidden_to_store.clone()
                hidden_to_store[torch.tensor(qacc_bad_envs, dtype=torch.bool, device=device)] = 0.0
            episode_memory[:, ep, :] = hidden_to_store
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

        train_env_mask_np = np.ones(B, dtype=bool)
        if args.drop_qacc_trajectories:
            train_env_mask_np = ~qacc_bad_envs
            num_dropped = int(qacc_bad_envs.sum())
            if num_dropped > 0:
                dropped_indices = np.nonzero(qacc_bad_envs)[0].astype(int).tolist()
                dropped_names = [assigned_classes[i] for i in dropped_indices]
                kept = int(train_env_mask_np.sum())
                print(
                    f"Dropping {num_dropped}/{B} env trajectories from PPO because of QACC/QVEL instability: "
                    f"{dropped_names}"
                )
                with open(qacc_drop_csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [
                            update + 1,
                            global_timesteps,
                            num_dropped,
                            kept,
                            json.dumps(dropped_indices),
                            json.dumps(dropped_names),
                            json.dumps([float(qacc_bad_max[i]) for i in dropped_indices]),
                            json.dumps([int(qacc_bad_argmax[i]) for i in dropped_indices]),
                        ]
                    )

        train_env_mask = torch.tensor(train_env_mask_np, dtype=torch.bool, device=device)
        train_B = int(train_env_mask_np.sum())

        if train_B > 0:
            b_inputs_train = b_inputs[train_env_mask]
            b_states_train = b_states[train_env_mask]
            b_actions_train = b_actions[train_env_mask]
            b_logprobs_train = b_logprobs[train_env_mask]
            b_rewards_train = b_rewards[train_env_mask]
            b_dones_train = b_dones[train_env_mask]
            b_values_train = b_values[train_env_mask]
            b_valid_train = b_valid[train_env_mask]
            b_task_ids_train = b_task_ids[train_env_mask]

            with torch.no_grad():
                next_value = torch.zeros(train_B, device=device)
            b_adv, b_returns = compute_gae(
                b_rewards_train,
                b_values_train,
                b_dones_train,
                next_value,
                args.gamma,
                args.gae_lambda,
            )

            train_assigned_classes = [assigned_classes[i] for i, keep in enumerate(train_env_mask_np) if keep]
            train_trial_ep_rewards = trial_ep_rewards[train_env_mask_np]
            train_trial_ep_successes = trial_ep_successes[train_env_mask_np]
        else:
            b_adv = torch.zeros((0, E, T), device=device)
            b_returns = torch.zeros((0, E, T), device=device)
            b_inputs_train = b_inputs[:0]
            b_states_train = b_states[:0]
            b_actions_train = b_actions[:0]
            b_logprobs_train = b_logprobs[:0]
            b_valid_train = b_valid[:0]
            b_task_ids_train = b_task_ids[:0]
            train_assigned_classes = []
            train_trial_ep_rewards = trial_ep_rewards[:0]
            train_trial_ep_successes = trial_ep_successes[:0]

        if args.log_rollout_learning_signal and train_B > 0:
            signal_rows = compute_learning_signal_by_class(
                assigned_classes=train_assigned_classes,
                trial_ep_rewards=train_trial_ep_rewards,
                trial_ep_successes=train_trial_ep_successes,
                b_adv=b_adv,
                b_returns=b_returns,
                b_values=b_values_train,
                b_logprobs=b_logprobs_train,
                b_valid=b_valid_train,
            )

            with open(rollout_signal_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for row in signal_rows:
                    writer.writerow(
                        [
                            update + 1,
                            global_timesteps,
                            row["env_name"],
                            row["num_envs"],
                            row["num_samples"],
                            row["trial_return"],
                            row["final_return"],
                            row["final_success"],
                            row["anysuccess"],
                            row["mean_ep_success"],
                            row["adv_mean"],
                            row["adv_std"],
                            row["adv_abs_mean"],
                            row["adv_min"],
                            row["adv_max"],
                            row["positive_adv_frac"],
                            row["negative_adv_frac"],
                            row["return_mean"],
                            row["return_std"],
                            row["value_mean"],
                            row["value_std"],
                            row["value_mse"],
                            row["value_abs_error"],
                            row["explained_variance"],
                            row["logprob_mean"],
                            row["logprob_std"],
                        ]
                    )

            print("Learning signal by task:")
            for row in signal_rows:
                print(
                    f"  {row['env_name']:<28s} "
                    f"succ={100 * row['final_success']:5.1f}% "
                    f"adv_std={row['adv_std']:.3f} "
                    f"adv_abs={row['adv_abs_mean']:.3f} "
                    f"v_mse={row['value_mse']:.3f} "
                    f"ev={row['explained_variance']:.3f}"
                )
        elif args.log_rollout_learning_signal:
            print("Learning signal by task: skipped because all env trajectories were dropped.")

        ppo_t0 = time.time()
        if train_B > 0:
            rollouts = (
                b_inputs_train,
                b_states_train,
                b_actions_train,
                b_logprobs_train,
                b_adv,
                b_returns,
                b_valid_train,
                b_task_ids_train,
            )
            loss, p_loss, v_loss, ent, ppo_class_rows = train_ppo(model, optimizer, rollouts, args, device)
        else:
            print("Skipping PPO update because all env trajectories were dropped by --drop_qacc_trajectories.")
            loss = float("nan")
            p_loss = float("nan")
            v_loss = float("nan")
            ent = float("nan")
            ppo_class_rows = []
        ppo_time = time.time() - ppo_t0

        if args.log_ppo_class_diagnostics:
            with open(ppo_class_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for row in ppo_class_rows:
                    task_id = int(row["task_id"])
                    writer.writerow(
                        [
                            update + 1,
                            global_timesteps,
                            id_to_task_class.get(task_id, f"task_{task_id}"),
                            task_id,
                            row["num_records"],
                            row["num_samples"],
                            row["loss"],
                            row["policy_loss"],
                            row["value_loss"],
                            row["entropy"],
                            row["grad_norm_mean"],
                            row["grad_norm_max"],
                        ]
                    )

            if len(ppo_class_rows) <= 12:
                print("PPO loss/grad by task class:")
                for row in ppo_class_rows:
                    task_id = int(row["task_id"])
                    print(
                        f"  {id_to_task_class.get(task_id, f'task_{task_id}'):<28s} "
                        f"loss={row['loss']:+.4f} "
                        f"p={row['policy_loss']:+.4f} "
                        f"v={row['value_loss']:.4f} "
                        f"ent={row['entropy']:.4f} "
                        f"grad={row['grad_norm_mean']:.3f} "
                        f"grad_max={row['grad_norm_max']:.3f}"
                    )
            else:
                print(f"PPO class diagnostics saved for {len(ppo_class_rows)} classes to {ppo_class_csv_path}")

        if train_B > 0:
            rollout_trial_return = float(train_trial_ep_rewards.sum(axis=1).mean())
            rollout_final_return = float(train_trial_ep_rewards[:, -1].mean())
            rollout_final_success = float(train_trial_ep_successes[:, -1].mean())
            rollout_anysuccess = float(train_trial_ep_successes.any(axis=1).mean())
            rollout_task_rows = summarize_rollout_by_task_class(
                train_assigned_classes,
                train_trial_ep_rewards,
                train_trial_ep_successes,
            )
        else:
            rollout_trial_return = float("nan")
            rollout_final_return = float("nan")
            rollout_final_success = float("nan")
            rollout_anysuccess = float("nan")
            rollout_task_rows = []
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
            f"timesteps={global_timesteps} | Data={data_time:.2f}s PPO={ppo_time:.2f}s | "
            f"loss={loss:.4f} p={p_loss:.4f} v={v_loss:.4f} ent={ent:.4f} | "
            f"rollout_trial_return={rollout_trial_return:.2f} "
            f"final_return={rollout_final_return:.2f} "
            f"final_success={100 * rollout_final_success:.1f}% "
            f"anysuccess={100 * rollout_anysuccess:.1f}%"
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
                            row.get("eval_valid", 1),
                            row.get("eval_qacc_bad", 0),
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

    save_training_checkpoint(
        os.path.join(out_dir, "ttt_ecet_policy.pth"),
        model,
        obs_normalizer,
        reward_normalizer,
        args,
    )
    envs.close()


if __name__ == "__main__":
    train()
