import os
import csv
import time
import json
import sys
from datetime import datetime
import random
import warnings
import multiprocessing as mp
import traceback

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




def save_action_histograms(
    out_dir,
    csv_path,
    update,
    timestep,
    env_actions,
    valid_mask,
    bins=80,
    hist_range=(-1.0, 1.0),
):
    """Save per-update histograms of executed env actions.

    env_actions: np.ndarray or torch tensor, shape (B, E, T, action_dim)
        The actual action sent to env.step(...). For squashed policies this is
        action_scale * tanh(raw_action), not the pre-tanh action.
    valid_mask: np.ndarray or torch tensor, shape (B, E, T)
        1 for collected rollout positions.
    """
    if torch.is_tensor(env_actions):
        env_actions = env_actions.detach().cpu().numpy()
    if torch.is_tensor(valid_mask):
        valid_mask = valid_mask.detach().cpu().numpy()

    env_actions = np.asarray(env_actions, dtype=np.float32)
    valid_mask = np.asarray(valid_mask).astype(bool)

    if env_actions.ndim != 4:
        raise ValueError(f"env_actions must have shape (B,E,T,D), got {env_actions.shape}")
    if valid_mask.shape != env_actions.shape[:3]:
        raise ValueError(f"valid_mask shape {valid_mask.shape} does not match env_actions {env_actions.shape[:3]}")

    action_dim = env_actions.shape[-1]
    flat_actions = env_actions[valid_mask]  # (N, action_dim)
    if flat_actions.size == 0:
        return

    hist_dir = os.path.join(out_dir, "action_histograms")
    os.makedirs(hist_dir, exist_ok=True)

    bin_edges = np.linspace(float(hist_range[0]), float(hist_range[1]), int(bins) + 1)
    counts = np.zeros((action_dim, int(bins)), dtype=np.int64)
    density = np.zeros((action_dim, int(bins)), dtype=np.float64)

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "update",
                    "timestep",
                    "action_dim",
                    "bin_left",
                    "bin_right",
                    "count",
                    "density",
                    "num_samples",
                    "mean",
                    "std",
                    "min",
                    "max",
                    "abs_mean",
                    "abs_max",
                    "below_hist_min_frac",
                    "above_hist_max_frac",
                    "sat_095_frac",
                    "sat_099_frac",
                ]
            )

        for dim in range(action_dim):
            x = flat_actions[:, dim].astype(np.float64)
            finite = np.isfinite(x)
            x = x[finite]
            if x.size == 0:
                continue

            c, _ = np.histogram(x, bins=bin_edges)
            d, _ = np.histogram(x, bins=bin_edges, density=True)
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

            counts[dim] = c
            density[dim] = d

            num_samples = int(x.size)
            mean = float(x.mean())
            std = float(x.std())
            xmin = float(x.min())
            xmax = float(x.max())
            abs_x = np.abs(x)
            abs_mean = float(abs_x.mean())
            abs_max = float(abs_x.max())
            below_frac = float(np.mean(x < hist_range[0]))
            above_frac = float(np.mean(x > hist_range[1]))
            sat_095_frac = float(np.mean(abs_x > 0.95))
            sat_099_frac = float(np.mean(abs_x > 0.99))

            for i in range(int(bins)):
                writer.writerow(
                    [
                        update,
                        timestep,
                        dim,
                        float(bin_edges[i]),
                        float(bin_edges[i + 1]),
                        int(c[i]),
                        float(d[i]),
                        num_samples,
                        mean,
                        std,
                        xmin,
                        xmax,
                        abs_mean,
                        abs_max,
                        below_frac,
                        above_frac,
                        sat_095_frac,
                        sat_099_frac,
                    ]
                )

    np.savez_compressed(
        os.path.join(hist_dir, f"action_hist_update_{int(update):04d}.npz"),
        update=np.asarray(update, dtype=np.int64),
        timestep=np.asarray(timestep, dtype=np.int64),
        bin_edges=bin_edges,
        counts=counts,
        density=density,
        hist_range=np.asarray(hist_range, dtype=np.float64),
    )

    # Best-effort plot. If matplotlib is not available on the HPC node, the CSV
    # and NPZ histogram data are still saved.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ncols = 2
        nrows = int(np.ceil(action_dim / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        width = bin_edges[1] - bin_edges[0]

        for dim in range(action_dim):
            ax = axes[dim // ncols][dim % ncols]
            ax.bar(centers, counts[dim], width=width, align="center")
            ax.set_title(f"action dim {dim}")
            ax.set_xlabel("executed env action")
            ax.set_ylabel("count")
            ax.set_xlim(hist_range[0], hist_range[1])
            ax.grid(True, alpha=0.25)

        for dim in range(action_dim, nrows * ncols):
            axes[dim // ncols][dim % ncols].axis("off")

        fig.suptitle(f"Executed action histograms | update {update} | timestep {timestep}")
        fig.tight_layout()
        fig.savefig(os.path.join(hist_dir, f"action_hist_update_{int(update):04d}.png"), dpi=160)
        plt.close(fig)
    except Exception as exc:
        if update == 1:
            print(f"Action histogram plotting disabled: {exc}")


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



def _combine_vector_infos(info_list):
    """Convert a list of per-env info dicts into a vector-style info dict."""
    if not info_list:
        return {}
    keys = set()
    for info in info_list:
        keys.update(info.keys())
    out = {}
    for key in keys:
        values = [info.get(key) for info in info_list]
        try:
            out[key] = np.asarray(values)
        except Exception:
            out[key] = values
    return out


def _env_worker(remote, parent_remote, classes, tasks, base_seed, mask_goal, env_indices):
    """Worker process for a shard of MetaWorldTaskSamplerEnv instances."""
    parent_remote.close()
    envs = []
    try:
        # Avoid accidental CPU oversubscription in each environment worker.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        envs = [
            MetaWorldTaskSamplerEnv(classes, tasks, seed=base_seed + int(env_idx), mask_goal=mask_goal)
            for env_idx in env_indices
        ]

        while True:
            cmd, data = remote.recv()
            if cmd == "get_spaces":
                env0 = envs[0]
                remote.send((env0.action_space, env0.observation_space))

            elif cmd == "sample":
                # data is a list of env_name/None values, one per local env.
                if len(data) != len(envs):
                    raise ValueError(f"Worker expected {len(envs)} task assignments, got {len(data)}")
                labels = [env.sample_new_task(env_name=env_name) for env, env_name in zip(envs, data)]
                remote.send(labels)

            elif cmd == "reset":
                obs_list = []
                info_list = []
                for env in envs:
                    obs, info = env.reset()
                    obs_list.append(obs)
                    info_list.append(info)
                remote.send((np.stack(obs_list, axis=0), info_list))

            elif cmd == "step":
                # data is an action array with shape (local_n, action_dim).
                if len(data) != len(envs):
                    raise ValueError(f"Worker expected {len(envs)} actions, got {len(data)}")
                obs_list = []
                rewards = []
                terminated = []
                truncated = []
                info_list = []
                for env, action in zip(envs, data):
                    obs, reward, term, trunc, info = env.step(action)
                    obs_list.append(obs)
                    rewards.append(reward)
                    terminated.append(term)
                    truncated.append(trunc)
                    info_list.append(info)
                remote.send(
                    (
                        np.stack(obs_list, axis=0),
                        np.asarray(rewards, dtype=np.float32),
                        np.asarray(terminated, dtype=bool),
                        np.asarray(truncated, dtype=bool),
                        info_list,
                    )
                )

            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.close()
                break

            else:
                raise RuntimeError(f"Unknown worker command: {cmd}")

    except KeyboardInterrupt:
        pass
    except Exception:
        tb = traceback.format_exc()
        try:
            remote.send(("__error__", tb))
        except Exception:
            pass
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass


class SubprocTaskVectorEnv:
    """Vector env that runs N envs across a smaller number of worker processes.

    Each worker owns a shard/batch of environments and steps those local envs
    serially. Different workers step in parallel. This is usually better on HPC
    than one process per env because it avoids spawning 50/100 Python processes
    and gives you explicit control over CPU usage.
    """

    def __init__(
        self,
        classes,
        tasks,
        num_envs,
        base_seed,
        mask_goal=True,
        num_workers=0,
        start_method="fork",
    ):
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_workers = self._resolve_num_workers(num_workers, self.num_envs)
        self.closed = False
        self.ctx = mp.get_context(start_method)

        # Split global env indices into contiguous shards. Keeping order makes
        # outputs align exactly with the input action/task-assignment arrays.
        all_indices = np.arange(self.num_envs)
        self.worker_indices = [chunk.astype(int).tolist() for chunk in np.array_split(all_indices, self.num_workers)]
        self.worker_indices = [chunk for chunk in self.worker_indices if len(chunk) > 0]
        self.num_workers = len(self.worker_indices)

        self.remotes, self.work_remotes = zip(*[self.ctx.Pipe() for _ in range(self.num_workers)])
        self.processes = []
        for remote, work_remote, env_indices in zip(self.remotes, self.work_remotes, self.worker_indices):
            proc = self.ctx.Process(
                target=_env_worker,
                args=(work_remote, remote, classes, tasks, base_seed, mask_goal, env_indices),
                daemon=True,
            )
            proc.start()
            work_remote.close()
            self.processes.append(proc)

        self.remotes[0].send(("get_spaces", None))
        action_space, observation_space = self._recv(self.remotes[0])
        self.single_action_space = action_space
        self.single_observation_space = observation_space

    @staticmethod
    def _resolve_num_workers(num_workers, num_envs):
        requested = int(num_workers or 0)
        if requested <= 0:
            cpu_count = os.cpu_count() or 1
            slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
            if slurm_cpus:
                try:
                    cpu_count = int(slurm_cpus)
                except ValueError:
                    pass
            requested = cpu_count
        return max(1, min(int(num_envs), requested))

    def _recv(self, remote):
        msg = remote.recv()
        if (
            isinstance(msg, tuple)
            and len(msg) == 2
            and isinstance(msg[0], str)
            and msg[0] == "__error__"
        ):
            raise RuntimeError("Environment worker failed:\n" + msg[1])
        return msg

    def _shard_sequence(self, values):
        return [[values[i] for i in env_indices] for env_indices in self.worker_indices]

    def sample_new_tasks(self, assigned_classes=None):
        """Sample/switch one task per env.

        assigned_classes may be None, a list of class names, or a list containing
        None entries. None means sample a class uniformly inside the env wrapper.
        Returns a flat list of actual class names in env order.
        """
        if assigned_classes is None:
            assigned_classes = [None] * self.num_envs
        if len(assigned_classes) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} assigned classes, got {len(assigned_classes)}")

        for remote, shard in zip(self.remotes, self._shard_sequence(assigned_classes)):
            remote.send(("sample", shard))
        shard_labels = [self._recv(remote) for remote in self.remotes]

        labels = [None] * self.num_envs
        for env_indices, local_labels in zip(self.worker_indices, shard_labels):
            for env_idx, label in zip(env_indices, local_labels):
                labels[env_idx] = label
        return labels

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [self._recv(remote) for remote in self.remotes]

        obs_parts = []
        info_list = []
        for obs, local_infos in results:
            obs_parts.append(obs)
            info_list.extend(local_infos)
        return np.concatenate(obs_parts, axis=0), _combine_vector_infos(info_list)

    def step(self, actions):
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected actions for {self.num_envs} envs, got {len(actions)}")
        for remote, env_indices in zip(self.remotes, self.worker_indices):
            remote.send(("step", actions[env_indices]))
        results = [self._recv(remote) for remote in self.remotes]

        obs_parts = []
        reward_parts = []
        terminated_parts = []
        truncated_parts = []
        info_list = []
        for obs, rewards, terminated, truncated, local_infos in results:
            obs_parts.append(obs)
            reward_parts.append(rewards)
            terminated_parts.append(terminated)
            truncated_parts.append(truncated)
            info_list.extend(local_infos)

        return (
            np.concatenate(obs_parts, axis=0),
            np.concatenate(reward_parts, axis=0).astype(np.float32),
            np.concatenate(terminated_parts, axis=0).astype(bool),
            np.concatenate(truncated_parts, axis=0).astype(bool),
            _combine_vector_infos(info_list),
        )

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for proc in self.processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self.closed = True

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
    """Run one max-length eval trial and summarize arbitrary prefixes."""
    classes = meta_learning.test_classes
    tasks = meta_learning.test_tasks
    n = args.eval_num_tasks
    eval_trial_length = args.eval_trial_length
    report_lengths = tuple(args.eval_report_lengths)

    envs = SubprocTaskVectorEnv(
        classes,
        tasks,
        num_envs=n,
        base_seed=args.seed + 10_000,
        mask_goal=should_mask_goal(args),
        num_workers=getattr(args, "eval_num_env_workers", 0),
        start_method=getattr(args, "env_start_method", "fork"),
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

        envs.sample_new_tasks(assigned_classes)

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
                if args.squash_actions:
                    action_np = (args.action_scale * torch.tanh(mean)).cpu().numpy().astype(np.float32)
                else:
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
    print(
        f"Env workers: train={getattr(args, 'num_env_workers', 0) or 'auto'} | "
        f"eval={getattr(args, 'eval_num_env_workers', 0) or 'auto'} | "
        f"start_method={getattr(args, 'env_start_method', 'fork')}"
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
    envs = SubprocTaskVectorEnv(
        meta_learning.train_classes,
        meta_learning.train_tasks,
        num_envs=args.num_envs,
        base_seed=args.seed,
        mask_goal=should_mask_goal(args),
        num_workers=getattr(args, "num_env_workers", 0),
        start_method=getattr(args, "env_start_method", "fork"),
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

    action_hist_csv_path = os.path.join(out_dir, "action_histograms.csv")

    raw_obs, _ = envs.reset()
    obs_normalizer.update(raw_obs)
    obs = obs_normalizer.normalize(raw_obs)

    global_episodes = 0
    global_timesteps = 0

    for update in range(args.num_updates):
        t0 = time.time()

        if args.random_task_sample:
            assigned_classes = envs.sample_new_tasks(None)
        else:
            assigned_classes = make_balanced_class_assignment(
                train_class_names,
                args.num_envs,
                task_assignment_rng,
            )
            assigned_classes = envs.sample_new_tasks(assigned_classes)

        raw_obs, _ = envs.reset()
        obs_normalizer.update(raw_obs)
        obs = obs_normalizer.normalize(raw_obs)

        B, E, T = args.num_envs, args.trial_length, args.rollout_steps
        b_inputs = torch.zeros((B, E, T, input_dim), device=device)
        b_states = torch.zeros((B, E, T, obs_dim), device=device)
        b_actions = torch.zeros((B, E, T, action_dim), device=device)
        b_env_actions = torch.zeros((B, E, T, action_dim), device=device)
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
                next_raw_obs, raw_reward, terminated, truncated, info = envs.step(action_np)
                global_timesteps += B
                done = np.logical_or(terminated, truncated)
                success = get_success_array(info, B)

                ep_rewards += raw_reward
                ep_success = np.logical_or(ep_success, success)
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
                b_env_actions[:, ep, step, :] = env_action
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

        data_time = time.time() - t0

        save_action_histograms(
            out_dir=out_dir,
            csv_path=action_hist_csv_path,
            update=update + 1,
            timestep=global_timesteps,
            env_actions=b_env_actions,
            valid_mask=b_valid,
            bins=80,
            hist_range=(-1.0, 1.0),
        )

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

        if args.log_rollout_learning_signal:
            signal_rows = compute_learning_signal_by_class(
                assigned_classes=assigned_classes,
                trial_ep_rewards=trial_ep_rewards,
                trial_ep_successes=trial_ep_successes,
                b_adv=b_adv,
                b_returns=b_returns,
                b_values=b_values,
                b_logprobs=b_logprobs,
                b_valid=b_valid if "b_valid" in locals() else None,
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
        loss, p_loss, v_loss, ent, ppo_class_rows = train_ppo(model, optimizer, rollouts, args, device)
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
    envs.close()


if __name__ == "__main__":
    train()
