#!/usr/bin/env python3
"""
t-SNE probe for trained TTT meta-RL checkpoints.

This script is separate from PPO training. It loads a trained checkpoint, runs
trials on Meta-World train/test classes, and collects BOTH:

  1. ttt_hidden:
       final within-episode TTT hidden state for each episode

  2. context_output:
       final aggregate/context vector used by the policy after episode
       aggregation, i.e. model.aggregate_step(episode_memory, ttt_hidden, ep).
       For aggregator_type='attn', this is the current-query attention context.
       For aggregator_type='ema', this is the EMA aggregate/context.

It then saves embeddings and both 2D and 3D t-SNE/PCA plots for both representations.

Example:
    python tsne_ttt_and_context_outputs.py \
      --run_dir csv_outputs/YOUR_ML10_RUN \
      --episodes 5 \
      --trials_per_class 5 \
      --out_dir csv_outputs/YOUR_ML10_RUN/ttt_context_tsne

If your checkpoint/run does not save obs_normalizer stats, the script uses identity
observation normalization by default. For exact eval-like behavior, provide:
    --obs_norm_npz path/to/obs_stats.npz
where the npz contains arrays named "mean" and "var".
"""

import argparse
import csv
import json
import hashlib
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt

import metaworld

from ttt import TTTConfig
from agent import TTTEpisodePolicy
from utils import MetaWorldTaskSamplerEnv, RunningMeanStd, get_agent_input, get_success_array


def parse_hidden_sizes(x, default=(64, 64)):
    if x is None:
        return tuple(default)
    if isinstance(x, (list, tuple)):
        return tuple(int(v) for v in x)
    if isinstance(x, str):
        if x.strip() == "":
            return tuple()
        return tuple(int(v.strip()) for v in x.split(",") if v.strip())
    return tuple(default)


def cfg_get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def make_args_from_config(cfg):
    # Robust defaults match your current training code as closely as possible.
    use_state_proj = bool(cfg_get(cfg, "use_state_proj", True))
    if "no_state_proj" in cfg:
        use_state_proj = not bool(cfg_get(cfg, "no_state_proj"))

    max_std = cfg_get(cfg, "max_std", 1.5)
    if max_std is not None:
        max_std = float(max_std)

    return SimpleNamespace(
        task_set=cfg_get(cfg, "task_set", "ML10"),
        env_name=cfg_get(cfg, "env_name", "reach-v3"),
        agent_mode=cfg_get(cfg, "agent_mode", "agent_rl2"),
        trial_length=int(cfg_get(cfg, "trial_length", 5)),
        rollout_steps=int(cfg_get(cfg, "rollout_steps", 500)),
        hidden_size=int(cfg_get(cfg, "hidden_size", 128)),
        num_attention_heads=int(cfg_get(cfg, "num_attention_heads", 1)),
        num_hidden_layers=int(cfg_get(cfg, "num_hidden_layers", 2)),
        mini_batch_size=int(cfg_get(cfg, "mini_batch_size", 16)),
        ttt_layer_type=cfg_get(cfg, "ttt_layer_type", "mlp"),
        policy_hidden_sizes=parse_hidden_sizes(cfg_get(cfg, "policy_hidden_sizes", (64, 64))),
        value_hidden_sizes=parse_hidden_sizes(cfg_get(cfg, "value_hidden_sizes", (64, 64))),
        aggregator_type=cfg_get(cfg, "aggregator_type", "ema"),
        ema_beta=float(cfg_get(cfg, "ema_beta", 0.7)),
        use_state_proj=use_state_proj,
        episode_attn_heads=int(cfg_get(cfg, "episode_attn_heads", 1)),
        min_std=float(cfg_get(cfg, "min_std", 0.5)),
        max_std=max_std,
        init_std=float(cfg_get(cfg, "init_std", 1.0)),
        seed=int(cfg_get(cfg, "seed", 1)),
        squash_actions=bool(cfg_get(cfg, "squash_actions", False)),
        action_scale=float(cfg_get(cfg, "action_scale", 1.0)),
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def should_mask_goal(task_set):
    return task_set == "ML1"


class IdentityObsNorm:
    def normalize(self, x):
        return x


class FrozenObsNorm:
    def __init__(self, mean, var, eps=1e-8):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.var = np.asarray(var, dtype=np.float32)
        self.eps = eps

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + self.eps)


class OnlineObsNorm:
    def __init__(self, shape):
        self.rms = RunningMeanStd(shape=shape)

    def update(self, x):
        self.rms.update(x)

    def normalize(self, x):
        return self.rms.normalize(x)


def load_obs_normalizer(args, obs_shape, checkpoint_obj=None):
    if args.obs_norm_npz:
        data = np.load(args.obs_norm_npz)
        return FrozenObsNorm(data["mean"], data["var"])

    if isinstance(checkpoint_obj, dict):
        for key in ["obs_normalizer", "obs_rms", "ob_rms"]:
            if key in checkpoint_obj:
                obj = checkpoint_obj[key]
                if isinstance(obj, dict) and "mean" in obj and "var" in obj:
                    return FrozenObsNorm(obj["mean"], obj["var"])

    if args.obs_norm_mode == "online":
        return OnlineObsNorm(shape=obs_shape)

    return IdentityObsNorm()


def build_metaworld(task_set, env_name):
    if task_set == "ML10":
        return metaworld.ML10()
    if task_set == "ML45":
        return metaworld.ML45()
    if task_set == "ML1":
        return metaworld.ML1(env_name)
    raise ValueError("This probe supports ML1/ML10/ML45.")


def make_model(cfg_args, obs_dim, action_dim, input_dim, device):
    ttt_config = TTTConfig(
        vocab_size=1,
        hidden_size=cfg_args.hidden_size,
        intermediate_size=cfg_args.hidden_size * 3,
        num_attention_heads=cfg_args.num_attention_heads,
        max_position_embeddings=max(2048, cfg_args.rollout_steps),
        num_hidden_layers=cfg_args.num_hidden_layers,
        ttt_layer_type=cfg_args.ttt_layer_type,
        rms_norm_eps=1e-5,
        use_cache=True,
        mini_batch_size=cfg_args.mini_batch_size,
        scan_checkpoint_group_size=0,
        tie_word_embeddings=False,
    )

    model = TTTEpisodePolicy(
        ttt_config,
        input_dim=input_dim,
        obs_dim=obs_dim,
        num_actions=action_dim,
        num_episodes=cfg_args.trial_length,
        continuous=True,
        policy_hidden_sizes=cfg_args.policy_hidden_sizes,
        value_hidden_sizes=cfg_args.value_hidden_sizes,
        aggregator_type=cfg_args.aggregator_type,
        ema_beta=cfg_args.ema_beta,
        use_state_proj=cfg_args.use_state_proj,
        episode_attn_heads=cfg_args.episode_attn_heads,
        min_std=cfg_args.min_std,
        max_std=cfg_args.max_std,
        init_std=cfg_args.init_std,
    ).to(device)
    return model


def load_checkpoint(model, checkpoint_path, device, strict=False):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    report = model.load_state_dict(state, strict=strict)
    if hasattr(report, "missing_keys"):
        if report.missing_keys or report.unexpected_keys:
            print("load_state_dict report:")
            print("  missing:", report.missing_keys)
            print("  unexpected:", report.unexpected_keys)
    return ckpt


def success_from_info(info):
    if isinstance(info, dict):
        if "success" in info:
            return bool(np.asarray(info["success"]).reshape(-1)[0])
        if "is_success" in info:
            return bool(np.asarray(info["is_success"]).reshape(-1)[0])
    return False



def make_vector_env_for_class(classes, tasks, task_set, class_name, num_envs, seed):
    """Create a SyncVectorEnv where every env samples the same task class.

    This batches all trials of one task class. SyncVectorEnv is not multi-process,
    but the policy/TTT forward is batched over trials, which removes the old
    one-trial-at-a-time model-forward bottleneck.
    """
    import gymnasium as gym

    def make_thunk(local_seed):
        def thunk():
            env = MetaWorldTaskSamplerEnv(
                classes,
                tasks,
                seed=local_seed,
                mask_goal=should_mask_goal(task_set),
            )
            env.sample_new_task(env_name=class_name)
            return env
        return thunk

    return gym.vector.SyncVectorEnv([make_thunk(seed + i) for i in range(num_envs)])


@torch.no_grad()
def rollout_class_vectorized(
    model,
    classes,
    tasks,
    task_set,
    class_name,
    episodes,
    rollout_steps,
    trials_per_class,
    agent_mode,
    obs_normalizer,
    device,
    seed,
    deterministic,
    squash_actions,
    action_scale,
    clip_actions,
):
    """Run all trials for one task class in one vectorized batch.

    Batch dimension B = trials_per_class. Each env is a separate task
    variation/sample from the same task class.
    """
    envs = make_vector_env_for_class(
        classes=classes,
        tasks=tasks,
        task_set=task_set,
        class_name=class_name,
        num_envs=trials_per_class,
        seed=seed,
    )

    raw_obs, _ = envs.reset()
    raw_obs = np.asarray(raw_obs, dtype=np.float32)
    if hasattr(obs_normalizer, "update"):
        obs_normalizer.update(raw_obs)
    obs = obs_normalizer.normalize(raw_obs)

    B = trials_per_class
    action_dim = envs.single_action_space.shape[0]
    low = envs.single_action_space.low
    high = envs.single_action_space.high

    episode_memory = model.init_episode_memory(B, device=device, num_episodes=episodes)
    prev_action = np.zeros((B, action_dim), dtype=np.float32)
    prev_reward = np.zeros((B, 1), dtype=np.float32)
    prev_done = np.zeros((B, 1), dtype=np.float32)

    ttt_embeddings = []
    context_embeddings = []
    rows = []

    for ep in range(episodes):
        cache_params = None
        ep_return = np.zeros(B, dtype=np.float32)
        ep_success = np.zeros(B, dtype=bool)
        steps_taken = np.zeros(B, dtype=np.int32)
        out = None
        final_context = None

        for step in range(rollout_steps):
            agent_input = get_agent_input(obs, prev_action, prev_reward, prev_done, agent_mode)
            inp_t = torch.tensor(agent_input, dtype=torch.float32, device=device)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

            out = model.act_step(inp_t, obs_t, episode_memory, ep, cache_params)
            final_context = model.aggregate_step(episode_memory, out.hidden_states, ep)

            mean, log_std = out.policy
            if deterministic:
                raw_action = mean
            else:
                dist = torch.distributions.Normal(mean, log_std.exp())
                raw_action = dist.sample()

            if squash_actions:
                env_action_t = action_scale * torch.tanh(raw_action)
            else:
                env_action_t = raw_action

            env_action = env_action_t.detach().cpu().numpy().astype(np.float32)
            if clip_actions:
                env_action = np.clip(env_action, low, high).astype(np.float32)

            next_raw_obs, reward, terminated, truncated, info = envs.step(env_action)
            done = np.logical_or(terminated, truncated)

            ep_return += reward.astype(np.float32)
            ep_success = np.logical_or(ep_success, get_success_array(info, B))
            steps_taken[~done] = step + 1
            steps_taken[done] = np.maximum(steps_taken[done], step + 1)

            cache_params = out.cache_params
            prev_action = env_action
            prev_reward = reward[:, None].astype(np.float32)
            prev_done = done[:, None].astype(np.float32)

            next_raw_obs = np.asarray(next_raw_obs, dtype=np.float32)
            if hasattr(obs_normalizer, "update"):
                obs_normalizer.update(next_raw_obs)
            obs = obs_normalizer.normalize(next_raw_obs)

            if done.all():
                break

        if out is None or final_context is None:
            envs.close()
            raise RuntimeError("Episode produced no rollout steps.")

        ttt_h = out.hidden_states.detach().cpu().numpy().astype(np.float32)
        ctx_h = final_context.detach().cpu().numpy().astype(np.float32)

        ttt_embeddings.append(ttt_h)
        context_embeddings.append(ctx_h)
        episode_memory[:, ep, :] = out.hidden_states.detach()

        for trial_idx in range(B):
            rows.append(
                {
                    "class_name": class_name,
                    "episode": ep + 1,
                    "episode_return": float(ep_return[trial_idx]),
                    "episode_success": int(ep_success[trial_idx]),
                    "steps": int(max(1, steps_taken[trial_idx])),
                    "trial": trial_idx,
                    "seed": seed + trial_idx,
                }
            )

        if ep < episodes - 1:
            raw_obs, _ = envs.reset()
            raw_obs = np.asarray(raw_obs, dtype=np.float32)
            if hasattr(obs_normalizer, "update"):
                obs_normalizer.update(raw_obs)
            obs = obs_normalizer.normalize(raw_obs)
            prev_action[:] = 0.0
            prev_reward[:] = 0.0
            prev_done[:] = 0.0

    envs.close()

    # Row order is episode-major then trial-minor, matching rows above.
    X_ttt = np.concatenate(ttt_embeddings, axis=0)
    X_ctx = np.concatenate(context_embeddings, axis=0)
    return X_ttt, X_ctx, rows


@torch.no_grad()
def rollout_one_class(
    model,
    classes,
    tasks,
    task_set,
    class_name,
    episodes,
    rollout_steps,
    agent_mode,
    obs_normalizer,
    device,
    seed,
    deterministic,
    squash_actions,
    action_scale,
    clip_actions,
):
    env = MetaWorldTaskSamplerEnv(
        classes,
        tasks,
        seed=seed,
        mask_goal=should_mask_goal(task_set),
    )
    env.sample_new_task(env_name=class_name)

    raw_obs, _ = env.reset()
    raw_obs = np.asarray(raw_obs, dtype=np.float32)
    if hasattr(obs_normalizer, "update"):
        obs_normalizer.update(raw_obs[None, :])
    obs = obs_normalizer.normalize(raw_obs[None, :])

    action_dim = env.action_space.shape[0]
    low = env.action_space.low
    high = env.action_space.high

    episode_memory = model.init_episode_memory(1, device=device, num_episodes=episodes)
    prev_action = np.zeros((1, action_dim), dtype=np.float32)
    prev_reward = np.zeros((1, 1), dtype=np.float32)
    prev_done = np.zeros((1, 1), dtype=np.float32)

    ttt_embeddings = []
    context_embeddings = []
    rows = []

    for ep in range(episodes):
        cache_params = None
        ep_return = 0.0
        ep_success = False
        out = None
        final_context = None

        for step in range(rollout_steps):
            agent_input = get_agent_input(obs, prev_action, prev_reward, prev_done, agent_mode)
            inp_t = torch.tensor(agent_input, dtype=torch.float32, device=device)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

            out = model.act_step(inp_t, obs_t, episode_memory, ep, cache_params)

            # Recompute the exact aggregate/context vector that the policy used
            # at this timestep. This does not require changing agent.py.
            final_context = model.aggregate_step(
                episode_memory,
                out.hidden_states,
                ep,
            )

            mean, log_std = out.policy
            if deterministic:
                raw_action = mean
            else:
                dist = torch.distributions.Normal(mean, log_std.exp())
                raw_action = dist.sample()

            if squash_actions:
                env_action_t = action_scale * torch.tanh(raw_action)
            else:
                env_action_t = raw_action

            env_action = env_action_t.detach().cpu().numpy()[0].astype(np.float32)
            if clip_actions:
                env_action = np.clip(env_action, low, high).astype(np.float32)

            next_raw_obs, reward, terminated, truncated, info = env.step(env_action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
            ep_success = ep_success or success_from_info(info)

            cache_params = out.cache_params
            prev_action = env_action[None, :]
            prev_reward = np.asarray([[reward]], dtype=np.float32)
            prev_done = np.asarray([[float(done)]], dtype=np.float32)

            next_raw_obs = np.asarray(next_raw_obs, dtype=np.float32)
            if hasattr(obs_normalizer, "update"):
                obs_normalizer.update(next_raw_obs[None, :])
            obs = obs_normalizer.normalize(next_raw_obs[None, :])

            if done:
                break

        if out is None or final_context is None:
            raise RuntimeError("Episode produced no rollout steps.")

        ttt_h = out.hidden_states.detach().cpu().numpy()[0].astype(np.float32)
        ctx_h = final_context.detach().cpu().numpy()[0].astype(np.float32)

        ttt_embeddings.append(ttt_h)
        context_embeddings.append(ctx_h)
        rows.append(
            {
                "class_name": class_name,
                "episode": ep + 1,
                "episode_return": ep_return,
                "episode_success": int(ep_success),
                "steps": step + 1,
            }
        )

        episode_memory[:, ep, :] = out.hidden_states.detach()

        if ep < episodes - 1:
            raw_obs, _ = env.reset()
            raw_obs = np.asarray(raw_obs, dtype=np.float32)
            if hasattr(obs_normalizer, "update"):
                obs_normalizer.update(raw_obs[None, :])
            obs = obs_normalizer.normalize(raw_obs[None, :])
            prev_action[:] = 0.0
            prev_reward[:] = 0.0
            prev_done[:] = 0.0

    env.close()
    return np.stack(ttt_embeddings, axis=0), np.stack(context_embeddings, axis=0), rows


def collect_embeddings(args, cfg_args, model, meta_learning, obs_normalizer, device):
    all_ttt = []
    all_ctx = []
    all_rows = []

    specs = []
    if args.include_train:
        for name in sorted(meta_learning.train_classes.keys()):
            specs.append(("train", name, meta_learning.train_classes, meta_learning.train_tasks))
    if args.include_test:
        for name in sorted(meta_learning.test_classes.keys()):
            specs.append(("test", name, meta_learning.test_classes, meta_learning.test_tasks))

    for split, class_name, classes, tasks in specs:
        seed = args.seed + 100000 * (split == "test") + 1000 * len(all_rows)

        if getattr(args, "serial_collection", False):
            # Old slow path: one env/trial at a time.
            for trial in range(args.trials_per_class):
                trial_seed = seed + trial
                print(
                    f"Collecting SERIAL split={split:<5s} class={class_name:<28s} "
                    f"trial={trial + 1}/{args.trials_per_class}"
                )
                ttt, ctx, rows = rollout_one_class(
                    model=model,
                    classes=classes,
                    tasks=tasks,
                    task_set=cfg_args.task_set,
                    class_name=class_name,
                    episodes=args.episodes,
                    rollout_steps=args.rollout_steps or cfg_args.rollout_steps,
                    agent_mode=cfg_args.agent_mode,
                    obs_normalizer=obs_normalizer,
                    device=device,
                    seed=trial_seed,
                    deterministic=not args.stochastic,
                    squash_actions=cfg_args.squash_actions or args.force_squash_actions,
                    action_scale=args.action_scale if args.action_scale is not None else cfg_args.action_scale,
                    clip_actions=args.clip_actions,
                )
                for r in rows:
                    r["split"] = split
                    r["trial"] = trial
                    r["seed"] = trial_seed
                    all_rows.append(r)
                all_ttt.append(ttt)
                all_ctx.append(ctx)
        else:
            # Fast path: vectorize all trials for this class.
            print(
                f"Collecting VECTORIZED split={split:<5s} class={class_name:<28s} "
                f"trials={args.trials_per_class}"
            )
            ttt, ctx, rows = rollout_class_vectorized(
                model=model,
                classes=classes,
                tasks=tasks,
                task_set=cfg_args.task_set,
                class_name=class_name,
                episodes=args.episodes,
                rollout_steps=args.rollout_steps or cfg_args.rollout_steps,
                trials_per_class=args.trials_per_class,
                agent_mode=cfg_args.agent_mode,
                obs_normalizer=obs_normalizer,
                device=device,
                seed=seed,
                deterministic=not args.stochastic,
                squash_actions=cfg_args.squash_actions or args.force_squash_actions,
                action_scale=args.action_scale if args.action_scale is not None else cfg_args.action_scale,
                clip_actions=args.clip_actions,
            )
            for r in rows:
                r["split"] = split
                all_rows.append(r)
            all_ttt.append(ttt)
            all_ctx.append(ctx)

    return np.concatenate(all_ttt, axis=0), np.concatenate(all_ctx, axis=0), all_rows


def make_nd(X, seed, perplexity, n_components):
    """Return an n-dimensional t-SNE embedding, with PCA fallback."""
    if X.shape[0] < 3:
        raise ValueError("Need at least 3 points for t-SNE/PCA.")
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3.")

    try:
        from sklearn.manifold import TSNE
        method = "tsne"
        perp = min(float(perplexity), max(2.0, float((X.shape[0] - 1) // 3)))
        Z = TSNE(
            n_components=n_components,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(X)
    except Exception as e:
        print(f"Could not run sklearn TSNE ({e}). Falling back to PCA.")
        method = "pca"
        Xc = X - X.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        Z = Xc @ vt[:n_components].T

    return Z, method


def make_2d(X, seed, perplexity):
    return make_nd(X, seed, perplexity, n_components=2)


def make_3d(X, seed, perplexity):
    return make_nd(X, seed, perplexity, n_components=3)


def save_csv(rows, Z_ttt_2d, Z_ctx_2d, Z_ttt_3d, Z_ctx_3d, out_path):
    fields = [
        "split",
        "class_name",
        "trial",
        "episode",
        "episode_return",
        "episode_success",
        "steps",
        "seed",
        "ttt_2d_x",
        "ttt_2d_y",
        "context_2d_x",
        "context_2d_y",
        "ttt_3d_x",
        "ttt_3d_y",
        "ttt_3d_z",
        "context_3d_x",
        "context_3d_y",
        "context_3d_z",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row, zt2, zc2, zt3, zc3 in zip(rows, Z_ttt_2d, Z_ctx_2d, Z_ttt_3d, Z_ctx_3d):
            r = dict(row)
            r["ttt_2d_x"] = float(zt2[0])
            r["ttt_2d_y"] = float(zt2[1])
            r["context_2d_x"] = float(zc2[0])
            r["context_2d_y"] = float(zc2[1])
            r["ttt_3d_x"] = float(zt3[0])
            r["ttt_3d_y"] = float(zt3[1])
            r["ttt_3d_z"] = float(zt3[2])
            r["context_3d_x"] = float(zc3[0])
            r["context_3d_y"] = float(zc3[1])
            r["context_3d_z"] = float(zc3[2])
            writer.writerow(r)



def make_distinct_color_map(labels):
    """Return a stable, high-contrast color map for task labels."""
    labels = sorted(list(labels))
    if not labels:
        return {}

    # For ML10/ML45-sized plots, concatenating categorical palettes gives more
    # distinct colors than repeatedly sampling tab20 alone.
    palette = []
    for cmap_name in ["tab20", "tab20b", "tab20c", "Set3", "Dark2", "Accent", "Paired"]:
        cmap = plt.get_cmap(cmap_name)
        n = getattr(cmap, "N", 20)
        palette.extend([cmap(i) for i in range(n)])

    if len(labels) > len(palette):
        # Fallback for very many labels.
        cmap = plt.get_cmap("nipy_spectral")
        palette = [cmap(i / max(1, len(labels) - 1)) for i in range(len(labels))]

    return {label: palette[i % len(palette)] for i, label in enumerate(labels)}


def get_plot_style(args):
    return {
        "marker_size": float(args.marker_size),
        "marker_alpha": float(args.marker_alpha),
        "edge_width": float(args.marker_edge_width),
        "edge_color": args.marker_edge_color,
        "legend_cols": int(args.legend_cols),
    }


def plot_by_task(Z, rows, out_path, title, style):
    classes = sorted(set(r["class_name"] for r in rows))
    color_map = make_distinct_color_map(classes)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)

    for cls in classes:
        for split, marker in [("train", "o"), ("test", "^")]:
            idx = [i for i, r in enumerate(rows) if r["class_name"] == cls and r["split"] == split]
            if not idx:
                continue
            ax.scatter(
                Z[idx, 0],
                Z[idx, 1],
                s=style["marker_size"],
                marker=marker,
                alpha=style["marker_alpha"],
                color=color_map[cls],
                edgecolors=style["edge_color"],
                linewidths=style["edge_width"],
                label=f"{split}:{cls}",
            )

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=style["legend_cols"], loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_by_episode(Z, rows, out_path, title, style):
    episodes = np.asarray([r["episode"] for r in rows])
    splits = [r["split"] for r in rows]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    sc = None
    for split, marker in [("train", "o"), ("test", "^")]:
        idx = np.asarray([i for i, s in enumerate(splits) if s == split], dtype=int)
        if idx.size == 0:
            continue
        sc = ax.scatter(
            Z[idx, 0],
            Z[idx, 1],
            c=episodes[idx],
            cmap="viridis",
            s=style["marker_size"],
            marker=marker,
            alpha=style["marker_alpha"],
            edgecolors=style["edge_color"],
            linewidths=style["edge_width"],
            label=split,
        )

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("episode index")
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_by_split(Z, rows, out_path, title, style):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    split_colors = {"train": "#1f77b4", "test": "#d62728"}
    for split, marker in [("train", "o"), ("test", "^")]:
        idx = [i for i, r in enumerate(rows) if r["split"] == split]
        if not idx:
            continue
        ax.scatter(
            Z[idx, 0],
            Z[idx, 1],
            s=style["marker_size"],
            marker=marker,
            alpha=style["marker_alpha"],
            color=split_colors.get(split, None),
            edgecolors=style["edge_color"],
            linewidths=style["edge_width"],
            label=split,
        )

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_3d_by_task(Z, rows, out_path, title, style):
    classes = sorted(set(r["class_name"] for r in rows))
    color_map = make_distinct_color_map(classes)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    for cls in classes:
        for split, marker in [("train", "o"), ("test", "^")]:
            idx = [i for i, r in enumerate(rows) if r["class_name"] == cls and r["split"] == split]
            if not idx:
                continue
            ax.scatter(
                Z[idx, 0],
                Z[idx, 1],
                Z[idx, 2],
                s=style["marker_size"],
                marker=marker,
                alpha=style["marker_alpha"],
                color=color_map[cls],
                edgecolors=style["edge_color"],
                linewidths=style["edge_width"],
                label=f"{split}:{cls}",
            )

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_zlabel("dim 3")
    ax.legend(fontsize=7, ncol=style["legend_cols"], loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_3d_by_episode(Z, rows, out_path, title, style):
    episodes = np.asarray([r["episode"] for r in rows])
    splits = [r["split"] for r in rows]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc = None
    for split, marker in [("train", "o"), ("test", "^")]:
        idx = np.asarray([i for i, s in enumerate(splits) if s == split], dtype=int)
        if idx.size == 0:
            continue
        sc = ax.scatter(
            Z[idx, 0],
            Z[idx, 1],
            Z[idx, 2],
            c=episodes[idx],
            cmap="viridis",
            s=style["marker_size"],
            marker=marker,
            alpha=style["marker_alpha"],
            edgecolors=style["edge_color"],
            linewidths=style["edge_width"],
            label=split,
        )

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label("episode index")
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_zlabel("dim 3")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_3d_by_split(Z, rows, out_path, title, style):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    split_colors = {"train": "#1f77b4", "test": "#d62728"}
    for split, marker in [("train", "o"), ("test", "^")]:
        idx = [i for i, r in enumerate(rows) if r["split"] == split]
        if not idx:
            continue
        ax.scatter(
            Z[idx, 0],
            Z[idx, 1],
            Z[idx, 2],
            s=style["marker_size"],
            marker=marker,
            alpha=style["marker_alpha"],
            color=split_colors.get(split, None),
            edgecolors=style["edge_color"],
            linewidths=style["edge_width"],
            label=split,
        )

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_zlabel("dim 3")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_representation(Z_2d, Z_3d, rows, out_dir, name, method_2d, method_3d, run_name, style):
    title_2d = f"2D {method_2d.upper()} of {name} | {run_name}"
    plot_by_task(Z_2d, rows, out_dir / f"{name}_2d_by_task.png", title_2d + " | colored by task", style)
    plot_by_episode(Z_2d, rows, out_dir / f"{name}_2d_by_episode.png", title_2d + " | colored by episode", style)
    plot_by_split(Z_2d, rows, out_dir / f"{name}_2d_by_split.png", title_2d + " | train/test split", style)

    title_3d = f"3D {method_3d.upper()} of {name} | {run_name}"
    plot_3d_by_task(Z_3d, rows, out_dir / f"{name}_3d_by_task.png", title_3d + " | colored by task", style)
    plot_3d_by_episode(Z_3d, rows, out_dir / f"{name}_3d_by_episode.png", title_3d + " | colored by episode", style)
    plot_3d_by_split(Z_3d, rows, out_dir / f"{name}_3d_by_split.png", title_3d + " | train/test split", style)


def _file_fingerprint(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return {"path": str(p)}
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _stable_hash(payload):
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _array_hash(x):
    arr = np.ascontiguousarray(x)
    h = hashlib.sha1()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.view(np.uint8))
    return h.hexdigest()[:16]


def make_rollout_cache_key(args, cfg_args, checkpoint_path, config_path):
    payload = {
        "kind": "rollout_embeddings_v2",
        "checkpoint": _file_fingerprint(checkpoint_path),
        "config": _file_fingerprint(config_path),
        "obs_norm_npz": _file_fingerprint(args.obs_norm_npz),
        "task_set": cfg_args.task_set,
        "agent_mode": cfg_args.agent_mode,
        "aggregator_type": cfg_args.aggregator_type,
        "episodes": args.episodes,
        "trials_per_class": args.trials_per_class,
        "rollout_steps": args.rollout_steps or cfg_args.rollout_steps,
        "seed": args.seed,
        "include_train": args.include_train,
        "include_test": args.include_test,
        "stochastic": args.stochastic,
        "force_squash_actions": args.force_squash_actions,
        "cfg_squash_actions": cfg_args.squash_actions,
        "action_scale": args.action_scale if args.action_scale is not None else cfg_args.action_scale,
        "clip_actions": args.clip_actions,
        "obs_norm_mode": args.obs_norm_mode,
        "serial_collection": getattr(args, "serial_collection", False),
    }
    return _stable_hash(payload), payload


def load_or_collect_embeddings(args, cfg_args, model, meta_learning, obs_normalizer, device, checkpoint_path, config_path, out_dir):
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    key, payload = make_rollout_cache_key(args, cfg_args, checkpoint_path, config_path)
    npz_path = cache_dir / f"rollout_embeddings_{key}.npz"
    rows_path = cache_dir / f"rollout_rows_{key}.json"
    meta_path = cache_dir / f"rollout_meta_{key}.json"

    use_cache = (not args.no_cache) and (not args.force_recompute_rollouts)
    if use_cache and npz_path.exists() and rows_path.exists():
        print(f"Loading cached rollout embeddings: {npz_path}")
        data = np.load(npz_path)
        with open(rows_path, "r") as f:
            rows = json.load(f)
        return data["ttt_hidden"], data["context_output"], rows, key

    print("Collecting rollout embeddings...")
    X_ttt, X_ctx, rows = collect_embeddings(args, cfg_args, model, meta_learning, obs_normalizer, device)

    if not args.no_cache:
        np.savez_compressed(npz_path, ttt_hidden=X_ttt, context_output=X_ctx)
        with open(rows_path, "w") as f:
            json.dump(rows, f, indent=2)
        with open(meta_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"Saved rollout cache: {npz_path}")

    return X_ttt, X_ctx, rows, key


def load_or_compute_nd(args, X, rep_name, n_components, rollout_key, out_dir):
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "kind": "embedding_nd_v2",
        "rollout_key": rollout_key,
        "rep_name": rep_name,
        "n_components": n_components,
        "perplexity": args.perplexity,
        "seed": args.seed,
        "shape": tuple(X.shape),
        "data_hash": _array_hash(X),
    }
    key = _stable_hash(payload)
    npz_path = cache_dir / f"{rep_name}_{n_components}d_{key}.npz"
    meta_path = cache_dir / f"{rep_name}_{n_components}d_meta_{key}.json"

    use_cache = (not args.no_cache) and (not args.force_recompute_tsne)
    if use_cache and npz_path.exists():
        print(f"Loading cached {rep_name} {n_components}D embedding: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        Z = data["Z"]
        method = str(data["method"])
        return Z, method

    Z, method = make_nd(X, seed=args.seed, perplexity=args.perplexity, n_components=n_components)

    if not args.no_cache:
        np.savez_compressed(npz_path, Z=Z, method=np.asarray(method))
        with open(meta_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"Saved {rep_name} {n_components}D cache: {npz_path}")

    return Z, method

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default='csv_outputs/nostate_ML10_mb128rew_adv_norm_istd1_run006', help="Run dir containing config.json and ttt_ecet_policy.pth")
    parser.add_argument("--checkpoint", default=None, help="Path to ttt_ecet_policy.pth")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--out_dir", default='csv_outputs/nostate_ML10_mb128rew_adv_norm_istd1_run006/tsne')

    parser.add_argument("--episodes", type=int, default=None, help="Episodes per trial. Default: config trial_length.")
    parser.add_argument("--trials_per_class", type=int, default=10)
    parser.add_argument("--serial_collection", action="store_true", help="Use the old one-trial-at-a-time collection path.")
    parser.add_argument("--rollout_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--include_train", action="store_true", default=True)
    parser.add_argument("--include_test", action="store_true", default=True)
    parser.add_argument("--no_train", action="store_false", dest="include_train")
    parser.add_argument("--no_test", action="store_false", dest="include_test")

    parser.add_argument("--stochastic", action="store_true", help="Sample policy actions instead of using mean.")
    parser.add_argument("--force_squash_actions", action="store_true")
    parser.add_argument("--action_scale", type=float, default=None)
    parser.add_argument("--clip_actions", action="store_true")

    parser.add_argument("--obs_norm_npz", default=None, help="Optional npz containing mean and var.")
    parser.add_argument(
        "--obs_norm_mode",
        choices=["identity", "online"],
        default="identity",
        help="Used if obs stats are not found/provided.",
    )

    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--marker_size", type=float, default=46.0, help="Marker size for circles/triangles.")
    parser.add_argument("--marker_alpha", type=float, default=0.85, help="Marker transparency.")
    parser.add_argument("--marker_edge_width", type=float, default=0.35, help="Marker edge width.")
    parser.add_argument("--marker_edge_color", default="black", help="Marker edge color.")
    parser.add_argument("--legend_cols", type=int, default=2, help="Number of legend columns for task plots.")
    parser.add_argument("--cache_dir", default=None, help="Directory for cached rollouts/t-SNE. Default: out_dir/cache.")
    parser.add_argument("--no_cache", action="store_true", help="Disable loading/saving all caches.")
    parser.add_argument("--force_recompute_rollouts", action="store_true", help="Ignore cached rollout embeddings.")
    parser.add_argument("--force_recompute_tsne", action="store_true", help="Ignore cached t-SNE/PCA projections.")
    parser.add_argument("--strict_load", action="store_true")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "ttt_ecet_policy.pth"
        config_path = Path(args.config) if args.config else run_dir / "config.json"
        out_dir = Path(args.out_dir) if args.out_dir else run_dir / "ttt_context_tsne"
        run_name = run_dir.name
    else:
        if args.checkpoint is None or args.config is None:
            raise ValueError("Provide either --run_dir or both --checkpoint and --config.")
        checkpoint_path = Path(args.checkpoint)
        config_path = Path(args.config)
        out_dir = Path(args.out_dir) if args.out_dir else checkpoint_path.parent / "ttt_context_tsne"
        run_name = checkpoint_path.parent.name

    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = load_config(config_path)
    cfg_args = make_args_from_config(cfg_dict)
    if args.episodes is None:
        args.episodes = cfg_args.trial_length

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading config: {config_path}")
    print(f"Loading checkpoint: {checkpoint_path}")

    meta_learning = build_metaworld(cfg_args.task_set, cfg_args.env_name)

    # Temporary env to infer spaces.
    tmp_classes = meta_learning.train_classes
    tmp_tasks = meta_learning.train_tasks
    tmp_class = sorted(tmp_classes.keys())[0]
    tmp_env = MetaWorldTaskSamplerEnv(
        tmp_classes,
        tmp_tasks,
        seed=args.seed,
        mask_goal=should_mask_goal(cfg_args.task_set),
    )
    tmp_env.sample_new_task(env_name=tmp_class)
    obs_dim = tmp_env.observation_space.shape[0]
    action_dim = tmp_env.action_space.shape[0]
    tmp_env.close()

    input_dim = obs_dim
    if cfg_args.agent_mode == "agent_v2":
        input_dim = obs_dim + action_dim + 1
    elif cfg_args.agent_mode == "agent_rl2":
        input_dim = obs_dim + action_dim + 2

    model = make_model(cfg_args, obs_dim, action_dim, input_dim, device)
    ckpt = load_checkpoint(model, checkpoint_path, device, strict=args.strict_load)
    model.eval()

    obs_normalizer = load_obs_normalizer(args, obs_shape=(obs_dim,), checkpoint_obj=ckpt)

    print("Collection settings:")
    print(f"  task_set={cfg_args.task_set}")
    print(f"  agent_mode={cfg_args.agent_mode}")
    print(f"  aggregator_type={cfg_args.aggregator_type}")
    print(f"  episodes={args.episodes}")
    print(f"  trials_per_class={args.trials_per_class}")
    print(f"  rollout_steps={args.rollout_steps or cfg_args.rollout_steps}")
    print(f"  squash_actions={cfg_args.squash_actions or args.force_squash_actions}")
    print(f"  action_scale={args.action_scale if args.action_scale is not None else cfg_args.action_scale}")
    print(f"  obs_norm={type(obs_normalizer).__name__}")

    X_ttt, X_ctx, rows, rollout_cache_key = load_or_collect_embeddings(
        args,
        cfg_args,
        model,
        meta_learning,
        obs_normalizer,
        device,
        checkpoint_path,
        config_path,
        out_dir,
    )
    print(f"Collected ttt_hidden: {X_ttt.shape}")
    print(f"Collected context_output: {X_ctx.shape}")

    Z_ttt_2d, method_ttt_2d = load_or_compute_nd(args, X_ttt, "ttt_hidden", 2, rollout_cache_key, out_dir)
    Z_ctx_2d, method_ctx_2d = load_or_compute_nd(args, X_ctx, "context_output", 2, rollout_cache_key, out_dir)
    Z_ttt_3d, method_ttt_3d = load_or_compute_nd(args, X_ttt, "ttt_hidden", 3, rollout_cache_key, out_dir)
    Z_ctx_3d, method_ctx_3d = load_or_compute_nd(args, X_ctx, "context_output", 3, rollout_cache_key, out_dir)

    np.savez(
        out_dir / "episode_embeddings_ttt_and_context.npz",
        ttt_hidden=X_ttt,
        context_output=X_ctx,
        ttt_2d=Z_ttt_2d,
        context_2d=Z_ctx_2d,
        ttt_3d=Z_ttt_3d,
        context_3d=Z_ctx_3d,
    )
    save_csv(
        rows,
        Z_ttt_2d,
        Z_ctx_2d,
        Z_ttt_3d,
        Z_ctx_3d,
        out_dir / "episode_embeddings_ttt_and_context.csv",
    )

    plot_style = get_plot_style(args)

    plot_representation(
        Z_ttt_2d,
        Z_ttt_3d,
        rows,
        out_dir,
        "ttt_hidden",
        method_ttt_2d,
        method_ttt_3d,
        run_name,
        plot_style,
    )
    plot_representation(
        Z_ctx_2d,
        Z_ctx_3d,
        rows,
        out_dir,
        "context_output",
        method_ctx_2d,
        method_ctx_3d,
        run_name,
        plot_style,
    )

    print(f"Saved outputs to: {out_dir}")
    print("  episode_embeddings_ttt_and_context.npz")
    print("  episode_embeddings_ttt_and_context.csv")
    print("  ttt_hidden_2d_by_task.png")
    print("  ttt_hidden_2d_by_episode.png")
    print("  ttt_hidden_2d_by_split.png")
    print("  ttt_hidden_3d_by_task.png")
    print("  ttt_hidden_3d_by_episode.png")
    print("  ttt_hidden_3d_by_split.png")
    print("  context_output_2d_by_task.png")
    print("  context_output_2d_by_episode.png")
    print("  context_output_2d_by_split.png")
    print("  context_output_3d_by_task.png")
    print("  context_output_3d_by_episode.png")
    print("  context_output_3d_by_split.png")


if __name__ == "__main__":
    main()
