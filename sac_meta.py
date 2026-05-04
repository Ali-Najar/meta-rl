"""Online SAC utilities for TTT MetaWorld trials.

This file keeps the PPO-style trial/context idea, but replaces PPO losses with
online SAC losses and an optional TTT forecasting auxiliary objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def mlp(input_dim: int, output_dim: int, hidden_sizes=(256, 256), act=nn.ReLU) -> nn.Sequential:
    layers = []
    d = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(d, h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, output_dim))
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, action_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim + z_dim, 2 * action_dim, hidden_sizes)
        self.action_dim = int(action_dim)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, z], dim=-1)
        mean, log_std = self.net(x).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs, z)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        pre_tanh = normal.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def act(self, obs: torch.Tensor, z: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(obs, z)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        pre_tanh = torch.distributions.Normal(mean, std).sample()
        return torch.tanh(pre_tanh)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, action_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.q = mlp(obs_dim + z_dim + action_dim, 1, hidden_sizes)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, action, z], dim=-1))


class KStepForecaster(nn.Module):
    """Open-loop dynamics/reward decoder.

    Given z_t, s_t, and actions a_t:t+K-1, predicts normalized next states
    s_{t+1:t+K} and raw rewards r_{t:t+K-1}.
    """

    def __init__(self, obs_dim: int, z_dim: int, action_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim + z_dim + action_dim, obs_dim + 1, hidden_sizes)
        self.obs_dim = int(obs_dim)

    def forward(self, z: torch.Tensor, start_obs: torch.Tensor, action_seq: torch.Tensor):
        current = start_obs
        pred_next_obs = []
        pred_rewards = []
        horizon = action_seq.shape[1]
        for k in range(horizon):
            out = self.net(torch.cat([z, current, action_seq[:, k]], dim=-1))
            delta_obs = out[:, : self.obs_dim]
            reward = out[:, self.obs_dim : self.obs_dim + 1]
            current = current + delta_obs
            pred_next_obs.append(current)
            pred_rewards.append(reward)
        return torch.stack(pred_next_obs, dim=1), torch.cat(pred_rewards, dim=1)


@dataclass
class SacBatch:
    inputs: torch.Tensor       # (B,E,T,D)
    states: torch.Tensor       # (B,E,T,O)
    actions: torch.Tensor      # (B,E,T,A)
    rewards: torch.Tensor      # (B,E,T)
    dones: torch.Tensor        # (B,E,T)
    next_states: torch.Tensor  # (B,E,T,O)
    episode_idx: torch.Tensor  # (B,)
    step_idx: torch.Tensor     # (B,)


@dataclass
class SacChunkBatch:
    """Batch of episode chunks for amortized context computation.

    Each item is one sampled trial/environment episode plus a contiguous chunk
    of timesteps.  The current episode is encoded once over an extended causal
    window, then the matching hidden state is gathered for each chunk timestep.
    Flattening B chunks of length C gives an effective transition batch of B*C
    while running the expensive TTT current-context encoder only B times.
    """

    inputs: torch.Tensor       # (B,E,T,D)
    states: torch.Tensor       # (B,E,T,O)
    actions: torch.Tensor      # (B,E,T,A)
    rewards: torch.Tensor      # (B,E,T)
    dones: torch.Tensor        # (B,E,T)
    next_states: torch.Tensor  # (B,E,T,O)
    episode_idx: torch.Tensor  # (B,)
    start_idx: torch.Tensor    # (B,)
    chunk_len: int


class TrialReplayBuffer:
    """Replay buffer that stores complete vectorized trials.

    Each stored trial has shape (num_envs, trial_length, rollout_steps, ...).
    Sampling returns whole trials for a sampled transition, because context must
    be reconstructed causally from previous episodes/current-window history.
    """

    def __init__(self, capacity_trials: int):
        self.capacity_trials = int(capacity_trials)
        self.storage = []
        self.pos = 0

    def __len__(self):
        return len(self.storage)

    def add_trial_batch(self, inputs, states, actions, rewards, dones, next_states):
        item = {
            "inputs": np.asarray(inputs, dtype=np.float32),
            "states": np.asarray(states, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
            "next_states": np.asarray(next_states, dtype=np.float32),
        }
        if len(self.storage) < self.capacity_trials:
            self.storage.append(item)
        else:
            self.storage[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity_trials

    def sample(self, batch_size: int, device: torch.device) -> SacBatch:
        if len(self.storage) == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")

        trial_ids = np.random.randint(0, len(self.storage), size=batch_size)
        inputs = []
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        ep_idx = []
        step_idx = []
        for tid in trial_ids:
            item = self.storage[int(tid)]
            b = item["inputs"].shape[0]
            e = item["inputs"].shape[1]
            t = item["inputs"].shape[2]
            env_i = np.random.randint(0, b)
            ep_i = np.random.randint(0, e)
            st_i = np.random.randint(0, t)
            inputs.append(item["inputs"][env_i])
            states.append(item["states"][env_i])
            actions.append(item["actions"][env_i])
            rewards.append(item["rewards"][env_i])
            dones.append(item["dones"][env_i])
            next_states.append(item["next_states"][env_i])
            ep_idx.append(ep_i)
            step_idx.append(st_i)

        return SacBatch(
            inputs=torch.tensor(np.stack(inputs), dtype=torch.float32, device=device),
            states=torch.tensor(np.stack(states), dtype=torch.float32, device=device),
            actions=torch.tensor(np.stack(actions), dtype=torch.float32, device=device),
            rewards=torch.tensor(np.stack(rewards), dtype=torch.float32, device=device),
            dones=torch.tensor(np.stack(dones), dtype=torch.float32, device=device),
            next_states=torch.tensor(np.stack(next_states), dtype=torch.float32, device=device),
            episode_idx=torch.tensor(ep_idx, dtype=torch.long, device=device),
            step_idx=torch.tensor(step_idx, dtype=torch.long, device=device),
        )

    def sample_episode_chunks(
        self,
        episode_batch_size: int,
        chunk_steps: int,
        device: torch.device,
        min_future_steps: int = 0,
    ) -> SacChunkBatch:
        """Sample B episodes and one contiguous timestep chunk from each.

        min_future_steps reserves extra valid timesteps after the chunk.  SAC
        critic/actor updates use 0 because next_states is already stored for
        each transition.  Forecasting uses horizon-1 so every start inside the
        chunk has enough future actions/rewards/next_states.
        """
        if len(self.storage) == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")

        episode_batch_size = max(1, int(episode_batch_size))
        requested_chunk = max(1, int(chunk_steps))
        min_future_steps = max(0, int(min_future_steps))

        trial_ids = np.random.randint(0, len(self.storage), size=episode_batch_size)
        inputs = []
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        ep_idx = []
        start_idx = []
        actual_chunk_len = None

        for tid in trial_ids:
            item = self.storage[int(tid)]
            b = item["inputs"].shape[0]
            e = item["inputs"].shape[1]
            t = item["inputs"].shape[2]

            if t <= min_future_steps:
                raise RuntimeError(
                    f"rollout_steps={t} is too small for min_future_steps={min_future_steps}"
                )
            chunk_len = min(requested_chunk, t - min_future_steps)
            if actual_chunk_len is None:
                actual_chunk_len = int(chunk_len)
            else:
                # Stored trials are normally same shape.  Keep one chunk_len so
                # tensors flatten cleanly even if a future buffer mixes shapes.
                actual_chunk_len = min(actual_chunk_len, int(chunk_len))

            env_i = np.random.randint(0, b)
            ep_i = np.random.randint(0, e)
            max_start = max(0, t - int(actual_chunk_len) - min_future_steps)
            st_i = np.random.randint(0, max_start + 1)

            inputs.append(item["inputs"][env_i])
            states.append(item["states"][env_i])
            actions.append(item["actions"][env_i])
            rewards.append(item["rewards"][env_i])
            dones.append(item["dones"][env_i])
            next_states.append(item["next_states"][env_i])
            ep_idx.append(ep_i)
            start_idx.append(st_i)

        return SacChunkBatch(
            inputs=torch.tensor(np.stack(inputs), dtype=torch.float32, device=device),
            states=torch.tensor(np.stack(states), dtype=torch.float32, device=device),
            actions=torch.tensor(np.stack(actions), dtype=torch.float32, device=device),
            rewards=torch.tensor(np.stack(rewards), dtype=torch.float32, device=device),
            dones=torch.tensor(np.stack(dones), dtype=torch.float32, device=device),
            next_states=torch.tensor(np.stack(next_states), dtype=torch.float32, device=device),
            episode_idx=torch.tensor(ep_idx, dtype=torch.long, device=device),
            start_idx=torch.tensor(start_idx, dtype=torch.long, device=device),
            chunk_len=int(actual_chunk_len or 1),
        )


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for src, tgt in zip(source.parameters(), target.parameters()):
            tgt.data.mul_(1.0 - tau).add_(tau * src.data)


def _sample_previous_indices(ep_idx: int, sample_k: int, mode: str) -> list[int]:
    if ep_idx <= 0:
        return []
    if sample_k is None or sample_k <= 0 or sample_k >= ep_idx:
        return list(range(ep_idx))
    k = int(sample_k)
    if mode == "last":
        return list(range(ep_idx - k, ep_idx))
    if mode == "recent":
        ids = np.arange(ep_idx)
        weights = np.arange(1, ep_idx + 1, dtype=np.float64)
        weights = weights / weights.sum()
        chosen = np.random.choice(ids, size=k, replace=False, p=weights)
        return sorted(int(x) for x in chosen)
    # uniform
    chosen = np.random.choice(np.arange(ep_idx), size=k, replace=False)
    return sorted(int(x) for x in chosen)


def _slice_window(x: torch.Tensor, end_step: int, context_seq_len: int) -> torch.Tensor:
    # x: (E,T,D) or (T,D). Returns window ending at end_step inclusive.
    if context_seq_len is None or context_seq_len <= 0:
        start = 0
    else:
        start = max(0, int(end_step) + 1 - int(context_seq_len))
    return x[start : int(end_step) + 1]


def _prev_episode_window(x_ep: torch.Tensor, context_seq_len: int, mode: str) -> torch.Tensor:
    # x_ep: (T,D). For previous completed episodes.
    T = x_ep.shape[0]
    if context_seq_len is None or context_seq_len <= 0 or context_seq_len >= T:
        return x_ep
    w = int(context_seq_len)
    if mode == "random":
        start = np.random.randint(0, T - w + 1)
    else:
        start = T - w
    return x_ep[start : start + w]


def _encode_sequence_final(context_model, seq: torch.Tensor, grad: bool) -> torch.Tensor:
    # seq: (B,W,D), returns (B,H).
    if grad:
        x = context_model.input_encoder(seq)
        h = context_model.model(inputs_embeds=x, use_cache=False, return_dict=True).last_hidden_state
        return h[:, -1, :]
    with torch.no_grad():
        x = context_model.input_encoder(seq)
        h = context_model.model(inputs_embeds=x, use_cache=False, return_dict=True).last_hidden_state
        return h[:, -1, :].detach()


def _encode_sequence_all(context_model, seq: torch.Tensor, grad: bool) -> torch.Tensor:
    # seq: (B,W,D), returns (B,W,H).
    if grad:
        x = context_model.input_encoder(seq)
        return context_model.model(inputs_embeds=x, use_cache=False, return_dict=True).last_hidden_state
    with torch.no_grad():
        x = context_model.input_encoder(seq)
        return context_model.model(inputs_embeds=x, use_cache=False, return_dict=True).last_hidden_state.detach()


def _right_pad_windows(windows: list[torch.Tensor], device: torch.device, dtype: torch.dtype):
    """Right-pad variable-length causal windows and return real lengths.

    Do not left-pad TTT inputs.  A standard attention_mask would not be enough
    here because the TTT layer treats every input token as part of its internal
    test-time learning/update trajectory.  Instead, put padding only after the
    real tokens and gather hidden states at the last real token.  Since the TTT
    block and optional pre-conv are causal, later right-padding cannot affect
    hidden states at earlier real positions.
    """
    if len(windows) == 0:
        raise ValueError("_right_pad_windows requires at least one window")
    lengths = torch.tensor([int(w.shape[0]) for w in windows], dtype=torch.long, device=device)
    max_w = int(lengths.max().item())
    padded = []
    for win in windows:
        if win.shape[0] <= 0:
            raise ValueError("TTT context window length must be positive")
        if win.shape[0] < max_w:
            pad = torch.zeros(max_w - win.shape[0], win.shape[1], device=device, dtype=dtype)
            win = torch.cat([win, pad], dim=0)
        padded.append(win)
    return torch.stack(padded, dim=0), lengths


def _encode_window_finals(context_model, windows: list[torch.Tensor], grad: bool, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Encode variable-length windows and gather the final real-token hidden."""
    batch, lengths = _right_pad_windows(windows, device, dtype)
    h_all = _encode_sequence_all(context_model, batch, grad=grad)
    bidx = torch.arange(h_all.shape[0], device=h_all.device)
    return h_all[bidx, lengths - 1]


def _aggregate_z(context_model, prev_finals: Optional[torch.Tensor], h_cur: torch.Tensor, ep_idx: torch.Tensor):
    """Aggregate previous finals and current hidden.

    prev_finals: None or padded (B,K,H), where K may be 0.
    h_cur: (B,H)
    ep_idx: (B,) original number of previous episodes, used for mean correction.
    """
    B, H = h_cur.shape
    if prev_finals is None or prev_finals.shape[1] == 0:
        return h_cur
    agg = getattr(context_model, "aggregator_type", "mean")
    if agg == "ema":
        z_list = []
        for b in range(B):
            mem = context_model._ema_from_finals(prev_finals[b : b + 1])
            z_list.append(context_model.ema_beta * mem.squeeze(0) + (1.0 - context_model.ema_beta) * h_cur[b])
        return torch.stack(z_list, dim=0)
    # mean. If sampled previous episodes were used, prev_finals.mean estimates all previous.
    prev_mean = prev_finals.mean(dim=1)
    e = ep_idx.float().clamp(min=1.0).view(B, 1)
    return (e * prev_mean + h_cur) / (e + 1.0)


def compute_context_for_indices(
    context_model,
    inputs: torch.Tensor,
    episode_idx: torch.Tensor,
    step_idx: torch.Tensor,
    context_episode_sample: int,
    context_episode_sample_mode: str,
    context_seq_len: int,
    prev_context_window_mode: str,
    detach_previous: bool = True,
    grad_current: bool = True,
) -> torch.Tensor:
    """Compute z_t for sampled transition indices from full-trial tensors.

    inputs: (B,E,T,D). episode_idx/step_idx are (B,). The current episode
    context is the causal window ending at step_idx. Previous episodes are
    represented by selected context windows and aggregated with mean/EMA.

    Efficiency note:
        Previous-episode windows for every item in the SAC batch are collected
        into one tensor and encoded by TTT in a single batched forward. This
        avoids doing B * K tiny TTT forwards when batch size and context sample
        count are large. The sampled indices are still selected per transition.
    """
    B, E, T, D = inputs.shape

    # Select previous episode ids per sample first. _sample_previous_indices
    # returns sorted ids, which preserves chronology for EMA aggregation.
    prev_ids_per_b = []
    prev_windows = []
    prev_owner = []
    for b in range(B):
        ep = int(episode_idx[b].item())
        ids = _sample_previous_indices(ep, context_episode_sample, context_episode_sample_mode)
        prev_ids_per_b.append(ids)
        for prev_ep in ids:
            win = _prev_episode_window(inputs[b, prev_ep], context_seq_len, prev_context_window_mode)
            prev_windows.append(win)
            prev_owner.append(b)

    # Encode all previous episode windows in one batched TTT call.  Use
    # right-padding and gather the final real token; never left-pad because TTT
    # would treat pad tokens as actual inner-update tokens.
    prev_finals_per_b = [[] for _ in range(B)]
    if len(prev_windows) > 0:
        prev_finals = _encode_window_finals(
            context_model,
            prev_windows,
            grad=not detach_previous,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        for j, owner in enumerate(prev_owner):
            prev_finals_per_b[owner].append(prev_finals[j])

    # Encode current causal windows in one batched TTT call.  Different samples
    # may have different prefix lengths near the beginning of an episode.
    # Right-pad, then gather each sample's last real-token hidden state.
    cur_windows = []
    for b in range(B):
        st = int(step_idx[b].item())
        start = 0 if context_seq_len <= 0 else max(0, st + 1 - context_seq_len)
        win = inputs[b, int(episode_idx[b].item()), start : st + 1]
        cur_windows.append(win)
    h_cur = _encode_window_finals(
        context_model,
        cur_windows,
        grad=grad_current,
        device=inputs.device,
        dtype=inputs.dtype,
    )

    z = []
    agg = getattr(context_model, "aggregator_type", "mean")
    for b in range(B):
        finals = prev_finals_per_b[b]
        if len(finals) == 0:
            z.append(h_cur[b])
            continue
        prev = torch.stack(finals, dim=0).unsqueeze(0)  # (1,K,H)
        if agg == "ema":
            mem = context_model._ema_from_finals(prev).squeeze(0)
            z.append(context_model.ema_beta * mem + (1.0 - context_model.ema_beta) * h_cur[b])
        else:
            # For mean aggregation, sampled previous episodes estimate the mean
            # over all previous episodes. The original episode index is the
            # number of previous episodes that exist.
            prev_mean = prev.mean(dim=1).squeeze(0)
            e = float(max(1, int(episode_idx[b].item())))
            z.append((e * prev_mean + h_cur[b]) / (e + 1.0))
    return torch.stack(z, dim=0)


def _aggregate_z_sequence(
    context_model,
    prev_finals_per_b,
    h_seq: torch.Tensor,
    episode_idx: torch.Tensor,
) -> torch.Tensor:
    """Aggregate previous episode finals with per-step current hidden states.

    h_seq is (B,N,H), where N is the number of chunk timesteps.  Previous
    episode memory is shared across the chunk, but the current hidden state is
    timestep-specific.
    """
    B, N, H = h_seq.shape
    agg = getattr(context_model, "aggregator_type", "mean")
    z_rows = []
    for b in range(B):
        finals = prev_finals_per_b[b]
        if len(finals) == 0:
            z_rows.append(h_seq[b])
            continue

        prev = torch.stack(finals, dim=0).unsqueeze(0)  # (1,K,H)
        if agg == "ema":
            mem = context_model._ema_from_finals(prev).squeeze(0)
            z_rows.append(context_model.ema_beta * mem.view(1, H) + (1.0 - context_model.ema_beta) * h_seq[b])
        else:
            # Mean aggregation treats sampled previous episodes as an estimate
            # of all previous episode finals. episode_idx is the number of
            # previous episodes that exist for this sampled current episode.
            prev_mean = prev.mean(dim=1).squeeze(0)
            e = float(max(1, int(episode_idx[b].item())))
            z_rows.append((e * prev_mean.view(1, H) + h_seq[b]) / (e + 1.0))
    return torch.stack(z_rows, dim=0)


def compute_context_for_chunk_steps(
    context_model,
    inputs: torch.Tensor,
    episode_idx: torch.Tensor,
    start_idx: torch.Tensor,
    chunk_len: int,
    context_episode_sample: int,
    context_episode_sample_mode: str,
    context_seq_len: int,
    prev_context_window_mode: str,
    detach_previous: bool = True,
    grad_current: bool = True,
    include_next: bool = False,
):
    """Compute exact per-timestep z for a sampled consecutive chunk.

    For each sampled episode chunk starting at t with length C, this encodes one
    extended causal current-episode sequence from the context boundary through
    t+C-1.  The hidden states corresponding to t, t+1, ..., t+C-1 are gathered
    and aggregated with the same previous-episode memory.  This avoids the old
    approximation that repeated z_t for every transition in the chunk.

    If include_next=True, the same extended forward also runs through t+C when
    available and returns z_{t+1}, ..., z_{t+C} for SAC target computation,
    clamping the final next index to T-1 just like the original transition-wise
    implementation.
    """
    B, E, T, D = inputs.shape
    C = max(1, int(chunk_len))

    # Previous completed episodes are shared across all timesteps in the sampled
    # chunk, so encode them once per sampled episode, not once per transition.
    prev_windows = []
    prev_owner = []
    for b in range(B):
        ep = int(episode_idx[b].item())
        ids = _sample_previous_indices(ep, context_episode_sample, context_episode_sample_mode)
        for prev_ep in ids:
            win = _prev_episode_window(inputs[b, prev_ep], context_seq_len, prev_context_window_mode)
            prev_windows.append(win)
            prev_owner.append(b)

    prev_finals_per_b = [[] for _ in range(B)]
    if len(prev_windows) > 0:
        prev_finals = _encode_window_finals(
            context_model,
            prev_windows,
            grad=not detach_previous,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        for j, owner in enumerate(prev_owner):
            prev_finals_per_b[owner].append(prev_finals[j])

    # Current episode: encode one extended causal window per sampled chunk, then
    # gather the hidden state for each transition.  Right padding is used so
    # gathered real-token hidden states do not receive artificial zero tokens
    # before them.
    cur_windows = []
    ext_starts = []
    max_w = 0
    extra = 1 if include_next else 0
    for b in range(B):
        ep = int(episode_idx[b].item())
        st = int(start_idx[b].item())
        if context_seq_len is None or int(context_seq_len) <= 0:
            ext_start = 0
        else:
            ext_start = max(0, st + 1 - int(context_seq_len))
        ext_end = min(T - 1, st + C - 1 + extra)
        win = inputs[b, ep, ext_start : ext_end + 1]
        cur_windows.append(win)
        ext_starts.append(ext_start)
        max_w = max(max_w, int(win.shape[0]))

    padded = []
    for win in cur_windows:
        if win.shape[0] < max_w:
            pad = torch.zeros(max_w - win.shape[0], win.shape[1], device=inputs.device, dtype=inputs.dtype)
            win = torch.cat([win, pad], dim=0)
        padded.append(win)
    cur_batch = torch.stack(padded, dim=0)
    h_all = _encode_sequence_all(context_model, cur_batch, grad=grad_current)

    device = inputs.device
    offsets = torch.arange(C, device=device).view(1, C)
    starts = start_idx.view(B, 1)
    ext_start_t = torch.tensor(ext_starts, dtype=torch.long, device=device).view(B, 1)
    bidx = torch.arange(B, device=device).view(B, 1).expand(B, C)

    current_steps = torch.clamp(starts + offsets, max=T - 1)
    current_pos = current_steps - ext_start_t
    h_cur = h_all[bidx, current_pos]  # (B,C,H)
    z_cur = _aggregate_z_sequence(context_model, prev_finals_per_b, h_cur, episode_idx)

    if not include_next:
        return z_cur.reshape(B * C, z_cur.shape[-1])

    next_steps = torch.clamp(starts + offsets + 1, max=T - 1)
    next_pos = next_steps - ext_start_t
    h_next = h_all[bidx, next_pos]
    z_next = _aggregate_z_sequence(context_model, prev_finals_per_b, h_next, episode_idx)
    return z_cur.reshape(B * C, z_cur.shape[-1]), z_next.reshape(B * C, z_next.shape[-1])


def _repeat_chunk_context(z_base: torch.Tensor, chunk_len: int) -> torch.Tensor:
    """Repeat one context vector per sampled episode across its chunk."""
    B, H = z_base.shape
    C = int(chunk_len)
    return z_base[:, None, :].expand(B, C, H).reshape(B * C, H)


def _chunk_indices(batch: SacChunkBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return broadcasted indices for flattened chunk gathering."""
    B = batch.inputs.shape[0]
    C = int(batch.chunk_len)
    device = batch.inputs.device
    arange_b = torch.arange(B, device=device)[:, None].expand(B, C)
    ep = batch.episode_idx[:, None].expand(B, C)
    offsets = torch.arange(C, device=device)[None, :]
    steps = batch.start_idx[:, None] + offsets
    return arange_b, ep, steps, offsets


def _gather_sac_chunk_tensors(batch: SacChunkBatch):
    """Flatten B chunks of length C into an effective B*C SAC batch."""
    bidx, epidx, steps, _ = _chunk_indices(batch)
    obs = batch.states[bidx, epidx, steps].reshape(-1, batch.states.shape[-1])
    actions = batch.actions[bidx, epidx, steps].reshape(-1, batch.actions.shape[-1])
    rewards = batch.rewards[bidx, epidx, steps].reshape(-1, 1)
    dones = batch.dones[bidx, epidx, steps].reshape(-1, 1)
    next_obs = batch.next_states[bidx, epidx, steps].reshape(-1, batch.next_states.shape[-1])
    return obs, actions, rewards, dones, next_obs


def _episode_batch_size_from_args(args) -> int:
    v = int(getattr(args, "sac_episode_batch_size", 0) or 0)
    if v > 0:
        return v
    chunk = max(1, int(getattr(args, "sac_chunk_steps", 1) or 1))
    return max(1, int(getattr(args, "sac_batch_size", 256)) // chunk)


def compute_forecast_loss_chunked(
    context_model,
    forecaster: KStepForecaster,
    batch: SacChunkBatch,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Forecast loss using one exact current-context forward per chunk.

    For B sampled episodes and C consecutive starts per episode, this produces
    B*C forecasting examples.  The current episode is encoded once over the
    extended causal chunk window, then the corresponding z_t is gathered for
    each transition.
    """
    horizon = int(args.sac_forecast_horizon)
    if horizon <= 0:
        zero = batch.inputs.new_tensor(0.0)
        return zero, {"forecast_obs_loss": 0.0, "forecast_reward_loss": 0.0}

    B, E, T, D = batch.inputs.shape
    C = int(batch.chunk_len)
    if T < horizon or C <= 0:
        zero = batch.inputs.new_tensor(0.0)
        return zero, {"forecast_obs_loss": 0.0, "forecast_reward_loss": 0.0}

    z = compute_context_for_chunk_steps(
        context_model,
        batch.inputs,
        batch.episode_idx,
        batch.start_idx,
        C,
        context_episode_sample=int(args.sac_context_episode_sample),
        context_episode_sample_mode=args.context_episode_sample_mode,
        context_seq_len=int(args.context_seq_len),
        prev_context_window_mode=args.prev_context_window_mode,
        detach_previous=bool(args.sac_detach_previous_context),
        grad_current=True,
        include_next=False,
    )

    bidx, epidx, steps, _ = _chunk_indices(batch)
    obs0 = batch.states[bidx, epidx, steps].reshape(B * C, batch.states.shape[-1])

    action_seq = []
    target_obs = []
    target_rewards = []
    for k in range(horizon):
        sk = steps + k
        action_seq.append(batch.actions[bidx, epidx, sk].reshape(B * C, batch.actions.shape[-1]))
        target_obs.append(batch.next_states[bidx, epidx, sk].reshape(B * C, batch.next_states.shape[-1]))
        target_rewards.append(batch.rewards[bidx, epidx, sk].reshape(B * C))
    action_seq = torch.stack(action_seq, dim=1)
    target_obs = torch.stack(target_obs, dim=1)
    target_rewards = torch.stack(target_rewards, dim=1)

    pred_obs, pred_rewards = forecaster(z, obs0, action_seq)
    obs_loss = F.mse_loss(pred_obs, target_obs)
    reward_loss = F.mse_loss(pred_rewards, target_rewards)
    loss = args.sac_forecast_obs_coef * obs_loss + args.sac_forecast_reward_coef * reward_loss
    return loss, {
        "forecast_obs_loss": float(obs_loss.detach().item()),
        "forecast_reward_loss": float(reward_loss.detach().item()),
    }


def compute_forecast_loss(
    context_model,
    forecaster: KStepForecaster,
    batch: SacBatch,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    horizon = int(args.sac_forecast_horizon)
    if horizon <= 0:
        zero = batch.inputs.new_tensor(0.0)
        return zero, {"forecast_obs_loss": 0.0, "forecast_reward_loss": 0.0}

    # Resample valid starts so t + horizon is inside the same episode.
    B, E, T, D = batch.inputs.shape
    max_start = max(0, T - horizon - 1)
    if max_start <= 0:
        zero = batch.inputs.new_tensor(0.0)
        return zero, {"forecast_obs_loss": 0.0, "forecast_reward_loss": 0.0}
    ep_idx = torch.randint(0, E, (B,), device=batch.inputs.device)
    step_idx = torch.randint(0, max_start + 1, (B,), device=batch.inputs.device)

    z = compute_context_for_indices(
        context_model,
        batch.inputs,
        ep_idx,
        step_idx,
        context_episode_sample=int(args.sac_context_episode_sample),
        context_episode_sample_mode=args.context_episode_sample_mode,
        context_seq_len=int(args.context_seq_len),
        prev_context_window_mode=args.prev_context_window_mode,
        detach_previous=bool(args.sac_detach_previous_context),
        grad_current=True,
    )

    obs0 = batch.states[torch.arange(B, device=batch.inputs.device), ep_idx, step_idx]
    action_seq = []
    target_obs = []
    target_rewards = []
    for k in range(horizon):
        action_seq.append(batch.actions[torch.arange(B, device=batch.inputs.device), ep_idx, step_idx + k])
        target_obs.append(batch.next_states[torch.arange(B, device=batch.inputs.device), ep_idx, step_idx + k])
        target_rewards.append(batch.rewards[torch.arange(B, device=batch.inputs.device), ep_idx, step_idx + k])
    action_seq = torch.stack(action_seq, dim=1)
    target_obs = torch.stack(target_obs, dim=1)
    target_rewards = torch.stack(target_rewards, dim=1)

    pred_obs, pred_rewards = forecaster(z, obs0, action_seq)
    obs_loss = F.mse_loss(pred_obs, target_obs)
    reward_loss = F.mse_loss(pred_rewards, target_rewards)
    loss = args.sac_forecast_obs_coef * obs_loss + args.sac_forecast_reward_coef * reward_loss
    return loss, {
        "forecast_obs_loss": float(obs_loss.detach().item()),
        "forecast_reward_loss": float(reward_loss.detach().item()),
    }


def sac_update(
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
    first_update: bool = False,
):
    """Run chunked forecasting-then-SAC training blocks.

    The old update path sampled individual transitions and recomputed TTT
    context for each one.  This path samples B episodes and a contiguous chunk
    of C timesteps from each episode.  It encodes one extended current-episode
    sequence per sampled chunk, then gathers the correct per-timestep contexts
    z_t, z_{t+1}, ... from that single forward.  Effective transition batch
    size is B*C, but current-context encoder batch size is only B.
    """
    stats_sum: Dict[str, float] = {}
    sac_steps = 0
    forecast_steps = 0
    initial_forecast_steps = 0

    train_epochs = max(1, int(getattr(args, "sac_train_epochs", 1)))
    episode_batch_size = _episode_batch_size_from_args(args)
    chunk_steps = max(1, int(getattr(args, "sac_chunk_steps", 1) or 1))
    effective_batch_size = episode_batch_size * chunk_steps

    def add_stat(name: str, value: float) -> None:
        stats_sum[name] = stats_sum.get(name, 0.0) + float(value)

    def run_forecast_step() -> None:
        nonlocal forecast_steps
        horizon = int(args.sac_forecast_horizon)
        min_future = max(0, horizon - 1)
        batch = replay.sample_episode_chunks(episode_batch_size, chunk_steps, device, min_future_steps=min_future)
        loss, fstats = compute_forecast_loss_chunked(context_model, forecaster, batch, args)
        optimizers["context_forecast"].zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(context_model.parameters()) + list(forecaster.parameters()),
            args.max_grad_norm,
        )
        optimizers["context_forecast"].step()
        add_stat("forecast_loss", float(loss.detach().item()))
        for k, v in fstats.items():
            add_stat(k, v)
        forecast_steps += 1

    if first_update:
        requested_initial = int(getattr(args, "sac_initial_forecast_epochs", 0))
        if requested_initial < 0:
            requested_initial = max(0, 5 * int(args.sac_forecast_epochs))
        for _ in range(requested_initial):
            run_forecast_step()
            initial_forecast_steps += 1

    for _epoch in range(train_epochs):
        # Forecasting updates: train TTT/context + forecaster before actor/critic.
        for _ in range(int(args.sac_forecast_epochs)):
            run_forecast_step()

        for _ in range(int(args.sac_updates_per_rollout)):
            batch = replay.sample_episode_chunks(episode_batch_size, chunk_steps, device, min_future_steps=0)
            B = batch.inputs.shape[0]
            C = int(batch.chunk_len)

            grad_ttt = bool(args.sac_update_ttt_with_sac)
            z_target_cached = None
            if grad_ttt:
                z = compute_context_for_chunk_steps(
                    context_model,
                    batch.inputs,
                    batch.episode_idx,
                    batch.start_idx,
                    C,
                    context_episode_sample=int(args.sac_context_episode_sample),
                    context_episode_sample_mode=args.context_episode_sample_mode,
                    context_seq_len=int(args.context_seq_len),
                    prev_context_window_mode=args.prev_context_window_mode,
                    detach_previous=bool(args.sac_detach_previous_context),
                    grad_current=True,
                    include_next=False,
                )
            else:
                z, z_target_cached = compute_context_for_chunk_steps(
                    context_model,
                    batch.inputs,
                    batch.episode_idx,
                    batch.start_idx,
                    C,
                    context_episode_sample=int(args.sac_context_episode_sample),
                    context_episode_sample_mode=args.context_episode_sample_mode,
                    context_seq_len=int(args.context_seq_len),
                    prev_context_window_mode=args.prev_context_window_mode,
                    detach_previous=True,
                    grad_current=False,
                    include_next=True,
                )
                z = z.detach()
                z_target_cached = z_target_cached.detach()

            obs, actions, rewards, dones, next_obs = _gather_sac_chunk_tensors(batch)

            with torch.no_grad():
                # Use exact per-transition next contexts from the same chunk
                # rollout: z_{t+1}, ..., z_{t+C}, clamped at the episode end.
                if z_target_cached is None:
                    _, z_target = compute_context_for_chunk_steps(
                        context_model,
                        batch.inputs,
                        batch.episode_idx,
                        batch.start_idx,
                        C,
                        context_episode_sample=int(args.sac_context_episode_sample),
                        context_episode_sample_mode=args.context_episode_sample_mode,
                        context_seq_len=int(args.context_seq_len),
                        prev_context_window_mode=args.prev_context_window_mode,
                        detach_previous=True,
                        grad_current=False,
                        include_next=True,
                    )
                else:
                    z_target = z_target_cached
                next_action, next_logp = actor.sample(next_obs, z_target)
                target_q1 = q1_target(next_obs, next_action, z_target)
                target_q2 = q2_target(next_obs, next_action, z_target)
                target_q = torch.min(target_q1, target_q2) - args.sac_alpha * next_logp
                y = args.sac_reward_scale * rewards + args.gamma * (1.0 - dones) * target_q

            # Critic update. Optionally update TTT context through critic loss.
            q1_pred = q1(obs, actions, z)
            q2_pred = q2(obs, actions, z)
            critic_loss = F.mse_loss(q1_pred, y) + F.mse_loss(q2_pred, y)
            optimizers["critic"].zero_grad(set_to_none=True)
            if grad_ttt:
                optimizers["context_sac"].zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), args.max_grad_norm)
            optimizers["critic"].step()
            if grad_ttt:
                torch.nn.utils.clip_grad_norm_(context_model.parameters(), args.max_grad_norm)
                optimizers["context_sac"].step()

            # Actor update. Recompute z if TTT was updated by critic.
            if grad_ttt:
                z_actor = compute_context_for_chunk_steps(
                    context_model,
                    batch.inputs,
                    batch.episode_idx,
                    batch.start_idx,
                    C,
                    context_episode_sample=int(args.sac_context_episode_sample),
                    context_episode_sample_mode=args.context_episode_sample_mode,
                    context_seq_len=int(args.context_seq_len),
                    prev_context_window_mode=args.prev_context_window_mode,
                    detach_previous=bool(args.sac_detach_previous_context),
                    grad_current=True,
                    include_next=False,
                )
            else:
                z_actor = z.detach()

            new_action, logp = actor.sample(obs, z_actor)
            q_pi = torch.min(q1(obs, new_action, z_actor), q2(obs, new_action, z_actor))
            actor_loss = (args.sac_alpha * logp - q_pi).mean()
            optimizers["actor"].zero_grad(set_to_none=True)
            if grad_ttt:
                optimizers["context_sac"].zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            optimizers["actor"].step()
            if grad_ttt:
                torch.nn.utils.clip_grad_norm_(context_model.parameters(), args.max_grad_norm)
                optimizers["context_sac"].step()

            soft_update(q1, q1_target, args.sac_tau)
            soft_update(q2, q2_target, args.sac_tau)

            add_stat("critic_loss", float(critic_loss.detach().item()))
            add_stat("actor_loss", float(actor_loss.detach().item()))
            add_stat("q1", float(q1_pred.detach().mean().item()))
            add_stat("q2", float(q2_pred.detach().mean().item()))
            add_stat("entropy", float((-logp.detach()).mean().item()))
            sac_steps += 1

    denom_forecast = max(1, forecast_steps)
    denom_sac = max(1, sac_steps)
    return {
        "forecast_loss": stats_sum.get("forecast_loss", 0.0) / denom_forecast,
        "forecast_obs_loss": stats_sum.get("forecast_obs_loss", 0.0) / denom_forecast,
        "forecast_reward_loss": stats_sum.get("forecast_reward_loss", 0.0) / denom_forecast,
        "critic_loss": stats_sum.get("critic_loss", 0.0) / denom_sac,
        "actor_loss": stats_sum.get("actor_loss", 0.0) / denom_sac,
        "q1": stats_sum.get("q1", 0.0) / denom_sac,
        "q2": stats_sum.get("q2", 0.0) / denom_sac,
        "entropy": stats_sum.get("entropy", 0.0) / denom_sac,
        "sac_steps": float(sac_steps),
        "forecast_steps": float(forecast_steps),
        "initial_forecast_steps": float(initial_forecast_steps),
        "sac_train_epochs": float(train_epochs),
        "sac_episode_batch_size": float(episode_batch_size),
        "sac_chunk_steps": float(chunk_steps),
        "sac_effective_batch_size": float(effective_batch_size),
    }
