import numpy as np
import warnings
import torch
import torch.nn as nn
import gymnasium as gym


class GoalMaskedEnv(gym.Wrapper):
    """Removes goal coordinates from observations."""
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        if hasattr(env.unwrapped, "goal_space"):
            goal_dim = env.unwrapped.goal_space.shape[0]
        else:
            goal_dim = 3
        self.proprio_dim = obs_space.shape[0] - goal_dim
        self.observation_space = gym.spaces.Box(
            low=obs_space.low[:self.proprio_dim],
            high=obs_space.high[:self.proprio_dim],
            dtype=obs_space.dtype,
        )

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)

        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        return obs[:self.proprio_dim], info

    def step(self, action):
        out = self.env.step(action)

        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = False
        else:
            raise RuntimeError(f"Unexpected step return format: {type(out)} / {out}")

        return obs[:self.proprio_dim], reward, terminated, truncated, info


class MetaWorldTaskSamplerEnv(gym.Env):
    """A single env that samples Meta-World task classes/variations like ECET.

    It can switch between ML10/ML45 classes because those tasks share spaces.
    """
    metadata = {"render_modes": []}

    def __init__(self, classes, tasks, seed=None, mask_goal=True, render_kwargs=None):
        super().__init__()
        self.classes = dict(classes)
        self.tasks = list(tasks)
        self.tasks_by_env = {}
        for task in self.tasks:
            self.tasks_by_env.setdefault(task.env_name, []).append(task)
        self.env_names = list(self.classes.keys())
        self.rng = np.random.default_rng(seed)
        self.mask_goal = mask_goal
        self.render_kwargs = render_kwargs or {}
        self.env_name = None
        self.env = None
        self.sample_new_task()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def _make_env(self, env_name):
        env_cls = self.classes[env_name]
        env = env_cls(**self.render_kwargs)
        if self.mask_goal:
            env = GoalMaskedEnv(env)
        return env

    def sample_new_task(self, env_name=None):
        # If env_name is provided, sample only a variation from that class.
        # If env_name is None, sample class uniformly at random.
        if env_name is None:
            env_idx = int(self.rng.integers(len(self.env_names)))
            env_name = self.env_names[env_idx]
        else:
            if env_name not in self.classes:
                raise ValueError(f"Unknown env_name {env_name}. Available: {self.env_names}")

        task_list = self.tasks_by_env[env_name]
        task_idx = int(self.rng.integers(len(task_list)))
        task = task_list[task_idx]

        if self.env is None or self.env_name != env_name:
            if self.env is not None:
                self.env.close()
            self.env = self._make_env(env_name)
            self.env_name = env_name

        self.env.unwrapped.set_task(task)
        return env_name

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)

        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        return obs, info

    def step(self, action):
        out = self.env.step(action)

        if isinstance(out, tuple) and len(out) == 5:
            return out

        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = False
            return obs, reward, terminated, truncated, info

        raise RuntimeError(f"Unexpected step return format: {type(out)} / {out}")

    def render(self):
        return self.env.render()

    def close(self):
        if self.env is not None:
            self.env.close()


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class RewardNormalizer:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.return_rms = RunningMeanStd(shape=())
        self.returns = None

    def normalize(self, rewards, dones):
        if self.returns is None or self.returns.shape != rewards.shape:
            self.returns = np.zeros_like(rewards, dtype=np.float32)
        self.returns = self.returns * self.gamma + rewards
        self.return_rms.update(self.returns)
        self.returns = self.returns * (1.0 - dones)
        return rewards / np.sqrt(self.return_rms.var + 1e-8)



def get_agent_input(obs, prev_action, prev_reward, prev_done, agent_mode):
    if obs.ndim == 1:
        obs = obs[np.newaxis, :]
        prev_action = prev_action[np.newaxis, :]
        prev_reward = np.array([[prev_reward]], dtype=np.float32)
        prev_done = np.array([[prev_done]], dtype=np.float32)
    if agent_mode == "agent_v1":
        return obs.astype(np.float32)
    if agent_mode == "agent_v2":
        return np.concatenate([obs, prev_action, prev_reward], axis=-1).astype(np.float32)
    if agent_mode == "agent_rl2":
        return np.concatenate([obs, prev_action, prev_reward, prev_done], axis=-1).astype(np.float32)
    raise ValueError(f"Unknown agent mode: {agent_mode}")


def get_success_array(info, num_envs):
    successes = info.get("success", info.get("_success", np.zeros(num_envs, dtype=bool)))
    if isinstance(successes, list):
        successes = np.array(successes)
    if np.isscalar(successes):
        successes = np.full(num_envs, bool(successes))
    return successes.astype(bool)


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """GAE over flattened trial time.

    rewards/values/dones: (B, E, T)
    next_value: (B,), typically zeros when collecting complete trials.
    """
    B, E, T = rewards.shape
    R = rewards.reshape(B, E * T)
    V = values.reshape(B, E * T)
    D = dones.reshape(B, E * T)
    advantages = torch.zeros_like(R)
    lastgaelam = torch.zeros(B, device=R.device)
    for t in reversed(range(E * T)):
        if t == E * T - 1:
            nextvalues = next_value
        else:
            nextvalues = V[:, t + 1]
        nextnonterminal = 1.0 - D[:, t]
        delta = R[:, t] + gamma * nextvalues * nextnonterminal - V[:, t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[:, t] = lastgaelam
    returns = advantages + V
    return advantages.reshape(B, E, T), returns.reshape(B, E, T)




def _select_time(x, positions):
    """Select the same time positions for every env in a minibatch."""
    return x[:, positions]


def _ppo_step_from_sequences(
    model,
    optimizer,
    mean,
    log_std,
    values,
    actions,
    old_log_probs,
    adv,
    returns,
    positions,
    args,
):
    """Compute one PPO step on selected positions from already-forwarded sequences."""
    if positions is None:
        mean_sel = mean
        log_std_sel = log_std
        values_sel = values
        actions_sel = actions
        old_log_probs_sel = old_log_probs
        adv_sel = adv
        returns_sel = returns
    else:
        mean_sel = _select_time(mean, positions)
        log_std_sel = _select_time(log_std, positions)
        values_sel = _select_time(values, positions)
        actions_sel = _select_time(actions, positions)
        old_log_probs_sel = _select_time(old_log_probs, positions)
        adv_sel = _select_time(adv, positions)
        returns_sel = _select_time(returns, positions)

    action_dim = mean_sel.shape[-1]
    dist = torch.distributions.Normal(
        mean_sel.reshape(-1, action_dim),
        log_std_sel.reshape(-1, action_dim).exp(),
    )
    new_log_probs = dist.log_prob(actions_sel.reshape(-1, action_dim)).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1).mean()

    old_log_probs_flat = old_log_probs_sel.reshape(-1)
    adv_flat = adv_sel.reshape(-1)
    returns_flat = returns_sel.reshape(-1)
    values_flat = values_sel.reshape(-1)

    ratio = (new_log_probs - old_log_probs_flat).exp()
    surr1 = ratio * adv_flat
    surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * adv_flat
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = nn.functional.mse_loss(values_flat, returns_flat)
    loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item(), float(grad_norm)


def _concat_episode_prefix(x, env_idx, ep_idx, step_end):
    """Return (mb_envs, ep_idx*T + step_end, ...) for episodes before ep plus current prefix."""
    parts = []
    if ep_idx > 0:
        mb = x[env_idx].shape[0]
        trailing_shape = x.shape[3:]
        parts.append(x[env_idx, :ep_idx].reshape(mb, ep_idx * x.shape[2], *trailing_shape))
    parts.append(x[env_idx, ep_idx, :step_end])
    return torch.cat(parts, dim=1)


def _concat_episode_prefix_local(x, ep_idx, step_end):
    """Return (mb_envs, ep_idx*T + step_end, ...) from an already-selected env batch."""
    parts = []
    mb = x.shape[0]
    trailing_shape = x.shape[3:]
    if ep_idx > 0:
        parts.append(x[:, :ep_idx].reshape(mb, ep_idx * x.shape[2], *trailing_shape))
    parts.append(x[:, ep_idx, :step_end])
    return torch.cat(parts, dim=1)


def _sample_previous_episodes(ep_idx, sample_count, device, mode="uniform"):
    """Sample previous episode indices and return them in chronological order."""
    if sample_count is None or sample_count <= 0 or ep_idx <= 0:
        return None
    k = min(sample_count, ep_idx)
    mode = str(mode or "uniform")
    if mode == "last":
        sampled = torch.arange(ep_idx - k, ep_idx, device=device)
    elif mode == "uniform":
        sampled = torch.randperm(ep_idx, device=device)[:k].sort().values
    elif mode == "recent":
        weights = torch.arange(1, ep_idx + 1, device=device, dtype=torch.float32)
        sampled = torch.multinomial(weights, num_samples=k, replacement=False).sort().values
    else:
        raise ValueError("context_episode_sample_mode must be one of: uniform, recent, last")
    return [int(x.item()) for x in sampled]


def _unpack_ppo_rollouts(rollouts):
    """Unpack PPO rollout tuple with optional valid mask and task-class ids.

    Supported forms:
      6: (inputs, states, actions, old_log_probs, adv, returns)
      7: above + b_valid
      8: above + b_valid + b_task_ids
    """
    if len(rollouts) == 8:
        b_inputs, b_states, b_actions, b_old_log_probs, b_adv, b_returns, b_valid, b_task_ids = rollouts
    elif len(rollouts) == 7:
        b_inputs, b_states, b_actions, b_old_log_probs, b_adv, b_returns, b_valid = rollouts
        b_task_ids = None
    elif len(rollouts) == 6:
        b_inputs, b_states, b_actions, b_old_log_probs, b_adv, b_returns = rollouts
        b_valid = torch.ones_like(b_adv)
        b_task_ids = None
    else:
        raise ValueError(f"Expected PPO rollouts tuple length 6, 7, or 8, got {len(rollouts)}")
    return b_inputs, b_states, b_actions, b_old_log_probs, b_adv, b_returns, b_valid, b_task_ids


def _normalize_advantages_tensor(b_adv, b_valid, b_task_ids=None, by_class=False, device=None):
    """Normalize an advantage tensor over valid positions."""
    valid = b_valid > 0.0
    if not valid.any():
        return b_adv

    if not by_class:
        adv_mean = b_adv[valid].mean()
        adv_std = b_adv[valid].std(unbiased=False)
        return (b_adv - adv_mean) / (adv_std + 1e-8)

    if b_task_ids is None:
        raise ValueError(
            "Per-class advantage normalization requires task class ids in the PPO rollout tuple."
        )
    if not torch.is_tensor(b_task_ids):
        b_task_ids = torch.as_tensor(b_task_ids, device=device, dtype=torch.long)
    else:
        b_task_ids = b_task_ids.to(device=device, dtype=torch.long)
    if b_task_ids.ndim != 1 or b_task_ids.shape[0] != b_adv.shape[0]:
        raise ValueError(
            f"b_task_ids must have shape (B,), got {tuple(b_task_ids.shape)} for b_adv shape {tuple(b_adv.shape)}"
        )

    b_adv_norm = b_adv.clone()
    for task_id in torch.unique(b_task_ids):
        env_mask = b_task_ids == task_id
        class_mask = valid & env_mask[:, None, None]
        if not class_mask.any():
            continue
        class_adv = b_adv[class_mask]
        class_mean = class_adv.mean()
        class_std = class_adv.std(unbiased=False)
        b_adv_norm[class_mask] = (b_adv[class_mask] - class_mean) / (class_std + 1e-8)
    return b_adv_norm


def _train_ppo_sequential(model, optimizer, rollouts, args, device):
    """Chronological chunk PPO update.

    This is the only PPO update path. For each ordered chunk, it forwards only
    the causal prefix needed for that chunk: previous complete episodes plus the
    current episode prefix. The loss is computed only on the current chunk.
    """
    b_inputs, b_states, b_actions, b_old_log_probs, b_adv, b_returns, b_valid, b_task_ids = _unpack_ppo_rollouts(rollouts)
    b_adv = _normalize_advantages_tensor(
        b_adv,
        b_valid,
        b_task_ids=b_task_ids,
        by_class=getattr(args, "normalize_advantage_by_class", False),
        device=device,
    )

    num_envs, num_eps, ep_len = b_adv.shape
    mb_envs = getattr(args, "ppo_minibatch_envs", 0) or num_envs
    mb_envs = min(mb_envs, num_envs)
    mb_steps = getattr(args, "ppo_minibatch_steps", 0) or ep_len
    mb_steps = min(mb_steps, ep_len)
    context_sample = getattr(args, "ppo_context_episode_sample", 0)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_loss = 0.0
    total_minibatches = 0
    last_norm = 0.0

    for _ in range(args.ppo_epochs):
        env_perm = torch.randperm(num_envs, device=device)
        for start_idx in range(0, num_envs, mb_envs):
            env_idx = env_perm[start_idx : start_idx + mb_envs]
            b_adv_mb = b_adv[env_idx]

            for ep_idx in range(num_eps):
                for step_start in range(0, ep_len, mb_steps):
                    step_end = min(step_start + mb_steps, ep_len)
                    context_indices = _sample_previous_episodes(
                        ep_idx,
                        context_sample,
                        device,
                        mode=getattr(args, "context_episode_sample_mode", "uniform"),
                    )

                    outputs = model.forward_prefix_flat(
                        b_inputs[env_idx],
                        b_states[env_idx],
                        last_episode=ep_idx,
                        last_step=step_end - 1,
                        context_episode_indices=context_indices,
                        return_dict=True,
                    )
                    mean, log_std = outputs.policy
                    values = outputs.value

                    if context_indices is None:
                        actions_seq = _concat_episode_prefix(b_actions, env_idx, ep_idx, step_end)
                        old_seq = _concat_episode_prefix(b_old_log_probs[..., None], env_idx, ep_idx, step_end).squeeze(-1)
                        adv_seq = _concat_episode_prefix_local(b_adv_mb[..., None], ep_idx, step_end).squeeze(-1)
                        returns_seq = _concat_episode_prefix(b_returns[..., None], env_idx, ep_idx, step_end).squeeze(-1)
                        pos = torch.arange(ep_idx * ep_len + step_start, ep_idx * ep_len + step_end, device=device)
                    else:
                        actions_seq = b_actions[env_idx, ep_idx, :step_end]
                        old_seq = b_old_log_probs[env_idx, ep_idx, :step_end]
                        adv_seq = b_adv_mb[:, ep_idx, :step_end]
                        returns_seq = b_returns[env_idx, ep_idx, :step_end]
                        pos = torch.arange(step_start, step_end, device=device)

                    loss, p_loss, v_loss, ent, last_norm = _ppo_step_from_sequences(
                        model,
                        optimizer,
                        mean,
                        log_std,
                        values,
                        actions_seq,
                        old_seq,
                        adv_seq,
                        returns_seq,
                        pos,
                        args,
                    )

                    total_policy_loss += p_loss
                    total_value_loss += v_loss
                    total_entropy += ent
                    total_loss += loss
                    total_minibatches += 1

    denom = max(1, total_minibatches)
    print(
        f"PPO Update -> mode=sequential scope=chunk | "
        f"Grad Norm: {float(last_norm):.2f} | optimizer_steps={total_minibatches}"
    )
    return total_loss / denom, total_policy_loss / denom, total_value_loss / denom, total_entropy / denom


def train_ppo(model, optimizer, rollouts, args, device):
    return _train_ppo_sequential(model, optimizer, rollouts, args, device)
