from dataclasses import dataclass
from typing import Optional, Tuple, Union
import math
import numpy as np
import torch
from torch import nn

from ttt import TTTPreTrainedModel, TTTModel, TTTCache


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=nn.Tanh,
        hidden_w_init=nn.init.orthogonal_,
        hidden_b_init=nn.init.zeros_,
        output_w_init=nn.init.orthogonal_,
        output_b_init=nn.init.zeros_,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            linear = nn.Linear(prev_dim, hidden_dim)
            hidden_w_init(linear.weight)
            hidden_b_init(linear.bias)
            layers.append(linear)
            layers.append(hidden_nonlinearity())
            prev_dim = hidden_dim
        output = nn.Linear(prev_dim, output_dim)
        output_w_init(output.weight)
        output_b_init(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianMLPHead(nn.Module):
    """Gaussian MLP actor head.

    If hidden_sizes=(), this becomes a linear Gaussian head.
    """

    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=nn.Tanh,
        learn_std=True,
        init_std=1.0,
        min_std=1e-6,
        max_std=None,
        std_parameterization="exp",
    ):
        super().__init__()
        self.mean_network = MLP(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
        )
        self.learn_std = learn_std
        self.min_std = min_std
        self.max_std = max_std
        self.std_parameterization = std_parameterization

        if std_parameterization == "exp":
            init_param = math.log(init_std)
        elif std_parameterization == "softplus":
            init_param = math.log(math.exp(init_std) - 1.0)
        else:
            raise ValueError(f"Unknown std_parameterization: {std_parameterization}")

        if learn_std:
            self.std_param = nn.Parameter(torch.ones(action_dim) * init_param)
        else:
            self.register_buffer("std_param", torch.ones(action_dim) * init_param)

    def std(self):
        if self.std_parameterization == "exp":
            std = torch.exp(self.std_param)
        elif self.std_parameterization == "softplus":
            std = torch.nn.functional.softplus(self.std_param)
        else:
            raise ValueError(f"Unknown std_parameterization: {self.std_parameterization}")
        if self.min_std is not None:
            std = torch.clamp(std, min=self.min_std)
        if self.max_std is not None:
            std = torch.clamp(std, max=self.max_std)
        return std

    def forward(self, x):
        mean = self.mean_network(x)
        log_std = torch.log(self.std()).expand_as(mean)
        return mean, log_std


@dataclass
class RLModelOutput:
    policy: Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]
    value: torch.FloatTensor
    hidden_states: Optional[torch.FloatTensor] = None
    cache_params: Optional[TTTCache] = None
    time_offset: int = 0


def orthogonal_init(module, gain=np.sqrt(2.0)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_last_linear(module, gain):
    """Initialize the final Linear layer inside an MLP-like module."""
    if hasattr(module, "net"):
        layers = module.net
    elif hasattr(module, "mean_network") and hasattr(module.mean_network, "net"):
        layers = module.mean_network.net
    else:
        return
    for layer in reversed(layers):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return


class TTTEpisodePolicy(TTTPreTrainedModel):
    """TTT episode encoder with EMA or attention cross-episode aggregation."""

    def __init__(
        self,
        config,
        input_dim: int,
        obs_dim: int,
        num_actions: int,
        num_episodes: int,
        continuous: bool = True,
        policy_hidden_sizes=(64, 64),
        value_hidden_sizes=(64, 64),
        aggregator_type: str = "ema",
        ema_beta: float = 0.7,
        use_state_proj: bool = True,
        episode_attn_heads: int = 1,
        min_std: float = 0.5,
        max_std: float = 1.5,
        init_std: float = 1.0,
    ):
        super().__init__(config)
        self.model = TTTModel(config)
        self.input_dim = input_dim
        self.obs_dim = obs_dim
        self.hidden_size = config.hidden_size
        self.num_actions = num_actions
        self.num_episodes = num_episodes
        self.continuous = continuous
        self.aggregator_type = aggregator_type
        self.ema_beta = float(ema_beta)
        self.use_state_proj = use_state_proj
        self.episode_attn_heads = int(episode_attn_heads or 1)

        if self.aggregator_type not in ["ema", "attn"]:
            raise ValueError("aggregator_type must be either 'ema' or 'attn'")
        if not (0.0 <= self.ema_beta <= 1.0):
            raise ValueError("ema_beta must be between 0 and 1")
        if self.episode_attn_heads <= 0:
            raise ValueError("episode_attn_heads must be >= 1")
        if self.hidden_size % self.episode_attn_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by "
                f"episode_attn_heads={self.episode_attn_heads}"
            )

        self.input_encoder = nn.Linear(input_dim, self.hidden_size)

        if self.aggregator_type == "attn":
            self.episode_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=self.episode_attn_heads,
                batch_first=True,
            )
            self.episode_attn_norm = nn.LayerNorm(self.hidden_size)
        else:
            self.episode_attention = None
            self.episode_attn_norm = None

        if self.use_state_proj:
            self.state_proj = nn.Linear(obs_dim, self.hidden_size, bias=False)
            self.head_input_dim = self.hidden_size * 2
        else:
            self.state_proj = None
            self.head_input_dim = self.hidden_size

        if continuous:
            self.policy_head = GaussianMLPHead(
                input_dim=self.head_input_dim,
                action_dim=num_actions,
                hidden_sizes=policy_hidden_sizes,
                hidden_nonlinearity=nn.Tanh,
                learn_std=True,
                init_std=init_std,
                min_std=min_std,
                max_std=max_std,
                std_parameterization="exp",
            )
        else:
            self.policy_head = MLP(
                input_dim=self.head_input_dim,
                output_dim=num_actions,
                hidden_sizes=policy_hidden_sizes,
                hidden_nonlinearity=nn.Tanh,
            )

        self.value_head = MLP(
            input_dim=self.head_input_dim,
            output_dim=1,
            hidden_sizes=value_hidden_sizes,
            hidden_nonlinearity=nn.Tanh,
        )

        self._init_ppo_weights()

    def _init_ppo_weights(self):
        """PPO-style orthogonal initialization for non-TTT components."""
        self.input_encoder.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2.0)))
        if self.state_proj is not None:
            self.state_proj.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2.0)))
        if self.episode_attention is not None:
            self.episode_attention.out_proj.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2.0)))
        if self.continuous:
            self.policy_head.mean_network.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2.0)))
            init_last_linear(self.policy_head.mean_network, gain=0.01)
        else:
            self.policy_head.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2.0)))
            init_last_linear(self.policy_head, gain=0.01)
        self.value_head.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2.0)))
        init_last_linear(self.value_head, gain=1.0)

    def _ema_update(self, memory: torch.Tensor, new_value: torch.Tensor) -> torch.Tensor:
        return self.ema_beta * memory + (1.0 - self.ema_beta) * new_value

    def _ema_from_finals(self, finals: torch.Tensor) -> Optional[torch.Tensor]:
        """Return EMA over finals in chronological order. finals: (B,K,H)."""
        if finals.shape[1] == 0:
            return None
        memory = finals[:, 0, :]
        for idx in range(1, finals.shape[1]):
            memory = self._ema_update(memory, finals[:, idx, :])
        return memory

    def aggregate_attention(self, prev_finals: torch.Tensor, current_hidden: torch.Tensor) -> torch.Tensor:
        """Current-query attention over previous episode finals.

        prev_finals: (B,K,H), where K may be zero.
        current_hidden: (B,H) or (B,T,H).

        No positional encoding is used, so previous episode memories are treated
        as a permutation-invariant set. The output keeps a residual current path:
        LayerNorm(current_hidden + attention(previous_finals)).
        """
        if self.episode_attention is None:
            raise RuntimeError("aggregate_attention called when aggregator_type != 'attn'")
        if prev_finals.ndim != 3:
            raise ValueError(f"prev_finals must be (B,K,H), got {prev_finals.shape}")
        if current_hidden.ndim not in (2, 3):
            raise ValueError(f"current_hidden must be (B,H) or (B,T,H), got {current_hidden.shape}")

        squeeze_time = current_hidden.ndim == 2
        current_seq = current_hidden[:, None, :] if squeeze_time else current_hidden
        B, T, H = current_seq.shape
        if prev_finals.shape[0] != B or prev_finals.shape[2] != H:
            raise ValueError(
                f"prev_finals shape {prev_finals.shape} is incompatible with current_hidden shape {current_hidden.shape}"
            )
        K = prev_finals.shape[1]
        if K == 0:
            return current_hidden

        cur_flat = current_seq.reshape(B * T, 1, H)
        prev_flat = prev_finals[:, None, :, :].expand(B, T, K, H).reshape(B * T, K, H)
        attn_out, _ = self.episode_attention(
            query=cur_flat,
            key=prev_flat,
            value=prev_flat,
            need_weights=False,
        )
        out = self.episode_attn_norm(cur_flat + attn_out).reshape(B, T, H)
        if squeeze_time:
            return out[:, 0, :]
        return out

    def aggregate_full_trial_ema(self, episode_hidden: torch.Tensor) -> torch.Tensor:
        B, E, T, H = episode_hidden.shape
        outputs = []
        final_embeddings = episode_hidden[:, :, -1, :]
        prev_memory = None
        for ep_idx in range(E):
            current_seq = episode_hidden[:, ep_idx, :, :]
            if prev_memory is None:
                z = current_seq
            else:
                z = self.ema_beta * prev_memory[:, None, :] + (1.0 - self.ema_beta) * current_seq
            outputs.append(z)
            current_final = final_embeddings[:, ep_idx, :]
            prev_memory = current_final if prev_memory is None else self._ema_update(prev_memory, current_final)
        return torch.stack(outputs, dim=1)

    def aggregate_full_trial_attention(self, episode_hidden: torch.Tensor) -> torch.Tensor:
        B, E, T, H = episode_hidden.shape
        outputs = []
        final_embeddings = episode_hidden[:, :, -1, :]
        for ep_idx in range(E):
            current_seq = episode_hidden[:, ep_idx, :, :]
            prev_finals = final_embeddings[:, :ep_idx, :]
            outputs.append(self.aggregate_attention(prev_finals, current_seq))
        return torch.stack(outputs, dim=1)

    def aggregate_full_trial(self, episode_hidden: torch.Tensor) -> torch.Tensor:
        if self.aggregator_type == "ema":
            return self.aggregate_full_trial_ema(episode_hidden)
        if self.aggregator_type == "attn":
            return self.aggregate_full_trial_attention(episode_hidden)
        raise ValueError(f"Unknown aggregator_type: {self.aggregator_type}")

    def aggregate_step_ema(self, episode_memory: torch.Tensor, current_hidden: torch.Tensor, episode_idx: int) -> torch.Tensor:
        if episode_idx <= 0:
            return current_hidden
        prev_memory = self._ema_from_finals(episode_memory[:, :episode_idx, :])
        return self.ema_beta * prev_memory + (1.0 - self.ema_beta) * current_hidden

    def aggregate_step_attention(self, episode_memory: torch.Tensor, current_hidden: torch.Tensor, episode_idx: int) -> torch.Tensor:
        prev_finals = episode_memory[:, :episode_idx, :]
        return self.aggregate_attention(prev_finals, current_hidden)

    def aggregate_step(self, episode_memory: torch.Tensor, current_hidden: torch.Tensor, episode_idx: int) -> torch.Tensor:
        if self.aggregator_type == "ema":
            return self.aggregate_step_ema(episode_memory, current_hidden, episode_idx)
        if self.aggregator_type == "attn":
            return self.aggregate_step_attention(episode_memory, current_hidden, episode_idx)
        raise ValueError(f"Unknown aggregator_type: {self.aggregator_type}")

    def _heads(self, z_task: torch.Tensor, current_obs: torch.Tensor):
        if self.use_state_proj:
            state_feat = self.state_proj(current_obs)
            head_input = torch.cat([z_task, state_feat], dim=-1)
        else:
            head_input = z_task
        policy_out = self.policy_head(head_input)
        value = self.value_head(head_input).squeeze(-1)
        return policy_out, value

    def get_policy_std(self):
        if not self.continuous:
            return None
        return self.policy_head.std()

    def forward(self, agent_inputs: torch.Tensor, current_obs: torch.Tensor, return_dict: bool = True):
        """Full-trial forward. Retained for debugging/evaluation utilities."""
        if agent_inputs.ndim != 4:
            raise ValueError(f"agent_inputs must be (B,E,T,D), got {agent_inputs.shape}")
        B, E, T, D = agent_inputs.shape
        H = self.hidden_size
        x = self.input_encoder(agent_inputs.reshape(B * E, T, D))
        base_outputs = self.model(inputs_embeds=x, use_cache=False, return_dict=True)
        episode_hidden = base_outputs.last_hidden_state.reshape(B, E, T, H)
        z_task = self.aggregate_full_trial(episode_hidden)
        policy_out, value = self._heads(z_task, current_obs)
        if not return_dict:
            return policy_out, value, episode_hidden
        return RLModelOutput(policy=policy_out, value=value, hidden_states=episode_hidden, cache_params=None)

    def _encode_episode_final(self, episode_inputs: torch.Tensor) -> torch.Tensor:
        """Encode a full episode and return the final hidden state."""
        x = self.input_encoder(episode_inputs)
        h = self.model(inputs_embeds=x, use_cache=False, return_dict=True).last_hidden_state
        return h[:, -1, :]

    def forward_prefix_flat(
        self,
        agent_inputs: torch.Tensor,
        current_obs: torch.Tensor,
        last_episode: int,
        last_step: int,
        context_episode_indices=None,
        return_dict: bool = True,
    ):
        """Forward the causal prefix needed for a PPO chunk.

        If context_episode_indices is None, all previous episodes are forwarded
        fully and the current episode is forwarded through last_step. If
        context_episode_indices is provided, only those previous full episodes
        are encoded as context and only the current episode prefix is returned.
        """
        if agent_inputs.ndim != 4:
            raise ValueError(f"agent_inputs must be (B,E,T,D), got {agent_inputs.shape}")
        B, E, T, D = agent_inputs.shape
        if last_episode < 0 or last_episode >= E:
            raise ValueError(f"last_episode={last_episode} is out of range for E={E}")
        if last_step < 0 or last_step >= T:
            raise ValueError(f"last_step={last_step} is out of range for T={T}")

        prefix_len = last_step + 1

        if context_episode_indices is not None:
            prev_finals = []
            for ep_idx in context_episode_indices:
                if ep_idx < 0 or ep_idx >= last_episode:
                    raise ValueError(f"context episode {ep_idx} must be in [0, {last_episode})")
                prev_finals.append(self._encode_episode_final(agent_inputs[:, ep_idx, :, :]))

            x_cur = self.input_encoder(agent_inputs[:, last_episode, :prefix_len, :])
            h_cur = self.model(inputs_embeds=x_cur, use_cache=False, return_dict=True).last_hidden_state

            if len(prev_finals) == 0:
                prev_finals_tensor = h_cur.new_zeros(B, 0, self.hidden_size)
            else:
                prev_finals_tensor = torch.stack(prev_finals, dim=1)

            if self.aggregator_type == "attn":
                z_task = self.aggregate_attention(prev_finals_tensor, h_cur)
            else:
                prev_memory = self._ema_from_finals(prev_finals_tensor)
                if prev_memory is None:
                    z_task = h_cur
                else:
                    z_task = self.ema_beta * prev_memory[:, None, :] + (1.0 - self.ema_beta) * h_cur

            obs_prefix = current_obs[:, last_episode, :prefix_len, :]
            policy_out, value = self._heads(z_task, obs_prefix)
            if not return_dict:
                return policy_out, value, h_cur
            return RLModelOutput(policy=policy_out, value=value, hidden_states=h_cur, cache_params=None, time_offset=0)

        h_list = []
        obs_list = []
        for ep_idx in range(last_episode + 1):
            length = T if ep_idx < last_episode else prefix_len
            x_ep = self.input_encoder(agent_inputs[:, ep_idx, :length, :])
            h_ep = self.model(inputs_embeds=x_ep, use_cache=False, return_dict=True).last_hidden_state
            h_list.append(h_ep)
            obs_list.append(current_obs[:, ep_idx, :length, :])

        if self.aggregator_type == "ema":
            z_list = []
            prev_memory = None
            for h_ep in h_list:
                if prev_memory is None:
                    z_ep = h_ep
                else:
                    z_ep = self.ema_beta * prev_memory[:, None, :] + (1.0 - self.ema_beta) * h_ep
                z_list.append(z_ep)
                prev_memory = h_ep[:, -1, :] if prev_memory is None else self._ema_update(prev_memory, h_ep[:, -1, :])
        else:
            z_list = []
            prev_finals = []
            for h_ep in h_list:
                if len(prev_finals) == 0:
                    prev_finals_tensor = h_ep.new_zeros(B, 0, self.hidden_size)
                else:
                    prev_finals_tensor = torch.stack(prev_finals, dim=1)
                z_ep = self.aggregate_attention(prev_finals_tensor, h_ep)
                z_list.append(z_ep)
                prev_finals.append(h_ep[:, -1, :])

        z_task = torch.cat(z_list, dim=1)
        obs_prefix = torch.cat(obs_list, dim=1)
        hidden_prefix = torch.cat(h_list, dim=1)
        policy_out, value = self._heads(z_task, obs_prefix)
        if not return_dict:
            return policy_out, value, hidden_prefix
        return RLModelOutput(policy=policy_out, value=value, hidden_states=hidden_prefix, cache_params=None)

    @torch.no_grad()
    def init_episode_memory(self, batch_size: int, device=None, num_episodes: Optional[int] = None):
        device = device or next(self.parameters()).device
        if num_episodes is None:
            num_episodes = self.num_episodes
        return torch.zeros(batch_size, num_episodes, self.hidden_size, device=device)

    def act_step(
        self,
        agent_input: torch.Tensor,
        current_obs: torch.Tensor,
        episode_memory: torch.Tensor,
        episode_idx: int,
        cache_params: Optional[TTTCache] = None,
    ):
        """One rollout/eval step using cached full-prefix current episode context."""
        x = self.input_encoder(agent_input[:, None, :])
        base_outputs = self.model(
            inputs_embeds=x,
            cache_params=cache_params,
            use_cache=True,
            return_dict=True,
        )
        current_hidden = base_outputs.last_hidden_state[:, -1, :]
        z_task = self.aggregate_step(episode_memory, current_hidden, episode_idx)
        policy_out, value = self._heads(z_task, current_obs)
        return RLModelOutput(
            policy=policy_out,
            value=value,
            cache_params=base_outputs.cache_params,
            hidden_states=current_hidden,
        )
