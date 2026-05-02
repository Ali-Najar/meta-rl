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
        hidden_w_init=nn.init.xavier_uniform_,
        hidden_b_init=nn.init.zeros_,
        output_w_init=nn.init.xavier_uniform_,
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

    If hidden_sizes=(), this becomes a linear Gaussian head, matching the old
    policy-head structure more closely.
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
            # inverse softplus
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
    """TTT episode encoder + optional ECET-style aggregation/head structure.

    Important ablation knobs:
      policy_hidden_sizes=(): linear policy mean head like the old code.
      value_hidden_sizes=(): linear value head like the old code.
      aggregator_type="concat": current ECET-style slot-specific linear aggregator.
      aggregator_type="mean": average previous episode finals + current TTT output.
      use_state_proj=False: head sees only TTT/aggregated hidden, like old code.
      init_type="ppo": PPO-style orthogonal head initialization.
      init_type="xavier": current Xavier MLP initialization.
    """

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
        aggregator_type: str = "concat",
        use_state_proj: bool = True,
        init_type: str = "xavier",
        context_seq_len: int = 0,
        prev_context_window_mode: str = "last",
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
        self.use_state_proj = use_state_proj
        self.init_type = init_type
        self.context_seq_len = int(context_seq_len or 0)
        self.prev_context_window_mode = str(prev_context_window_mode or "last")

        if self.context_seq_len < 0:
            raise ValueError("context_seq_len must be >= 0. Use 0 for full episode context.")

        if self.prev_context_window_mode not in ["last", "random"]:
            raise ValueError("prev_context_window_mode must be either 'last' or 'random'")

        if self.aggregator_type not in ["concat", "mean"]:
            raise ValueError("aggregator_type must be either 'concat' or 'mean'")
        if self.init_type not in ["xavier", "ppo"]:
            raise ValueError("init_type must be either 'xavier' or 'ppo'")

        self.input_encoder = nn.Linear(input_dim, self.hidden_size)

        # Current ECET-style aggregator: flatten episode slots and learn a
        # slot-specific linear map. Not used when aggregator_type='mean'.
        if self.aggregator_type == "concat":
            self.episode_aggregator = nn.Linear(num_episodes * self.hidden_size, self.hidden_size)
        else:
            self.episode_aggregator = None

        # Optional ECET-style current-state projection phi_2(s).
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

        # TTTModel handles its own init. For ablation, choose only the non-TTT
        # heads/projections init style here.
        if self.init_type == "ppo":
            self._init_ppo_weights()

    def _init_ppo_weights(self):
        """PPO-style orthogonal initialization for non-TTT components."""
        # The old code did not explicitly orthogonal-init ObservationEncoder,
        # but this flag intentionally tests the PPO-style init variant for the
        # full non-TTT policy stack.
        self.input_encoder.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2.0)))

        if self.state_proj is not None:
            self.state_proj.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2.0)))

        if self.episode_aggregator is not None:
            self.episode_aggregator.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2.0)))

        if self.continuous:
            self.policy_head.mean_network.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2.0)))
            init_last_linear(self.policy_head.mean_network, gain=0.01)
        else:
            self.policy_head.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2.0)))
            init_last_linear(self.policy_head, gain=0.01)

        self.value_head.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2.0)))
        init_last_linear(self.value_head, gain=1.0)

    def _aggregator_weight_by_slot(self):
        if self.episode_aggregator is None:
            raise RuntimeError("_aggregator_weight_by_slot called when aggregator_type != 'concat'")
        return self.episode_aggregator.weight.view(self.hidden_size, self.num_episodes, self.hidden_size)

    def aggregate_full_trial_concat(self, episode_hidden: torch.Tensor) -> torch.Tensor:
        """Current ECET-style linear aggregator.

        episode_hidden: (B, E, T, H)
        returns: (B, E, T, H)
        """
        B, E, T, H = episode_hidden.shape
        if E != self.num_episodes:
            raise ValueError(f"Expected E={self.num_episodes}, got {E}")

        W = self._aggregator_weight_by_slot()  # (H_out, E, H_in)
        bias = self.episode_aggregator.bias
        outputs = []

        prev_contrib = episode_hidden.new_zeros(B, H)
        final_embeddings = episode_hidden[:, :, -1, :]  # (B, E, H)

        for ep_idx in range(E):
            if ep_idx > 0:
                prev_final = final_embeddings[:, ep_idx - 1, :]
                prev_contrib = prev_contrib + torch.einsum(
                    "bh,oh->bo", prev_final, W[:, ep_idx - 1, :]
                )

            current_seq = episode_hidden[:, ep_idx, :, :]
            current_contrib = torch.einsum("bth,oh->bto", current_seq, W[:, ep_idx, :])
            z = current_contrib + prev_contrib[:, None, :] + bias.view(1, 1, H)
            outputs.append(z)

        return torch.stack(outputs, dim=1)

    def aggregate_full_trial_mean(self, episode_hidden: torch.Tensor) -> torch.Tensor:
        """Average previous episode finals plus current TTT output.

        With num_episodes=1, this is exactly episode_hidden, so it removes the
        extra aggregator transform and is closest to the old code.
        """
        B, E, T, H = episode_hidden.shape
        outputs = []
        prev_sum = episode_hidden.new_zeros(B, H)
        final_embeddings = episode_hidden[:, :, -1, :]

        for ep_idx in range(E):
            current_seq = episode_hidden[:, ep_idx, :, :]
            z = (prev_sum[:, None, :] + current_seq) / float(ep_idx + 1)
            outputs.append(z)
            prev_sum = prev_sum + final_embeddings[:, ep_idx, :]

        return torch.stack(outputs, dim=1)

    def aggregate_full_trial(self, episode_hidden: torch.Tensor) -> torch.Tensor:
        if self.aggregator_type == "concat":
            return self.aggregate_full_trial_concat(episode_hidden)
        return self.aggregate_full_trial_mean(episode_hidden)

    def aggregate_step_concat(
        self,
        episode_memory: torch.Tensor,
        current_hidden: torch.Tensor,
        episode_idx: int,
    ) -> torch.Tensor:
        W = self._aggregator_weight_by_slot()
        H = self.hidden_size
        z = self.episode_aggregator.bias.view(1, H).expand(current_hidden.shape[0], H)

        if episode_idx > 0:
            prev = episode_memory[:, :episode_idx, :]
            W_prev = W[:, :episode_idx, :]
            z = z + torch.einsum("beh,oeh->bo", prev, W_prev)

        z = z + torch.einsum("bh,oh->bo", current_hidden, W[:, episode_idx, :])
        return z

    def aggregate_step_mean(
        self,
        episode_memory: torch.Tensor,
        current_hidden: torch.Tensor,
        episode_idx: int,
    ) -> torch.Tensor:
        if episode_idx > 0:
            prev_sum = episode_memory[:, :episode_idx, :].sum(dim=1)
            return (prev_sum + current_hidden) / float(episode_idx + 1)
        return current_hidden

    def aggregate_step(
        self,
        episode_memory: torch.Tensor,
        current_hidden: torch.Tensor,
        episode_idx: int,
    ) -> torch.Tensor:
        if self.aggregator_type == "concat":
            return self.aggregate_step_concat(episode_memory, current_hidden, episode_idx)
        return self.aggregate_step_mean(episode_memory, current_hidden, episode_idx)

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
        """Full-trial forward used for PPO update.

        agent_inputs: (B, E, T, input_dim)
        current_obs:  (B, E, T, obs_dim)
        """
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


    def _window_start(self, end_step: int) -> int:
        """Start index for a context window ending at end_step inclusive."""
        if self.context_seq_len <= 0:
            return 0
        return max(0, end_step + 1 - self.context_seq_len)

    def _episode_window(
        self,
        episode_inputs: torch.Tensor,
        end_step: Optional[int] = None,
        random_window: bool = False,
    ) -> torch.Tensor:
        """Slice an episode tensor to a context window.

        episode_inputs: (B, T, D).

        If context_seq_len=0, returns the full prefix through end_step.

        If context_seq_len>0 and random_window=False, returns the last
        context_seq_len tokens ending at end_step. This is used for the
        current episode because the context must lead to the current state.

        If context_seq_len>0 and random_window=True, samples one contiguous
        window of length context_seq_len from the available episode/prefix.
        This is intended only for previous completed episodes during PPO.
        """
        if episode_inputs.ndim != 3:
            raise ValueError(f"episode_inputs must be (B,T,D), got {episode_inputs.shape}")
        T = episode_inputs.shape[1]
        if end_step is None:
            end_step = T - 1
        if end_step < 0 or end_step >= T:
            raise ValueError(f"end_step={end_step} is out of range for T={T}")

        if self.context_seq_len <= 0:
            return episode_inputs[:, : end_step + 1, :]

        available = end_step + 1
        window_len = min(self.context_seq_len, available)

        if random_window and available > window_len:
            max_start = available - window_len
            start = int(torch.randint(0, max_start + 1, (1,), device=episode_inputs.device).item())
        else:
            start = available - window_len

        return episode_inputs[:, start : start + window_len, :]

    def _prev_episode_window(self, episode_inputs: torch.Tensor) -> torch.Tensor:
        """Context window for a previous completed episode during PPO.

        With --prev_context_window_mode last, this is the final window.
        With --prev_context_window_mode random, this is a random contiguous
        window. The current episode never uses this helper; its window always
        ends at the current state.
        """
        return self._episode_window(
            episode_inputs,
            end_step=None,
            random_window=(self.prev_context_window_mode == "random"),
        )

    def _encode_episode_final(self, episode_inputs: torch.Tensor) -> torch.Tensor:
        """Encode one episode/window and return the final hidden state."""
        x = self.input_encoder(episode_inputs)
        h = self.model(inputs_embeds=x, use_cache=False, return_dict=True).last_hidden_state
        return h[:, -1, :]

    @torch.no_grad()
    def encode_episode_finals_detached(self, agent_inputs: torch.Tensor) -> torch.Tensor:
        """Encode each full episode once and return detached final embeddings.

        Args:
            agent_inputs: (B, E, T, input_dim)

        Returns:
            finals: (B, E, H), where finals[:, e] is the final TTT hidden
            state of episode e. This is intended as stop-gradient context for
            efficient sequential PPO; rollout behavior is unchanged.
        """
        if agent_inputs.ndim != 4:
            raise ValueError(f"agent_inputs must be (B,E,T,D), got {agent_inputs.shape}")
        B, E, T, D = agent_inputs.shape
        H = self.hidden_size

        # If context_seq_len > 0, each completed episode is represented by
        # one context window. For previous-episode context, that window is
        # either the final window or a random contiguous window depending on
        # prev_context_window_mode. If context_seq_len == 0, this is the old
        # full-episode encoding.
        if self.context_seq_len <= 0 or self.prev_context_window_mode == "last":
            start = 0 if self.context_seq_len <= 0 else max(0, T - self.context_seq_len)
            window = agent_inputs[:, :, start:T, :]
            W = window.shape[2]
            x = self.input_encoder(window.reshape(B * E, W, D))
            h = self.model(inputs_embeds=x, use_cache=False, return_dict=True).last_hidden_state
            finals = h[:, -1, :].reshape(B, E, H)
            return finals.detach()

        # Random-window mode cannot be flattened with a shared start because
        # each episode slot may sample a different start. Encode slot by slot.
        finals = []
        for ep_idx in range(E):
            window = self._prev_episode_window(agent_inputs[:, ep_idx, :, :])
            finals.append(self._encode_episode_final(window))
        return torch.stack(finals, dim=1).detach()

    def forward_current_prefix_with_context(
        self,
        agent_inputs: torch.Tensor,
        current_obs: torch.Tensor,
        episode_idx: int,
        last_step: int,
        context_finals: torch.Tensor,
        context_episode_indices=None,
        return_dict: bool = True,
    ):
        """Forward only current episode prefix using detached previous context.

        This is the fast stop-gradient context path for sequential PPO. It is
        only defined for aggregator_type='mean'. Previous episodes are supplied
        as already-computed detached final embeddings; gradients flow through
        the current episode prefix and heads, but not through previous episodes.

        Returns policy/value for current episode positions 0..last_step only.
        """
        if self.aggregator_type != "mean":
            raise ValueError("Detached context reuse is only valid with aggregator_type='mean'")
        if agent_inputs.ndim != 4:
            raise ValueError(f"agent_inputs must be (B,E,T,D), got {agent_inputs.shape}")
        B, E, T, D = agent_inputs.shape
        if episode_idx < 0 or episode_idx >= E:
            raise ValueError(f"episode_idx={episode_idx} is out of range for E={E}")
        if last_step < 0 or last_step >= T:
            raise ValueError(f"last_step={last_step} is out of range for T={T}")
        if context_finals.shape[:2] != (B, E):
            raise ValueError(
                f"context_finals must have shape (B,E,H) matching ({B},{E},H), "
                f"got {context_finals.shape}"
            )

        prefix_len = last_step + 1
        window_start = self._window_start(last_step)
        cur_inputs = agent_inputs[:, episode_idx, window_start:prefix_len, :]
        x_cur = self.input_encoder(cur_inputs)
        h_cur = self.model(inputs_embeds=x_cur, use_cache=False, return_dict=True).last_hidden_state

        if episode_idx <= 0:
            z_task = h_cur
        else:
            if context_episode_indices is None:
                prev_sum = context_finals[:, :episode_idx, :].sum(dim=1)
                z_task = (prev_sum[:, None, :] + h_cur) / float(episode_idx + 1)
            else:
                used = len(context_episode_indices)
                if used == 0:
                    z_task = h_cur
                else:
                    for ep in context_episode_indices:
                        if ep < 0 or ep >= episode_idx:
                            raise ValueError(f"context episode {ep} must be in [0, {episode_idx})")
                    prev_sum = context_finals[:, context_episode_indices, :].sum(dim=1)
                    prev_mean = prev_sum / float(used)
                    z_task = (float(episode_idx) * prev_mean[:, None, :] + h_cur) / float(episode_idx + 1)

        obs_prefix = current_obs[:, episode_idx, window_start:prefix_len, :]
        policy_out, value = self._heads(z_task, obs_prefix)
        if not return_dict:
            return policy_out, value, h_cur
        return RLModelOutput(
            policy=policy_out,
            value=value,
            hidden_states=h_cur,
            cache_params=None,
            time_offset=window_start,
        )


    def forward_prefix_flat(
        self,
        agent_inputs: torch.Tensor,
        current_obs: torch.Tensor,
        last_episode: int,
        last_step: int,
        context_episode_indices=None,
        return_dict: bool = True,
    ):
        """Forward only the prefix needed for a sequential PPO chunk.

        Args:
            agent_inputs: (B, E, T, input_dim)
            current_obs:  (B, E, T, obs_dim)
            last_episode: current episode index whose prefix is included
            last_step: inclusive last step index inside last_episode
            context_episode_indices: optional list of previous episode indices.
                This is only supported for aggregator_type='mean'. When provided,
                the output contains only the current episode prefix, conditioned on
                the sampled previous episode finals plus the current TTT output.

        Returns:
            RLModelOutput where policy/value are flattened over the computed time
            dimension: (B, L, ...). If context_episode_indices is None, L is all
            transitions from episode 0 through (last_episode, last_step). If it is
            provided, L is last_step + 1 for the current episode only.
        """
        if agent_inputs.ndim != 4:
            raise ValueError(f"agent_inputs must be (B,E,T,D), got {agent_inputs.shape}")

        B, E, T, D = agent_inputs.shape
        if last_episode < 0 or last_episode >= E:
            raise ValueError(f"last_episode={last_episode} is out of range for E={E}")
        if last_step < 0 or last_step >= T:
            raise ValueError(f"last_step={last_step} is out of range for T={T}")

        prefix_len = last_step + 1

        # Optional sampled context mode. This is meant for large trial_length and
        # aggregator_type='mean'. It avoids forwarding all previous episodes.
        if context_episode_indices is not None:
            if self.aggregator_type != "mean":
                raise ValueError("context_episode_indices is only valid with aggregator_type='mean'")

            prev_sum = agent_inputs.new_zeros(B, self.hidden_size)
            used = 0
            for ep_idx in context_episode_indices:
                if ep_idx < 0 or ep_idx >= last_episode:
                    raise ValueError(
                        f"context episode {ep_idx} must be in [0, {last_episode})"
                    )
                # Previous episodes are represented by a short window ending at
                # the episode's final stored timestep when context_seq_len > 0.
                prev_window = self._prev_episode_window(agent_inputs[:, ep_idx, :, :])
                h_prev_final = self._encode_episode_final(prev_window)
                prev_sum = prev_sum + h_prev_final
                used += 1

            window_start = self._window_start(last_step)
            x_cur = self.input_encoder(agent_inputs[:, last_episode, window_start:prefix_len, :])
            h_cur = self.model(inputs_embeds=x_cur, use_cache=False, return_dict=True).last_hidden_state

            # 1. Unbiased full mean approximation of previous episodes
            if used > 0:
                prev_mean = prev_sum / float(used)
                z_task = (float(last_episode) * prev_mean[:, None, :] + h_cur) / float(last_episode + 1)
            else:
                z_task = h_cur

            # 2. Normal mean
            # z_task = (prev_sum[:, None, :] + h_cur) / float(used + 1)
            
            obs_prefix = current_obs[:, last_episode, window_start:prefix_len, :]
            policy_out, value = self._heads(z_task, obs_prefix)
            if not return_dict:
                return policy_out, value, h_cur
            return RLModelOutput(
                policy=policy_out,
                value=value,
                hidden_states=h_cur,
                cache_params=None,
                time_offset=window_start,
            )

        # Windowed exact mode: previous episodes are summarized by a short
        # final window, and the current episode is represented by the short
        # window ending at last_step. This is the ECET-style low-context path.
        # It returns only current-episode window outputs, with time_offset
        # indicating where those outputs start in the original episode.
        if self.context_seq_len > 0:
            window_start = self._window_start(last_step)
            h_cur = self.model(
                inputs_embeds=self.input_encoder(agent_inputs[:, last_episode, window_start:prefix_len, :]),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state

            if self.aggregator_type == "mean":
                if last_episode > 0:
                    prev_sum = agent_inputs.new_zeros(B, self.hidden_size)
                    for ep_idx in range(last_episode):
                        prev_window = self._prev_episode_window(agent_inputs[:, ep_idx, :, :])
                        prev_sum = prev_sum + self._encode_episode_final(prev_window)
                    z_task = (prev_sum[:, None, :] + h_cur) / float(last_episode + 1)
                else:
                    z_task = h_cur
            else:
                W = self._aggregator_weight_by_slot()
                bias = self.episode_aggregator.bias
                prev_contrib = agent_inputs.new_zeros(B, self.hidden_size)
                for ep_idx in range(last_episode):
                    prev_window = self._prev_episode_window(agent_inputs[:, ep_idx, :, :])
                    h_prev_final = self._encode_episode_final(prev_window)
                    prev_contrib = prev_contrib + torch.einsum(
                        "bh,oh->bo", h_prev_final, W[:, ep_idx, :]
                    )
                current_contrib = torch.einsum("bth,oh->bto", h_cur, W[:, last_episode, :])
                z_task = current_contrib + prev_contrib[:, None, :] + bias.view(1, 1, self.hidden_size)

            obs_prefix = current_obs[:, last_episode, window_start:prefix_len, :]
            policy_out, value = self._heads(z_task, obs_prefix)
            if not return_dict:
                return policy_out, value, h_cur
            return RLModelOutput(
                policy=policy_out,
                value=value,
                hidden_states=h_cur,
                cache_params=None,
                time_offset=window_start,
            )

        # Exact prefix mode: forward completed previous episodes fully and the
        # current episode only up to last_step. No future tokens are processed.
        h_list = []
        obs_list = []
        for ep_idx in range(last_episode + 1):
            length = T if ep_idx < last_episode else prefix_len
            x_ep = self.input_encoder(agent_inputs[:, ep_idx, :length, :])
            h_ep = self.model(inputs_embeds=x_ep, use_cache=False, return_dict=True).last_hidden_state
            h_list.append(h_ep)
            obs_list.append(current_obs[:, ep_idx, :length, :])

        if self.aggregator_type == "mean":
            z_list = []
            prev_sum = agent_inputs.new_zeros(B, self.hidden_size)
            for ep_idx, h_ep in enumerate(h_list):
                z_ep = (prev_sum[:, None, :] + h_ep) / float(ep_idx + 1)
                z_list.append(z_ep)
                prev_sum = prev_sum + h_ep[:, -1, :]
        else:
            W = self._aggregator_weight_by_slot()
            bias = self.episode_aggregator.bias
            z_list = []
            prev_contrib = agent_inputs.new_zeros(B, self.hidden_size)
            for ep_idx, h_ep in enumerate(h_list):
                current_contrib = torch.einsum("bth,oh->bto", h_ep, W[:, ep_idx, :])
                z_ep = current_contrib + prev_contrib[:, None, :] + bias.view(1, 1, self.hidden_size)
                z_list.append(z_ep)
                prev_contrib = prev_contrib + torch.einsum(
                    "bh,oh->bo", h_ep[:, -1, :], W[:, ep_idx, :]
                )

        z_task = torch.cat(z_list, dim=1)
        obs_prefix = torch.cat(obs_list, dim=1)
        hidden_prefix = torch.cat(h_list, dim=1)
        policy_out, value = self._heads(z_task, obs_prefix)
        if not return_dict:
            return policy_out, value, hidden_prefix
        return RLModelOutput(policy=policy_out, value=value, hidden_states=hidden_prefix, cache_params=None)

    @torch.no_grad()
    def init_episode_memory(self, batch_size: int, device=None, num_episodes: Optional[int] = None):
        """Allocate episode memory.

        num_episodes can be larger than the training trial length for evaluation
        when aggregator_type='mean'. The slot-specific concat aggregator is tied
        to self.num_episodes and cannot extrapolate to extra slots.
        """
        device = device or next(self.parameters()).device
        if num_episodes is None:
            num_episodes = self.num_episodes
        if self.aggregator_type == "concat" and num_episodes != self.num_episodes:
            raise ValueError(
                "num_episodes different from training self.num_episodes is only supported "
                "with aggregator_type='mean'."
            )
        return torch.zeros(batch_size, num_episodes, self.hidden_size, device=device)

    def act_step(
        self,
        agent_input: torch.Tensor,
        current_obs: torch.Tensor,
        episode_memory: torch.Tensor,
        episode_idx: int,
        cache_params: Optional[TTTCache] = None,
        context_window_inputs: Optional[torch.Tensor] = None,
    ):
        """One rollout/eval step.

        Default behavior is the old cached one-token update over the whole
        current episode. If context_window_inputs is provided, it should be
        (B, W, input_dim) and the current hidden state is computed from that
        window ending at the current state. In that windowed mode, cache is not
        used because the context is explicitly the short window.
        """
        if context_window_inputs is not None:
            x = self.input_encoder(context_window_inputs)
            base_outputs = self.model(
                inputs_embeds=x,
                cache_params=None,
                use_cache=False,
                return_dict=True,
            )
        else:
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
            cache_params=None if context_window_inputs is not None else base_outputs.cache_params,
            hidden_states=current_hidden,
        )
