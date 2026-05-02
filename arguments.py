import argparse


def parse_hidden_sizes(value):
    """Parse hidden-size CLI strings.

    Examples:
      --policy_hidden_sizes 64,64  -> (64, 64)
      --policy_hidden_sizes 64     -> (64,)
      --policy_hidden_sizes 0      -> ()   # linear head
      --policy_hidden_sizes none   -> ()   # linear head
    """
    if value is None:
        return ()
    value = str(value).strip().lower()
    if value in ["", "0", "none", "linear", "[]", "()"]:
        return ()
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def parse_int_list(value):
    """Parse comma-separated integer lists, e.g. 5,25."""
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in ["", "none", "0"]:
        return None
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def get_args():
    parser = argparse.ArgumentParser(description="TTT-ECET-style PPO Training")

    parser.add_argument("--task_set", type=str, default="ML1", choices=["ML1", "ML10", "ML45"])
    parser.add_argument("--env_name", type=str, default="push-v3")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=50, help="Number of parallel environments to run during training.")
    parser.add_argument("--trial_length", type=int, default=1, help="Episodes per trial and episode-memory slots.")
    parser.add_argument("--rollout_steps", type=int, default=500, help="Steps per episode.")

    parser.add_argument("--agent_mode", type=str, default="agent_rl2", choices=["agent_v1", "agent_v2", "agent_rl2"])
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--ttt_layer_type", type=str, default="mlp", choices=["linear", "mlp"])

    parser.add_argument(
        "--context_seq_len",
        type=int,
        default=0,
        help=(
            "TTT context window length per episode. Use 0 to encode the whole episode/prefix. "
            "If >0, previous episodes are represented by a context_seq_len transition window, "
            "and rollout/eval/current-episode PPO uses the window ending at the current state."
        ),
    )
    parser.add_argument(
        "--prev_context_window_mode",
        type=str,
        default="last",
        choices=["last", "random"],
        help=(
            "How to choose the context_seq_len window for previous completed episodes during PPO. "
            "last = use the final window of each previous episode; random = sample a random "
            "contiguous window from each previous episode. Current episode windows always end at "
            "the current state. Rollout/eval previous-episode memory remains the observed final "
            "episode embedding."
        ),
    )
    parser.add_argument(
        "--context_episode_sample_mode",
        type=str,
        default="uniform",
        choices=["uniform", "recent", "last"],
        help=(
            "How to choose previous episodes when --ppo_context_episode_sample > 0. "
            "uniform = random previous episodes; recent = sample previous episodes with "
            "higher probability for newer episodes; last = use the most recent K previous episodes. "
            "Sampled episode indices are sorted chronologically before aggregation."
        ),
    )

    # Ablation knobs for the ECET-style additions.
    parser.add_argument(
        "--policy_hidden_sizes",
        type=parse_hidden_sizes,
        default=(),
        help="Comma-separated actor hidden sizes. Use 0/none/linear for a linear policy mean head.",
    )
    parser.add_argument(
        "--value_hidden_sizes",
        type=parse_hidden_sizes,
        default=(),
        help="Comma-separated critic hidden sizes. Use 0/none/linear for a linear value head.",
    )
    parser.add_argument(
        "--aggregator_type",
        type=str,
        default="mean",
        choices=["concat", "mean", "ema"],
        help=(
            "concat = current slot-specific linear aggregator; "
            "mean = average previous episode finals plus current TTT output; "
            "ema = recency-weighted exponential moving average over previous episode finals."
        ),
    )
    parser.add_argument(
        "--ema_beta",
        type=float,
        default=0.7,
        help=(
            "EMA decay for --aggregator_type ema. Higher values keep longer memory; "
            "lower values weight newer episodes/current hidden more strongly."
        ),
    )
    parser.add_argument(
        "--no_state_proj",
        action="store_false",
        dest="use_state_proj",
        help="Disable ECET-style current-state projection concatenated into the policy/value heads.",
    )
    parser.set_defaults(use_state_proj=True)
    parser.add_argument(
        "--init_type",
        type=str,
        default="ppo",
        choices=["xavier", "ppo"],
        help="xavier = current MLP Xavier init; ppo = PPO-style orthogonal init for non-TTT policy stack.",
    )

    parser.add_argument("--num_updates", type=int, default=1000)
    parser.add_argument("--ppo_epochs", type=int, default=30)
    parser.add_argument("--ppo_minibatch_envs", type=int, default=30, help="Number of trials/envs per PPO minibatch.")
    parser.add_argument("--ppo_minibatch_steps", type=int, default=0, help=("Number of random transition positions per PPO optimizer step inside selected env trials. Use 0 to train on all selected trial steps at once."))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_std", type=float, default=0.1)
    parser.add_argument("--max_std", type=float, default=1.5)
    parser.add_argument("--init_std", type=float, default=0.5)
    parser.add_argument("--lr_decay", action="store_true")
    parser.add_argument("--lr_decay_start", type=float, default=0.0)
    parser.add_argument("--lr_end_factor", type=float, default=0.05)

    parser.add_argument("--eval_interval", type=int, default=5, help="Evaluate every N updates.")
    parser.add_argument("--eval_num_tasks", type=int, default=25, help="Sample this many eval variations, not all variations.")
    parser.add_argument("--eval_num_trials", type=int, default=1, help="Number of evaluation trials per vectorized eval worker.")
    parser.add_argument(
        "--eval_trial_length",
        type=int,
        default=None,
        help=(
            "Number of episodes to run per evaluation trial. If omitted, uses --trial_length. "
            "This can be larger than training trial_length when aggregator_type=mean or ema."
        ),
    )
    parser.add_argument(
        "--eval_report_lengths",
        type=parse_int_list,
        default=None,
        help=(
            "Comma-separated prefix lengths to summarize from one eval rollout, e.g. 5,25. "
            "Evaluation runs once for --eval_trial_length episodes, then reports metrics for these prefixes."
        ),
    )
    parser.add_argument("--comment", type=str, default="")

    # Run/output management.
    parser.add_argument("--run_root", type=str, default="csv_outputs", help="Root directory for numbered run folders.")
    parser.add_argument("--run_name", type=str, default="", help="Optional base name for this run. If empty, a name is built from key args.")

    # PPO update strategy.
    parser.add_argument(
        "--ppo_update_mode",
        type=str,
        default="random",
        choices=["random", "sequential"],
        help=(
            "random = old random transition chunking; sequential = chronological chunks, "
            "forwarding only the prefix needed for the current chunk."
        ),
    )
    parser.add_argument(
        "--ppo_sequential_loss_scope",
        type=str,
        default="chunk",
        choices=["chunk", "prefix"],
        help=(
            "For --ppo_update_mode sequential: chunk = loss only on the current ordered chunk; "
            "prefix = loss on all transitions from the start of the trial through the current chunk."
        ),
    )
    parser.add_argument(
        "--ppo_context_episode_sample",
        type=int,
        default=0,
        help=(
            "For sequential PPO with aggregator_type=mean/ema only: if >0, sample this many previous "
            "episodes as context plus the current episode, instead of forwarding all previous episodes. "
            "Use 0 to use all previous episodes."
        ),
    )

    parser.add_argument(
        "--detach_context_episodes",
        action="store_true",
        help=(
            "For sequential PPO with aggregator_type=mean/ema: encode previous episode final "
            "embeddings once with no_grad/detach for each env minibatch and reuse them "
            "across current-episode chunks. This saves PPO compute but uses stale, "
            "stop-gradient previous-episode context. Rollout is unchanged."
        ),
    )

    args = parser.parse_args()

    if args.eval_trial_length is None:
        args.eval_trial_length = args.trial_length

    if args.eval_report_lengths is None:
        args.eval_report_lengths = (args.eval_trial_length,)
    else:
        seen = set()
        cleaned = []
        for length in args.eval_report_lengths:
            if length <= 0:
                raise ValueError("All --eval_report_lengths must be positive.")
            if length not in seen:
                seen.add(length)
                cleaned.append(length)
        args.eval_report_lengths = tuple(cleaned)

    if max(args.eval_report_lengths) > args.eval_trial_length:
        raise ValueError(
            f"max(eval_report_lengths)={max(args.eval_report_lengths)} is larger than "
            f"eval_trial_length={args.eval_trial_length}. Increase --eval_trial_length."
        )

    if args.context_seq_len is None:
        args.context_seq_len = 0

    if not (0.0 <= args.ema_beta <= 1.0):
        raise ValueError("--ema_beta must be between 0 and 1 inclusive.")
    if args.context_seq_len < 0:
        raise ValueError("--context_seq_len must be >= 0. Use 0 for full-episode context.")

    return args
