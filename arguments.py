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
        choices=["concat", "mean"],
        help="concat = current slot-specific linear aggregator; mean = average previous episode finals plus current TTT output.",
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

    parser.add_argument("--eval_interval", type=int, default=10, help="Evaluate every N updates.")
    parser.add_argument("--eval_num_tasks", type=int, default=20, help="Sample this many eval variations, not all variations.")
    parser.add_argument("--eval_num_trials", type=int, default=1, help="Number of trials to evaluate per task.")
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
            "For sequential PPO with aggregator_type=mean only: if >0, sample this many previous "
            "episodes as context plus the current episode, instead of forwarding all previous episodes. "
            "Use 0 to use all previous episodes."
        ),
    )

    return parser.parse_args()
