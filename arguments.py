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
    parser = argparse.ArgumentParser(description="TTT-ECET PPO Training")

    parser.add_argument("--task_set", type=str, default="ML1", choices=["ML1", "ML10", "ML45"])
    parser.add_argument("--env_name", type=str, default="push-v3")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=50, help="Number of parallel environments during training.")
    parser.add_argument(
        "--num_env_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocess workers for train_multicore.py. Each worker owns a batch "
            "of envs. Use 0 for auto based on SLURM_CPUS_PER_TASK/os.cpu_count. "
            "Ignored by train.py."
        ),
    )
    parser.add_argument(
        "--eval_num_env_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocess workers for evaluation in train_multicore.py. Use 0 for auto. "
            "Set this lower than --eval_num_tasks to avoid launching too many processes."
        ),
    )
    parser.add_argument(
        "--env_start_method",
        type=str,
        default="fork",
        choices=["fork", "spawn", "forkserver"],
        help="Multiprocessing start method used by train_multicore.py.",
    )
    parser.add_argument("--trial_length", type=int, default=5, help="Episodes per trial and episode-memory slots.")
    parser.add_argument("--rollout_steps", type=int, default=500, help="Steps per episode.")
    parser.add_argument("--random_task_sample", action="store_true", help="Sample training task classes independently instead of balancing per update.")

    parser.add_argument("--agent_mode", type=str, default="agent_rl2", choices=["agent_v1", "agent_v2", "agent_rl2"])
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=1)
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--ttt_layer_type", type=str, default="mlp", choices=["linear", "mlp"])

    parser.add_argument(
        "--context_episode_sample_mode",
        type=str,
        default="uniform",
        choices=["uniform", "recent", "last"],
        help=(
            "How to choose previous episodes when --ppo_context_episode_sample > 0. "
            "uniform = random previous episodes; recent = higher probability for newer episodes; "
            "last = use the most recent K previous episodes. Sampled indices are sorted chronologically."
        ),
    )

    parser.add_argument(
        "--policy_hidden_sizes",
        type=parse_hidden_sizes,
        default=(64,64),
        help="Comma-separated actor hidden sizes. Use 0/none/linear for a linear policy mean head.",
    )
    parser.add_argument(
        "--value_hidden_sizes",
        type=parse_hidden_sizes,
        default=(64,64),
        help="Comma-separated critic hidden sizes. Use 0/none/linear for a linear value head.",
    )
    parser.add_argument(
        "--aggregator_type",
        type=str,
        default="attn",
        choices=["ema", "attn"],
        help=(
            "ema = recency-weighted EMA over previous episode finals plus current hidden; "
            "attn = current-hidden query attention over previous episode finals."
        ),
    )
    parser.add_argument(
        "--ema_beta",
        type=float,
        default=0.5,
        help="EMA decay for --aggregator_type ema. Higher values keep longer memory.",
    )
    parser.add_argument(
        "--episode_attn_heads",
        type=int,
        default=1,
        help="Number of attention heads for --aggregator_type attn.",
    )
    parser.add_argument(
        "--no_state_proj",
        action="store_false",
        dest="use_state_proj",
        help="Disable ECET-style current-state projection concatenated into the policy/value heads.",
    )
    parser.set_defaults(use_state_proj=True)

    parser.add_argument("--num_updates", type=int, default=400)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--ppo_minibatch_envs", type=int, default=10, help="Number of trials/envs per PPO minibatch.")
    parser.add_argument(
        "--ppo_minibatch_steps",
        type=int,
        default=0,
        help="Number of ordered timesteps per PPO optimizer step. Use 0 to train on the whole episode chunk.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument(
        "--normalize_advantage_by_class",
        action="store_true",
        help=(
            "Normalize PPO advantages separately within each sampled task class instead of using one global "
            "rollout mean/std. Labels are used only for loss scaling, not as policy inputs."
        ),
    )
    parser.add_argument(
        "--normalize_reward_by_class",
        action="store_true",
        help=(
            "Normalize rewards for PPO training targets with one running reward normalizer per task class. "
            "The policy/TTT prev_reward input still uses the raw environment reward."
        ),
    )

    parser.add_argument(
        "--log_rollout_learning_signal",
        action="store_true",
        help=(
            "Log per-task rollout learning-signal diagnostics to "
            "rollout_task_learning_signal.csv and print the compact signal table. "
            "Disabled by default because it adds CSV I/O and extra CPU work."
        ),
    )
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--log_ppo_class_diagnostics",
        action="store_true",
        help=(
            "Log per-task-class PPO loss and diagnostic gradient norms to "
            "ppo_task_class_diagnostics.csv. This is disabled by default because "
            "it adds extra autograd calls and CSV I/O."
        ),
    )
    parser.add_argument("--min_std", type=float, default=0.1)
    parser.add_argument("--max_std", type=float, default=1.5)
    parser.add_argument("--init_std", type=float, default=0.5)
    parser.add_argument(
        "--squash_actions",
        action="store_true",
        help=(
            "Use tanh-squashed Gaussian actions for PPO. The environment receives "
            "tanh(raw_action), while PPO stores raw pre-tanh actions and uses the "
            "correct tanh log-probability correction."
        ),
    )
    parser.add_argument(
        "--squash_logprob_eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon for tanh-squashed Gaussian log-prob correction.",
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied after tanh when --squash_actions is set: "
            "env_action = action_scale * tanh(raw_action). Use <= 1.0."
        ),
    )

    parser.add_argument("--eval_interval", type=int, default=5, help="Evaluate every N updates.")
    parser.add_argument("--eval_num_tasks", type=int, default=100, help="Sample this many eval variations.")
    parser.add_argument("--eval_num_trials", type=int, default=1, help="Number of evaluation trials per vectorized eval worker.")
    parser.add_argument(
        "--eval_trial_length",
        type=int,
        default=25,
        help="Number of episodes to run per evaluation trial. If omitted, uses --trial_length.",
    )
    parser.add_argument(
        "--eval_report_lengths",
        type=parse_int_list,
        default=(5,25),
        help="Comma-separated prefix lengths to summarize from one eval rollout, e.g. 5,25.",
    )
    parser.add_argument("--comment", type=str, default="")

    parser.add_argument("--run_root", type=str, default="csv_outputs", help="Root directory for numbered run folders.")
    parser.add_argument("--run_name", type=str, default="", help="Optional base name for this run.")

    parser.add_argument(
        "--ppo_context_episode_sample",
        type=int,
        default=0,
        help=(
            "If >0, sample this many previous episodes as context for a current chunk instead of "
            "forwarding all previous episodes. Use 0 to use all previous episodes."
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

    if not (0.0 <= args.ema_beta <= 1.0):
        raise ValueError("--ema_beta must be between 0 and 1 inclusive.")
    if args.episode_attn_heads <= 0:
        raise ValueError("--episode_attn_heads must be positive.")
    if args.ppo_context_episode_sample < 0:
        raise ValueError("--ppo_context_episode_sample must be non-negative.")

    return args
