#!/usr/bin/env python3
"""
Plot per-task PPO learning-signal diagnostics.

Expected input:
    rollout_task_learning_signal.csv

Usage:
    python plot_task_learning_signal.py \
        --csv runs/my_run/rollout_task_learning_signal.csv \
        --out_dir runs/my_run/task_signal_plots

Optional:
    python plot_task_learning_signal.py \
        --csv runs/my_run/rollout_task_learning_signal.csv \
        --out_dir runs/my_run/task_signal_plots \
        --smooth 5 \
        --last_n 30
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_METRICS = [
    "trial_return",
    "final_return",
    "final_success",
    "anysuccess",
    "mean_ep_success",
    "adv_mean",
    "adv_std",
    "adv_abs_mean",
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


def safe_filename(name: str) -> str:
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def maybe_smooth(df: pd.DataFrame, metric: str, smooth: int) -> pd.DataFrame:
    if smooth <= 1:
        df[f"{metric}_plot"] = df[metric]
        return df

    parts = []
    for env_name, g in df.groupby("env_name", sort=False):
        g = g.sort_values("update").copy()
        g[f"{metric}_plot"] = g[metric].rolling(window=smooth, min_periods=1).mean()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def plot_metric_timeseries(df: pd.DataFrame, metric: str, out_dir: Path, smooth: int):
    if metric not in df.columns:
        return

    d = df[["update", "env_name", metric]].dropna().copy()
    if d.empty:
        return

    d = maybe_smooth(d, metric, smooth)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    for env_name, g in d.groupby("env_name", sort=True):
        g = g.sort_values("update")
        ax.plot(g["update"], g[f"{metric}_plot"], label=env_name, linewidth=1.8)

    ax.set_title(f"{metric} by task class" + (f" | smoothed={smooth}" if smooth > 1 else ""))
    ax.set_xlabel("PPO update")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"timeseries_{safe_filename(metric)}.png", dpi=160)
    plt.close(fig)


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    p = df.pivot_table(index="env_name", columns="update", values=metric, aggfunc="mean")
    return p.sort_index()


def plot_metric_heatmap(df: pd.DataFrame, metric: str, out_dir: Path):
    if metric not in df.columns:
        return

    p = pivot_metric(df, metric)
    if p.empty:
        return

    fig = plt.figure(figsize=(14, max(4.5, 0.45 * len(p.index))))
    ax = fig.add_subplot(111)

    data = p.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", interpolation="nearest")

    ax.set_title(f"{metric} heatmap by task class")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Task class")

    updates = p.columns.to_numpy()
    if len(updates) > 0:
        tick_count = min(12, len(updates))
        tick_idx = np.linspace(0, len(updates) - 1, tick_count).astype(int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([str(int(updates[i])) for i in tick_idx], rotation=45, ha="right")

    ax.set_yticks(np.arange(len(p.index)))
    ax.set_yticklabels(p.index)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    fig.tight_layout()
    fig.savefig(out_dir / f"heatmap_{safe_filename(metric)}.png", dpi=160)
    plt.close(fig)


def plot_metric_final_bar(df: pd.DataFrame, metric: str, out_dir: Path, last_n: int):
    if metric not in df.columns:
        return

    d = df.dropna(subset=[metric]).copy()
    if d.empty:
        return

    max_update = int(d["update"].max())
    start_update = max_update - last_n + 1 if last_n > 0 else int(d["update"].min())
    d = d[d["update"] >= start_update]

    summary = d.groupby("env_name", sort=True)[metric].mean().sort_values(ascending=False)
    if summary.empty:
        return

    fig = plt.figure(figsize=(11, max(4.5, 0.45 * len(summary))))
    ax = fig.add_subplot(111)

    y = np.arange(len(summary.index))
    ax.barh(y, summary.to_numpy())
    ax.set_yticks(y)
    ax.set_yticklabels(summary.index)
    ax.invert_yaxis()
    ax.set_xlabel(f"mean {metric}")
    ax.set_title(f"Mean {metric} over last {last_n} updates")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"last{last_n}_bar_{safe_filename(metric)}.png", dpi=160)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, x_metric: str, y_metric: str, out_dir: Path, last_n: int):
    if x_metric not in df.columns or y_metric not in df.columns:
        return

    d = df.dropna(subset=[x_metric, y_metric]).copy()
    if d.empty:
        return

    max_update = int(d["update"].max())
    start_update = max_update - last_n + 1 if last_n > 0 else int(d["update"].min())
    d = d[d["update"] >= start_update]

    summary = d.groupby("env_name", sort=True)[[x_metric, y_metric]].mean().dropna()
    if summary.empty:
        return

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)

    ax.scatter(summary[x_metric], summary[y_metric], s=70)

    for env_name, row in summary.iterrows():
        ax.annotate(
            env_name,
            (row[x_metric], row[y_metric]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(f"mean {x_metric} over last {last_n}")
    ax.set_ylabel(f"mean {y_metric} over last {last_n}")
    ax.set_title(f"{y_metric} vs {x_metric} by task class")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"scatter_{safe_filename(y_metric)}_vs_{safe_filename(x_metric)}_last{last_n}.png", dpi=160)
    plt.close(fig)


def write_last_n_summary(df: pd.DataFrame, metrics, out_dir: Path, last_n: int):
    max_update = int(df["update"].max())
    start_update = max_update - last_n + 1 if last_n > 0 else int(df["update"].min())
    d = df[df["update"] >= start_update].copy()

    usable = [m for m in metrics if m in d.columns]
    summary = d.groupby(["env_name"], sort=True)[usable].mean().reset_index()
    summary.insert(1, "start_update", start_update)
    summary.insert(2, "end_update", max_update)

    summary_path = out_dir / f"summary_last{last_n}.csv"
    summary.to_csv(summary_path, index=False)

    rank_metrics = [
        m for m in [
            "final_success",
            "anysuccess",
            "adv_abs_mean",
            "adv_std",
            "value_mse",
            "value_abs_error",
            "explained_variance",
            "return_std",
        ]
        if m in summary.columns
    ]

    if rank_metrics:
        rank_path = out_dir / f"rank_summary_last{last_n}.txt"
        with open(rank_path, "w") as f:
            for m in rank_metrics:
                ascending = m not in ["final_success", "anysuccess", "explained_variance"]
                ranked = summary[["env_name", m]].sort_values(m, ascending=ascending)
                f.write(f"\n=== {m} {'LOW to HIGH' if ascending else 'HIGH to LOW'} ===\n")
                f.write(ranked.to_string(index=False))
                f.write("\n")


def plot_compact_dashboard(df: pd.DataFrame, out_dir: Path, last_n: int):
    metrics = [
        "final_success",
        "anysuccess",
        "adv_abs_mean",
        "adv_std",
        "value_mse",
        "explained_variance",
    ]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return

    max_update = int(df["update"].max())
    start_update = max_update - last_n + 1 if last_n > 0 else int(df["update"].min())
    d = df[df["update"] >= start_update].copy()

    summary = d.groupby("env_name", sort=True)[metrics].mean()
    if summary.empty:
        return

    for metric in metrics:
        vals = summary[metric].sort_values(ascending=False)
        fig = plt.figure(figsize=(11, max(4.5, 0.45 * len(vals))))
        ax = fig.add_subplot(111)
        y = np.arange(len(vals.index))
        ax.barh(y, vals.to_numpy())
        ax.set_yticks(y)
        ax.set_yticklabels(vals.index)
        ax.invert_yaxis()
        ax.set_title(f"Dashboard: {metric} by task | last {last_n} updates")
        ax.set_xlabel(metric)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"dashboard_{safe_filename(metric)}_last{last_n}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default='csv_outputs/nostate_ML10_smallAC_run001/rollout_task_learning_signal.csv', help="Path to rollout_task_learning_signal.csv")
    parser.add_argument("--out_dir", default=None, help="Directory to write plots")
    parser.add_argument("--smooth", type=int, default=1, help="Rolling mean over updates for time series")
    parser.add_argument("--last_n", type=int, default=20, help="Number of final updates for summary bars/scatters")
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metrics to plot. Default plots all known metrics present in the CSV.",
    )
    parser.add_argument("--skip_heatmaps", action="store_true", help="Skip heatmaps.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "task_learning_signal_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = {"update", "env_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    for col in df.columns:
        if col != "env_name":
            df[col] = pd.to_numeric(df[col], errors="ignore")

    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    else:
        metrics = [m for m in DEFAULT_METRICS if m in df.columns]

    if not metrics:
        raise ValueError("No plottable metrics found. Check --metrics or CSV columns.")

    print(f"Loaded {csv_path}")
    print(f"Rows: {len(df)} | tasks: {df['env_name'].nunique()} | updates: {df['update'].nunique()}")
    print(f"Writing plots to: {out_dir}")
    print(f"Metrics: {', '.join(metrics)}")

    for metric in metrics:
        plot_metric_timeseries(df, metric, out_dir, smooth=args.smooth)
        plot_metric_final_bar(df, metric, out_dir, last_n=args.last_n)
        if not args.skip_heatmaps:
            plot_metric_heatmap(df, metric, out_dir)

    scatter_pairs = [
        ("adv_abs_mean", "final_success"),
        ("adv_std", "final_success"),
        ("value_mse", "final_success"),
        ("value_abs_error", "final_success"),
        ("explained_variance", "final_success"),
        ("return_std", "final_success"),
        ("adv_abs_mean", "value_mse"),
        ("explained_variance", "value_mse"),
        ("anysuccess", "final_success"),
    ]
    for x_metric, y_metric in scatter_pairs:
        plot_scatter(df, x_metric, y_metric, out_dir, last_n=args.last_n)

    write_last_n_summary(df, metrics, out_dir, last_n=args.last_n)
    plot_compact_dashboard(df, out_dir, last_n=args.last_n)

    print("Done.")
    print(f"Summary CSV: {out_dir / f'summary_last{args.last_n}.csv'}")
    print(f"Rank summary: {out_dir / f'rank_summary_last{args.last_n}.txt'}")


if __name__ == "__main__":
    main()
