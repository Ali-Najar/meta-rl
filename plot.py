from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler


ROOT = Path("csv_outputs")

# Set to 1 or None to disable moving average
MOVING_AVERAGE_WINDOW = 5

METRICS = [
    ("rollout_anysuccess", "Rollout Any Success"),
    ("rollout_trial_return", "Rollout Trial Return"),
    ("eval_final_return", "Eval Final Return"),
    ("eval_final_success", "Eval Final Success"),
]

# Colorblind-friendly Okabe-Ito palette
COLORBLIND_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

LINESTYLES = ["-", "--", "-.", ":"]


def load_metrics(root: Path):
    csv_files = sorted(root.glob("*/metrics.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No metrics.csv files found inside {root}")

    runs = {}

    for csv_path in csv_files:
        run_name = csv_path.parent.name
        df = pd.read_csv(csv_path)

        df["update"] = pd.to_numeric(df["update"], errors="coerce")
        df = df.dropna(subset=["update"]).sort_values("update")

        runs[run_name] = df

    return runs


def apply_moving_average(series: pd.Series, window: int | None):
    if window is None or window <= 1:
        return series

    return series.rolling(window=window, min_periods=1).mean()


def plot_metric(runs, metric: str, title: str, moving_average_window: int | None = None):
    plt.figure(figsize=(10, 6))

    ax = plt.gca()
    ax.set_prop_cycle(
        cycler(color=COLORBLIND_COLORS) *
        cycler(linestyle=LINESTYLES)
    )

    for run_name, df in runs.items():
        if metric not in df.columns:
            print(f"Skipping {run_name}: missing column {metric}")
            continue

        # Drop NaNs only for this metric.
        # For eval metrics, this means we only plot actual eval points.
        plot_df = df[["update", metric]].dropna().copy()

        if plot_df.empty:
            print(f"Skipping {run_name}: no valid values for {metric}")
            continue

        plot_df[metric] = apply_moving_average(
            plot_df[metric],
            moving_average_window,
        )

        plt.plot(
            plot_df["update"],
            plot_df[metric],
            linewidth=2.0,
            label=run_name,
        )

    plt.xlabel("Update")
    plt.ylabel(title)

    if moving_average_window is not None and moving_average_window > 1:
        plt.title(f"{title} per Update — Moving Average Window = {moving_average_window}")
    else:
        plt.title(f"{title} per Update")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    suffix = (
        f"_ma{moving_average_window}"
        if moving_average_window is not None and moving_average_window > 1
        else ""
    )

    out_path = ROOT / f"{metric}{suffix}.png"
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    runs = load_metrics(ROOT)

    for metric, title in METRICS:
        plot_metric(
            runs,
            metric,
            title,
            moving_average_window=MOVING_AVERAGE_WINDOW,
        )