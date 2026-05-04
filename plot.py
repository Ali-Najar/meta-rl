from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler


COLORBLIND_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]
LINESTYLES = ["-", "--", "-.", ":"]


def parse_int_list(value):
    if value is None or str(value).strip() == "":
        return []
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def load_metrics(root: Path):
    csv_files = sorted(root.glob("*/metrics.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No metrics.csv files found inside {root}")

    runs = {}
    for csv_path in csv_files:
        run_name = csv_path.parent.name
        ##################################
        if "ML10" not in run_name:
            continue
        ##################################
        df = pd.read_csv(csv_path)
        x_col = "timestep" if "timestep" in df.columns else "update"
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df = df.dropna(subset=[x_col]).sort_values(x_col)
        runs[run_name] = (df, x_col)
    return runs


def apply_moving_average(series: pd.Series, window: int | None):
    if window is None or window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def plot_metric(runs, root: Path, metric: str, title: str, moving_average_window: int | None = None):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_prop_cycle(cycler(color=COLORBLIND_COLORS) * cycler(linestyle=LINESTYLES))

    plotted = False
    for run_name, (df, x_col) in runs.items():
        if metric not in df.columns:
            print(f"Skipping {run_name}: missing column {metric}")
            continue

        plot_df = df[[x_col, metric]].dropna().copy()
        if plot_df.empty:
            print(f"Skipping {run_name}: no valid values for {metric}")
            continue

        plot_df[metric] = apply_moving_average(plot_df[metric], moving_average_window)
        plt.plot(plot_df[x_col], plot_df[metric], linewidth=2.0, label=run_name)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Timesteps")
    plt.ylabel(title)
    if moving_average_window is not None and moving_average_window > 1:
        plt.title(f"{title} vs Timesteps - Moving Average Window = {moving_average_window}")
    else:
        plt.title(f"{title} vs Timesteps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    suffix = f"_ma{moving_average_window}" if moving_average_window is not None and moving_average_window > 1 else ""
    out_path = root / f"{metric}{suffix}.png"
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training/evaluation metrics vs timesteps.")
    parser.add_argument("--root", type=str, default="csv_outputs")
    parser.add_argument("--moving_average_window", type=int, default=1)
    parser.add_argument(
        "--eval_report_lengths",
        type=str,
        default=25,
        help="Comma-separated eval prefix lengths to plot anysuccess for, e.g. 5,25.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    runs = load_metrics(root)

    default_metrics = [
        ("rollout_anysuccess", "Rollout Any Success"),
        ("rollout_trial_return", "Rollout Trial Return"),
    ]

    for metric, title in default_metrics:
        plot_metric(runs, root, metric, title, args.moving_average_window)

    for length in parse_int_list(args.eval_report_lengths):
        metric = f"eval_len_{length}_anysuccess"
        title = f"Eval Any Success Up To Episode {length}"
        plot_metric(runs, root, metric, title, args.moving_average_window)


if __name__ == "__main__":
    main()
