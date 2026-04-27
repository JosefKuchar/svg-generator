from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import pandas as pd
import seaborn as sns


RUNS = {
    "Pretrained raster model": Path("results/wandb/pretrained_model_9lhzpdb0.csv"),
    "Reset model": Path("results/wandb/reset_model_kg4mahzv.csv"),
}
OUTPUT_DIR = Path("text/assets/wandb")
OUTPUT_NAME = "pretrained-vs-reset_train_loss"

STEP_COLUMN = "_step"
LOSS_COLUMN = "train_loss"
SMOOTHING_WINDOW = 15


def configure_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="deep",
        font="DejaVu Sans",
        rc={
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 14,
            "legend.frameon": False,
            "grid.alpha": 0.28,
            "lines.linewidth": 2.1,
        },
    )


def compact_decimal(value: float, _position: int) -> str:
    if value <= 0:
        return ""
    if value >= 1:
        return f"{value:g}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def load_runs() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing_paths = [str(path) for path in RUNS.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing W&B export(s): {', '.join(missing_paths)}")

    for run_name, csv_path in RUNS.items():
        df = pd.read_csv(csv_path)
        missing_columns = {STEP_COLUMN, LOSS_COLUMN} - set(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is missing required column(s): {missing}")

        df = df[[STEP_COLUMN, LOSS_COLUMN]].dropna().copy()
        df["run"] = run_name
        frames.append(df)

    max_shared_step = min(int(df[STEP_COLUMN].max()) for df in frames)
    trimmed = [
        df[df[STEP_COLUMN] <= max_shared_step].copy()
        for df in frames
    ]
    combined = pd.concat(trimmed, ignore_index=True)
    combined = combined.sort_values(["run", STEP_COLUMN])
    combined[f"{LOSS_COLUMN}_smooth"] = combined.groupby("run", group_keys=False)[
        LOSS_COLUMN
    ].transform(
        lambda values: values.rolling(
            window=SMOOTHING_WINDOW,
            min_periods=1,
            center=True,
        ).mean()
    )
    return combined


def save_figure(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_NAME}.pdf", bbox_inches="tight")


def plot_loss(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    sns.scatterplot(
        data=df,
        x=STEP_COLUMN,
        y=LOSS_COLUMN,
        hue="run",
        ax=ax,
        alpha=0.08,
        s=13,
        linewidth=0,
        legend=False,
    )
    sns.lineplot(
        data=df,
        x=STEP_COLUMN,
        y=f"{LOSS_COLUMN}_smooth",
        hue="run",
        ax=ax,
    )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5, 7), numticks=24))
    ax.yaxis.set_major_formatter(FuncFormatter(compact_decimal))
    ax.yaxis.set_minor_formatter(FuncFormatter(compact_decimal))
    ax.legend(title=None, loc="best")
    ax.margins(x=0.03)
    ax.grid(axis="x", alpha=0.15)
    ax.grid(axis="y", which="minor", alpha=0.12)

    fig.tight_layout()
    save_figure(fig)
    plt.close(fig)


def main() -> None:
    configure_style()
    df = load_runs()
    plot_loss(df)
    max_step = int(df[STEP_COLUMN].max())
    print(f"Saved {OUTPUT_DIR / f'{OUTPUT_NAME}.pdf'} using steps 0-{max_step}")


if __name__ == "__main__":
    main()
