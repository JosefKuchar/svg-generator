from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import pandas as pd
import seaborn as sns


RUNS = {
    "With image encoder": Path("results/wandb/feasible-sea-46_4qe1lvw8.csv"),
    "Without image encoder": Path("results/wandb/rose-elevator-45_hphazsct.csv"),
}
OUTPUT_DIR = Path("text/assets/wandb")

STEP_COLUMN = "_step"
MAX_STEP = 150_000
TRAIN_LOSS_COLUMN = "train_loss"
MSE_COLUMNS = {
    "train_inference/image_mse": "Train",
    "val/image_mse": "Validation",
}
TRAIN_LOSS_SMOOTHING_WINDOW = 15
MSE_SMOOTHING_WINDOW = 5


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


def add_smoothed_column(
    df: pd.DataFrame,
    value_column: str,
    window: int,
    group_column: str,
) -> pd.DataFrame:
    df = df.sort_values([group_column, STEP_COLUMN]).copy()
    df[f"{value_column}_smooth"] = df.groupby(group_column, group_keys=False)[
        value_column
    ].transform(
        lambda values: values.rolling(
            window=window,
            min_periods=1,
            center=True,
        ).mean()
    )
    return df


def load_csvs(required_columns: set[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    missing_paths = [str(path) for path in RUNS.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing W&B export(s): {', '.join(missing_paths)}")

    for run_name, csv_path in RUNS.items():
        df = pd.read_csv(csv_path)
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is missing required column(s): {missing}")
        frames[run_name] = df

    return frames


def shared_max_step(frames: dict[str, pd.DataFrame]) -> int:
    return min(MAX_STEP, *(int(df[STEP_COLUMN].max()) for df in frames.values()))


def load_train_loss() -> pd.DataFrame:
    frames = load_csvs({STEP_COLUMN, TRAIN_LOSS_COLUMN})
    max_step = shared_max_step(frames)
    loss_frames: list[pd.DataFrame] = []

    for run_name, df in frames.items():
        loss = (
            df.loc[df[STEP_COLUMN] <= max_step, [STEP_COLUMN, TRAIN_LOSS_COLUMN]]
            .dropna()
            .copy()
        )
        loss["run"] = run_name
        loss_frames.append(loss)

    combined = pd.concat(loss_frames, ignore_index=True)
    return add_smoothed_column(
        combined,
        value_column=TRAIN_LOSS_COLUMN,
        window=TRAIN_LOSS_SMOOTHING_WINDOW,
        group_column="run",
    )


def load_mse() -> pd.DataFrame:
    frames = load_csvs({STEP_COLUMN, *MSE_COLUMNS.keys()})
    max_step = shared_max_step(frames)
    mse_frames: list[pd.DataFrame] = []

    for run_name, df in frames.items():
        mse = (
            df.loc[df[STEP_COLUMN] <= max_step, [STEP_COLUMN, *MSE_COLUMNS.keys()]]
            .melt(
                id_vars=STEP_COLUMN,
                var_name="metric",
                value_name="image_mse",
            )
            .dropna()
        )
        mse["run"] = run_name
        mse["metric"] = mse["metric"].map(MSE_COLUMNS)
        mse_frames.append(mse)

    combined = pd.concat(mse_frames, ignore_index=True)
    combined["series"] = combined["run"] + " " + combined["metric"]
    return add_smoothed_column(
        combined,
        value_column="image_mse",
        window=MSE_SMOOTHING_WINDOW,
        group_column="series",
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")


def configure_log_axis(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5, 7), numticks=24))
    ax.yaxis.set_major_formatter(FuncFormatter(compact_decimal))
    ax.yaxis.set_minor_formatter(FuncFormatter(compact_decimal))
    ax.margins(x=0.03)
    ax.grid(axis="x", alpha=0.15)
    ax.grid(axis="y", which="minor", alpha=0.12)


def plot_train_loss(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    sns.scatterplot(
        data=df,
        x=STEP_COLUMN,
        y=TRAIN_LOSS_COLUMN,
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
        y=f"{TRAIN_LOSS_COLUMN}_smooth",
        hue="run",
        ax=ax,
    )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    configure_log_axis(ax)
    ax.legend(title=None, loc="best")

    fig.tight_layout()
    save_figure(fig, "image-encoder-ablation_train_loss")
    plt.close(fig)


def plot_mse(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    sns.scatterplot(
        data=df,
        x=STEP_COLUMN,
        y="image_mse",
        hue="run",
        style="metric",
        ax=ax,
        alpha=0.08,
        s=18,
        linewidth=0,
        legend=False,
    )
    sns.lineplot(
        data=df,
        x=STEP_COLUMN,
        y="image_mse_smooth",
        hue="run",
        style="metric",
        ax=ax,
    )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Image MSE")
    configure_log_axis(ax)
    ax.legend(title=None, loc="best")

    fig.tight_layout()
    save_figure(fig, "image-encoder-ablation_image_mse")
    plt.close(fig)


def main() -> None:
    configure_style()
    loss = load_train_loss()
    mse = load_mse()
    plot_train_loss(loss)
    plot_mse(mse)
    max_step = min(int(loss[STEP_COLUMN].max()), int(mse[STEP_COLUMN].max()))
    print(f"Saved plots to {OUTPUT_DIR.resolve()} using steps 0-{max_step}")


if __name__ == "__main__":
    main()
