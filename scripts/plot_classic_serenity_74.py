from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
import pandas as pd
import seaborn as sns


CSV_PATH = Path("results/wandb/classic-serenity-74_fz13xy6t.csv")
OUTPUT_DIR = Path("text/assets/wandb")

STEP_COLUMN = "_step"
TRAIN_LOSS_COLUMN = "train_loss"
MSE_COLUMNS = {
    "train_inference/image_mse": "Train",
    "val/image_mse": "Validation",
}
TRAIN_LOSS_SMOOTHING_WINDOW = 75
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


def save_figure(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")


def add_smoothed_column(
    df: pd.DataFrame,
    value_column: str,
    window: int,
    group_column: str | None = None,
) -> pd.DataFrame:
    df = df.sort_values(STEP_COLUMN).copy()
    if group_column is None:
        df[f"{value_column}_smooth"] = (
            df[value_column].rolling(window=window, min_periods=1, center=True).mean()
        )
        return df

    df[f"{value_column}_smooth"] = df.groupby(group_column, group_keys=False)[
        value_column
    ].transform(
        lambda values: values.rolling(window=window, min_periods=1, center=True).mean()
    )
    return df


def compact_decimal(value: float, _position: int) -> str:
    if value <= 0:
        return ""
    if value >= 1:
        return f"{value:g}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def plot_train_loss(df: pd.DataFrame) -> None:
    train_loss = df[[STEP_COLUMN, TRAIN_LOSS_COLUMN]].dropna()
    train_loss = add_smoothed_column(
        train_loss,
        value_column=TRAIN_LOSS_COLUMN,
        window=TRAIN_LOSS_SMOOTHING_WINDOW,
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    sns.scatterplot(
        data=train_loss,
        x=STEP_COLUMN,
        y=TRAIN_LOSS_COLUMN,
        ax=ax,
        color="#3267a8",
        alpha=0.055,
        s=7,
        linewidth=0,
    )
    sns.lineplot(
        data=train_loss,
        x=STEP_COLUMN,
        y=f"{TRAIN_LOSS_COLUMN}_smooth",
        ax=ax,
        color="#3267a8",
    )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5, 7), numticks=24))
    ax.yaxis.set_major_formatter(FuncFormatter(compact_decimal))
    ax.yaxis.set_minor_formatter(FuncFormatter(compact_decimal))
    ax.margins(x=0.01)
    ax.grid(axis="x", alpha=0.15)
    ax.grid(axis="y", which="minor", alpha=0.12)

    fig.tight_layout()
    save_figure(fig, "classic-serenity-74_train_loss")
    plt.close(fig)


def plot_mse_metrics(df: pd.DataFrame) -> None:
    mse = (
        df[[STEP_COLUMN, *MSE_COLUMNS.keys()]]
        .melt(
            id_vars=STEP_COLUMN,
            var_name="metric",
            value_name="image_mse",
        )
        .dropna()
    )
    mse["metric"] = mse["metric"].map(MSE_COLUMNS)
    mse = add_smoothed_column(
        mse,
        value_column="image_mse",
        window=MSE_SMOOTHING_WINDOW,
        group_column="metric",
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    sns.scatterplot(
        data=mse,
        x=STEP_COLUMN,
        y="image_mse",
        hue="metric",
        ax=ax,
        alpha=0.09,
        s=20,
        linewidth=0,
        legend=False,
    )
    sns.lineplot(
        data=mse,
        x=STEP_COLUMN,
        y="image_mse_smooth",
        hue="metric",
        ax=ax,
    )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Image MSE")
    ax.set_yscale("log")
    ax.legend(title=None, loc="best")
    ax.margins(x=0.03)
    ax.grid(axis="x", alpha=0.15)
    ax.grid(axis="y", which="minor", alpha=0.12)

    fig.tight_layout()
    save_figure(fig, "classic-serenity-74_image_mse")
    plt.close(fig)


def main() -> None:
    configure_style()
    df = pd.read_csv(CSV_PATH)

    missing_columns = {
        STEP_COLUMN,
        TRAIN_LOSS_COLUMN,
        *MSE_COLUMNS.keys(),
    } - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    plot_train_loss(df)
    plot_mse_metrics(df)
    print(f"Saved plots to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
