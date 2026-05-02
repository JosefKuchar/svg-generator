from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import pandas as pd
import seaborn as sns


RUNS = {
    "Flow matching": Path("results/wandb/classic-serenity-74_fz13xy6t.csv"),
    "Autoregressive": Path("results/wandb/stilted-fire-78_ja95uglj.csv"),
}
OUTPUT_DIR = Path("text/assets/wandb")
OUTPUT_NAME = "flow-matching-vs-autoregressive_image_mse"

STEP_COLUMN = "_step"
MAX_STEP = 250_000
MSE_COLUMNS = {
    "train_inference/image_mse": "Train",
    "val/image_mse": "Validation",
}
SMOOTHING_WINDOW = 5


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
    required_columns = {STEP_COLUMN, *MSE_COLUMNS.keys()}

    for run_name, csv_path in RUNS.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing W&B export: {csv_path}")

        df = pd.read_csv(csv_path)
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is missing required column(s): {missing}")

        df = df.loc[df[STEP_COLUMN] <= MAX_STEP, [STEP_COLUMN, *MSE_COLUMNS.keys()]]
        df = df.melt(
            id_vars=STEP_COLUMN,
            var_name="metric",
            value_name="image_mse",
        ).dropna()
        df["run"] = run_name
        df["metric"] = df["metric"].map(MSE_COLUMNS)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["run", "metric", STEP_COLUMN])
    combined["series"] = combined["run"] + " " + combined["metric"]
    combined["image_mse_smooth"] = combined.groupby("series", group_keys=False)[
        "image_mse"
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
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5, 7), numticks=24))
    ax.yaxis.set_major_formatter(FuncFormatter(compact_decimal))
    ax.yaxis.set_minor_formatter(FuncFormatter(compact_decimal))
    ax.set_xlim(0, MAX_STEP)
    ax.legend(title=None, loc="best")
    ax.margins(x=0.01)
    ax.grid(axis="x", alpha=0.15)
    ax.grid(axis="y", which="minor", alpha=0.12)

    fig.tight_layout()
    save_figure(fig)
    plt.close(fig)


def main() -> None:
    configure_style()
    df = load_runs()
    plot_mse(df)
    print(f"Saved {OUTPUT_DIR / f'{OUTPUT_NAME}.pdf'} using steps 0-{MAX_STEP}")


if __name__ == "__main__":
    main()
