import statistics
import time
from pathlib import Path

import typer
from datasets import load_dataset

from raster import render_svg, render_svg_bg, render2


app = typer.Typer()


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(0.95 * (len(sorted_values) - 1))
    return sorted_values[idx]


def _load_svgs_from_files(svg_dir: Path, pattern: str, limit: int) -> list[str]:
    paths = sorted(svg_dir.glob(pattern))
    if not paths:
        raise ValueError(f"No SVG files matched '{pattern}' in {svg_dir}")
    if limit > 0:
        paths = paths[:limit]
    return [path.read_text(encoding="utf-8") for path in paths]


def _load_svgs_from_dataset(
    dataset_name: str,
    split: str,
    limit: int,
    seed: int,
) -> list[str]:
    dataset = load_dataset(dataset_name, split=split)
    if seed >= 0:
        dataset = dataset.shuffle(seed=seed)
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [item["item_svg"] for item in dataset]


@app.command()
def main(
    source: str = typer.Option(
        "dataset",
        help="Where to load SVGs from: dataset or files",
    ),
    dataset_name: str = typer.Option(
        "JosefKuchar/bezier-dataset",
        help="Hugging Face dataset name (used when source=dataset)",
    ),
    split: str = typer.Option(
        "train",
        help="Dataset split (used when source=dataset)",
    ),
    svg_dir: Path = typer.Option(
        Path("."),
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory with SVG files (used when source=files)",
    ),
    svg_pattern: str = typer.Option(
        "*.svg",
        help="SVG glob pattern (used when source=files)",
    ),
    num_svgs: int = typer.Option(
        200,
        min=1,
        help="How many SVGs to benchmark",
    ),
    warmup: int = typer.Option(
        10,
        min=0,
        help="Warmup iterations before measuring",
    ),
    seed: int = typer.Option(
        42,
        help="Shuffle seed for dataset source; set -1 to disable shuffling",
    ),
    white_background: bool = typer.Option(
        False,
        help="Use render_svg_bg instead of render_svg",
    ),
):
    """Benchmark SVG rastering throughput and latency."""

    if source not in {"dataset", "files"}:
        raise typer.BadParameter("source must be either 'dataset' or 'files'")

    if source == "dataset":
        svgs = _load_svgs_from_dataset(dataset_name, split, num_svgs, seed)
        print(f"Loaded {len(svgs)} SVGs from {dataset_name}:{split}")
    else:
        svgs = _load_svgs_from_files(svg_dir, svg_pattern, num_svgs)
        print(f"Loaded {len(svgs)} SVGs from {svg_dir} ({svg_pattern})")

    renderer = render_svg_bg if white_background else render_svg

    for i in range(warmup):
        renderer(svgs[i % len(svgs)])

    timings: list[float] = []
    total_start = time.perf_counter()
    for svg in svgs:
        start = time.perf_counter()
        image = renderer(svg)
        _ = image.size
        timings.append(time.perf_counter() - start)
    total_time = time.perf_counter() - total_start

    print("\nBenchmark results")
    print(f"- Runs: {len(timings)}")
    print(f"- Total time: {total_time:.4f} s")
    print(f"- Throughput: {len(timings) / total_time:.2f} SVG/s")
    print(f"- Mean: {statistics.mean(timings) * 1000:.2f} ms")
    print(f"- Median: {statistics.median(timings) * 1000:.2f} ms")
    print(f"- P95: {_p95(timings) * 1000:.2f} ms")
    print(f"- Min: {min(timings) * 1000:.2f} ms")
    print(f"- Max: {max(timings) * 1000:.2f} ms")


if __name__ == "__main__":
    app()
