import statistics
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import typer

from synthetic import SyntheticBezierDataset


app = typer.Typer()


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(0.95 * (len(sorted_values) - 1))
    return sorted_values[idx]


@app.command()
def main(
    num_samples: int = typer.Option(200, min=1, help="How many samples to benchmark"),
    warmup: int = typer.Option(
        10, min=0, help="Warmup samples before measuring timings"
    ),
    canvas_size: int = typer.Option(256, min=32, help="Canvas width and height"),
    max_segments: int = typer.Option(
        256, min=1, help="Maximum bezier segments per sample"
    ),
    min_shapes: int = typer.Option(1, min=1, help="Minimum shapes per scene"),
    max_shapes: int = typer.Option(10, min=1, help="Maximum shapes per scene"),
    base_seed: int = typer.Option(42, help="Base random seed"),
    epoch: int = typer.Option(0, min=0, help="Synthetic epoch offset"),
):
    """Benchmark SyntheticBezierDataset sample generation latency and throughput."""

    if min_shapes > max_shapes:
        raise typer.BadParameter("min_shapes must be <= max_shapes")

    total_needed = max(num_samples + warmup, 1)

    init_start = time.perf_counter()
    dataset = SyntheticBezierDataset(
        length=total_needed,
        canvas_size=canvas_size,
        max_segments=max_segments,
        min_shapes=min_shapes,
        max_shapes=max_shapes,
        base_seed=base_seed,
        epoch=epoch,
    )
    init_time = time.perf_counter() - init_start

    for i in range(warmup):
        curve_tensor, image_tensor = dataset[i]
        _ = curve_tensor.shape
        _ = image_tensor.shape

    timings: list[float] = []
    total_start = time.perf_counter()
    for i in range(warmup, warmup + num_samples):
        start = time.perf_counter()
        curve_tensor, image_tensor = dataset[i]
        _ = curve_tensor.shape
        _ = image_tensor.shape
        timings.append(time.perf_counter() - start)
    total_time = time.perf_counter() - total_start

    print("Synthetic dataset benchmark")
    print(f"- Init time: {init_time:.4f} s")
    print(f"- Warmup samples: {warmup}")
    print(f"- Measured samples: {len(timings)}")
    print(f"- Total time: {total_time:.4f} s")
    print(f"- Throughput: {len(timings) / total_time:.2f} samples/s")
    print(f"- Mean: {statistics.mean(timings) * 1000:.2f} ms")
    print(f"- Median: {statistics.median(timings) * 1000:.2f} ms")
    print(f"- P95: {_p95(timings) * 1000:.2f} ms")
    print(f"- Min: {min(timings) * 1000:.2f} ms")
    print(f"- Max: {max(timings) * 1000:.2f} ms")


if __name__ == "__main__":
    app()
