import csv
import math
import re
import statistics
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer
from PIL import Image, ImageFilter
from tqdm.auto import tqdm

from raster import render_svg_bg


app = typer.Typer()


PNG_SUFFIX = ".png"
SVG_SUFFIX = ".svg"
COMMAND_RE = re.compile(r"[AaCcHhLlMmQqSsTtVvZz]")


@dataclass
class PairMetrics:
    name: str
    width: int
    height: int
    mse: float
    mae: float
    psnr: float
    ssim: float
    mask_iou: float
    boundary_f1_1px: float
    boundary_f1_2px: float
    boundary_f1_4px: float
    chamfer_px: float
    hausdorff_px: float
    ref_svg_bytes: int | None
    gen_svg_bytes: int | None
    ref_elements: int | None
    gen_elements: int | None
    ref_paths: int | None
    gen_paths: int | None
    ref_path_commands: int | None
    gen_path_commands: int | None
    render_time_ms: float | None


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _render_svg_file(path: Path, width: int, height: int) -> Image.Image:
    return render_svg_bg(path.read_bytes(), width=width, height=height).convert("RGB")


def _image_to_float_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _psnr(mse: float) -> float:
    if mse == 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    # Global SSIM is less sensitive than windowed SSIM, but it is dependency-free
    # and useful for tracking broad structural similarity during experiments.
    c1 = 0.01**2
    c2 = 0.03**2
    values: list[float] = []
    for channel in range(3):
        x = a[..., channel]
        y = b[..., channel]
        mu_x = float(np.mean(x))
        mu_y = float(np.mean(y))
        var_x = float(np.mean((x - mu_x) ** 2))
        var_y = float(np.mean((y - mu_y) ** 2))
        cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))
        numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
        values.append(numerator / denominator if denominator else 1.0)
    return float(np.mean(values))


def _foreground_mask(image: Image.Image, threshold: int) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    # The datasets in this project are typically rendered on white backgrounds.
    return gray < threshold


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def _edge_points(image: Image.Image, threshold: int, max_points: int) -> np.ndarray:
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_mask = np.asarray(edges, dtype=np.uint8) > threshold
    points = np.argwhere(edge_mask)
    if len(points) == 0:
        return points.astype(np.float32)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    return points.astype(np.float32)


def _nearest_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.array([], dtype=np.float32)

    chunk_size = 1024
    nearest: list[np.ndarray] = []
    for start in range(0, len(a), chunk_size):
        chunk = a[start : start + chunk_size]
        distances = np.sqrt(np.sum((chunk[:, None, :] - b[None, :, :]) ** 2, axis=2))
        nearest.append(np.min(distances, axis=1))
    return np.concatenate(nearest)


def _boundary_scores(
    ref_image: Image.Image,
    gen_image: Image.Image,
    edge_threshold: int,
    max_edge_points: int,
) -> tuple[float, float, float, float, float]:
    ref_points = _edge_points(ref_image, edge_threshold, max_edge_points)
    gen_points = _edge_points(gen_image, edge_threshold, max_edge_points)

    if len(ref_points) == 0 and len(gen_points) == 0:
        return 1.0, 1.0, 1.0, 0.0, 0.0
    if len(ref_points) == 0 or len(gen_points) == 0:
        return 0.0, 0.0, 0.0, float("inf"), float("inf")

    ref_to_gen = _nearest_distances(ref_points, gen_points)
    gen_to_ref = _nearest_distances(gen_points, ref_points)
    chamfer = float((np.mean(ref_to_gen) + np.mean(gen_to_ref)) / 2.0)
    hausdorff = float(max(np.max(ref_to_gen), np.max(gen_to_ref)))

    f_scores = []
    for tolerance in (1.0, 2.0, 4.0):
        recall = float(np.mean(ref_to_gen <= tolerance))
        precision = float(np.mean(gen_to_ref <= tolerance))
        if precision + recall == 0:
            f_scores.append(0.0)
        else:
            f_scores.append(2.0 * precision * recall / (precision + recall))

    return f_scores[0], f_scores[1], f_scores[2], chamfer, hausdorff


def _svg_stats(path: Path | None) -> tuple[int, int, int, int] | tuple[None, None, None, None]:
    if path is None or not path.exists():
        return None, None, None, None

    content = path.read_text(encoding="utf-8", errors="replace")
    try:
        root = ET.fromstring(content)
        elements = sum(1 for _ in root.iter())
        paths = 0
        path_commands = 0
        for element in root.iter():
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "path":
                paths += 1
                path_commands += len(COMMAND_RE.findall(element.attrib.get("d", "")))
    except ET.ParseError:
        elements = len(re.findall(r"<[A-Za-z][^!?/\s>]*", content))
        paths = len(re.findall(r"<path\b", content))
        path_commands = len(COMMAND_RE.findall(content))

    return path.stat().st_size, elements, paths, path_commands


def _format_float(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def _mean(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and not math.isinf(float(value))]
    return statistics.mean(clean) if clean else None


def _collect_pair_names(ref_dir: Path, generated_dir: Path) -> list[str]:
    ref_names = {path.stem for path in ref_dir.glob(f"*{PNG_SUFFIX}")}
    gen_names = {path.stem for path in generated_dir.glob(f"*{PNG_SUFFIX}")}
    common = sorted(ref_names & gen_names)
    if not common:
        raise typer.BadParameter("No matching PNG filenames found between folders")
    return common


def _evaluate_pair(
    name: str,
    ref_dir: Path,
    generated_dir: Path,
    *,
    render_svgs: bool,
    foreground_threshold: int,
    edge_threshold: int,
    max_edge_points: int,
) -> PairMetrics:
    ref_png = ref_dir / f"{name}{PNG_SUFFIX}"
    gen_png = generated_dir / f"{name}{PNG_SUFFIX}"
    ref_svg = ref_dir / f"{name}{SVG_SUFFIX}"
    gen_svg = generated_dir / f"{name}{SVG_SUFFIX}"

    ref_image = _open_rgb(ref_png)
    width, height = ref_image.size

    render_time_ms: float | None = None
    if render_svgs:
        start = time.perf_counter()
        if ref_svg.exists():
            ref_image = _render_svg_file(ref_svg, width=width, height=height)
        if gen_svg.exists():
            gen_image = _render_svg_file(gen_svg, width=width, height=height)
        else:
            gen_image = _open_rgb(gen_png)
        render_time_ms = (time.perf_counter() - start) * 1000.0
    else:
        gen_image = _open_rgb(gen_png)

    if gen_image.size != ref_image.size:
        gen_image = gen_image.resize(ref_image.size, Image.Resampling.LANCZOS)

    ref_arr = _image_to_float_array(ref_image)
    gen_arr = _image_to_float_array(gen_image)
    mse = _mse(ref_arr, gen_arr)

    ref_mask = _foreground_mask(ref_image, foreground_threshold)
    gen_mask = _foreground_mask(gen_image, foreground_threshold)
    boundary_f1_1px, boundary_f1_2px, boundary_f1_4px, chamfer, hausdorff = _boundary_scores(
        ref_image,
        gen_image,
        edge_threshold=edge_threshold,
        max_edge_points=max_edge_points,
    )

    ref_svg_bytes, ref_elements, ref_paths, ref_commands = _svg_stats(ref_svg)
    gen_svg_bytes, gen_elements, gen_paths, gen_commands = _svg_stats(gen_svg)

    return PairMetrics(
        name=name,
        width=width,
        height=height,
        mse=mse,
        mae=_mae(ref_arr, gen_arr),
        psnr=_psnr(mse),
        ssim=_ssim(ref_arr, gen_arr),
        mask_iou=_mask_iou(ref_mask, gen_mask),
        boundary_f1_1px=boundary_f1_1px,
        boundary_f1_2px=boundary_f1_2px,
        boundary_f1_4px=boundary_f1_4px,
        chamfer_px=chamfer,
        hausdorff_px=hausdorff,
        ref_svg_bytes=ref_svg_bytes,
        gen_svg_bytes=gen_svg_bytes,
        ref_elements=ref_elements,
        gen_elements=gen_elements,
        ref_paths=ref_paths,
        gen_paths=gen_paths,
        ref_path_commands=ref_commands,
        gen_path_commands=gen_commands,
        render_time_ms=render_time_ms,
    )


def _write_csv(path: Path, rows: list[PairMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PairMetrics.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _print_summary(rows: list[PairMetrics]) -> None:
    summary = {
        "pairs": len(rows),
        "mse": _mean([row.mse for row in rows]),
        "mae": _mean([row.mae for row in rows]),
        "psnr": _mean([row.psnr for row in rows]),
        "ssim": _mean([row.ssim for row in rows]),
        "mask_iou": _mean([row.mask_iou for row in rows]),
        "boundary_f1_1px": _mean([row.boundary_f1_1px for row in rows]),
        "boundary_f1_2px": _mean([row.boundary_f1_2px for row in rows]),
        "boundary_f1_4px": _mean([row.boundary_f1_4px for row in rows]),
        "chamfer_px": _mean([row.chamfer_px for row in rows]),
        "hausdorff_px": _mean([row.hausdorff_px for row in rows]),
        "gen_svg_bytes": _mean([row.gen_svg_bytes for row in rows]),
        "gen_elements": _mean([row.gen_elements for row in rows]),
        "gen_paths": _mean([row.gen_paths for row in rows]),
        "gen_path_commands": _mean([row.gen_path_commands for row in rows]),
        "render_time_ms": _mean([row.render_time_ms for row in rows]),
    }

    for key, value in summary.items():
        print(f"{key}: {_format_float(value)}")


@app.command()
def main(
    ref_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    generated_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    output_csv: Path | None = typer.Option(
        None,
        "--output-csv",
        "-o",
        help="Optional path for per-image metrics.",
    ),
    limit: int | None = typer.Option(
        None,
        min=1,
        help="Evaluate only the first N matching pairs.",
    ),
    render_svgs: bool = typer.Option(
        False,
        help="Render SVG files at the reference PNG size before computing image metrics.",
    ),
    foreground_threshold: int = typer.Option(
        250,
        min=0,
        max=255,
        help="Grayscale threshold for foreground masks; lower values are considered foreground.",
    ),
    edge_threshold: int = typer.Option(
        16,
        min=0,
        max=255,
        help="Threshold applied after PIL FIND_EDGES for contour metrics.",
    ),
    max_edge_points: int = typer.Option(
        4096,
        min=128,
        help="Maximum sampled edge points per image for Chamfer/Hausdorff computation.",
    ),
):
    """Evaluate matching raster/SVG vectorization outputs in two folders."""

    names = _collect_pair_names(ref_dir, generated_dir)
    if limit is not None:
        names = names[:limit]

    rows = [
        _evaluate_pair(
            name,
            ref_dir,
            generated_dir,
            render_svgs=render_svgs,
            foreground_threshold=foreground_threshold,
            edge_threshold=edge_threshold,
            max_edge_points=max_edge_points,
        )
        for name in tqdm(names, desc="Evaluating pairs")
    ]

    _print_summary(rows)
    if output_csv is not None:
        _write_csv(output_csv, rows)
        print(f"wrote_csv: {output_csv}")


if __name__ == "__main__":
    app()
