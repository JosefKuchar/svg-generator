import csv
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import typer
from PIL import Image
from tqdm.auto import tqdm

from evaluate_vectorization import (
    SVG_SUFFIX,
    _boundary_scores,
    _format_float,
    _foreground_mask,
    _image_to_byte_float_array,
    _mae,
    _mask_iou,
    _mse,
    _psnr,
    _render_error_message,
    _ssim,
    _svg_stats,
)
from raster import render_svg_bg


app = typer.Typer()

PNG_SUFFIXES = {".png"}


@dataclass
class RasterVectorizationMetrics:
    name: str
    png_path: str
    svg_path: str
    valid: bool
    width: int
    height: int
    mse: float | None
    mae: float | None
    psnr: float | None
    ssim: float | None
    mask_iou: float | None
    boundary_f1_1px: float | None
    boundary_f1_2px: float | None
    boundary_f1_4px: float | None
    chamfer_px: float | None
    hausdorff_px: float | None
    svg_bytes: int | None
    svg_elements: int | None
    svg_paths: int | None
    svg_path_commands: int | None
    render_time_ms: float | None
    error: str | None


def _png_paths(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in PNG_SUFFIXES
    )


def _load_reference_png(path: Path, size: int | None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def _render_svg_file(path: Path, width: int, height: int) -> Image.Image:
    return render_svg_bg(path.read_bytes(), width=width, height=height).convert("RGB")


def _empty_invalid_row(
    *,
    name: str,
    png_path: Path,
    svg_path: Path,
    width: int,
    height: int,
    render_time_ms: float | None,
    error: str,
) -> RasterVectorizationMetrics:
    svg_bytes, svg_elements, svg_paths, svg_path_commands = _svg_stats(svg_path)
    return RasterVectorizationMetrics(
        name=name,
        png_path=str(png_path),
        svg_path=str(svg_path),
        valid=False,
        width=width,
        height=height,
        mse=None,
        mae=None,
        psnr=None,
        ssim=None,
        mask_iou=None,
        boundary_f1_1px=None,
        boundary_f1_2px=None,
        boundary_f1_4px=None,
        chamfer_px=None,
        hausdorff_px=None,
        svg_bytes=svg_bytes,
        svg_elements=svg_elements,
        svg_paths=svg_paths,
        svg_path_commands=svg_path_commands,
        render_time_ms=render_time_ms,
        error=error,
    )


def _evaluate_pair(
    png_path: Path,
    svg_dir: Path,
    *,
    raster_output_dir: Path | None,
    raster_size: int | None,
    foreground_threshold: int,
    edge_threshold: int,
    max_edge_points: int,
) -> RasterVectorizationMetrics:
    name = png_path.stem
    svg_path = svg_dir / f"{name}{SVG_SUFFIX}"

    ref_image = _load_reference_png(png_path, raster_size)
    width, height = ref_image.size

    if not svg_path.exists():
        return _empty_invalid_row(
            name=name,
            png_path=png_path,
            svg_path=svg_path,
            width=width,
            height=height,
            render_time_ms=None,
            error="missing SVG",
        )

    start = time.perf_counter()
    try:
        gen_image = _render_svg_file(svg_path, width=width, height=height)
    except Exception as error:
        return _empty_invalid_row(
            name=name,
            png_path=png_path,
            svg_path=svg_path,
            width=width,
            height=height,
            render_time_ms=(time.perf_counter() - start) * 1000.0,
            error=_render_error_message(error),
        )

    render_time_ms = (time.perf_counter() - start) * 1000.0

    if gen_image.size != ref_image.size:
        gen_image = gen_image.resize(ref_image.size, Image.Resampling.LANCZOS)

    if raster_output_dir is not None:
        raster_output_dir.mkdir(parents=True, exist_ok=True)
        gen_image.save(raster_output_dir / f"{name}.png")

    ref_arr = _image_to_byte_float_array(ref_image)
    gen_arr = _image_to_byte_float_array(gen_image)
    mse = _mse(ref_arr, gen_arr)

    ref_unit_arr = ref_arr / 255.0
    gen_unit_arr = gen_arr / 255.0

    ref_mask = _foreground_mask(ref_image, foreground_threshold)
    gen_mask = _foreground_mask(gen_image, foreground_threshold)
    boundary_f1_1px, boundary_f1_2px, boundary_f1_4px, chamfer, hausdorff = (
        _boundary_scores(
            ref_image,
            gen_image,
            edge_threshold=edge_threshold,
            max_edge_points=max_edge_points,
        )
    )

    svg_bytes, svg_elements, svg_paths, svg_path_commands = _svg_stats(svg_path)

    return RasterVectorizationMetrics(
        name=name,
        png_path=str(png_path),
        svg_path=str(svg_path),
        valid=True,
        width=width,
        height=height,
        mse=mse,
        mae=_mae(ref_arr, gen_arr),
        psnr=_psnr(mse),
        ssim=_ssim(ref_unit_arr, gen_unit_arr),
        mask_iou=_mask_iou(ref_mask, gen_mask),
        boundary_f1_1px=boundary_f1_1px,
        boundary_f1_2px=boundary_f1_2px,
        boundary_f1_4px=boundary_f1_4px,
        chamfer_px=chamfer,
        hausdorff_px=hausdorff,
        svg_bytes=svg_bytes,
        svg_elements=svg_elements,
        svg_paths=svg_paths,
        svg_path_commands=svg_path_commands,
        render_time_ms=render_time_ms,
        error=None,
    )


def _mean(values: list[float | int | None]) -> float | None:
    clean = [
        float(value)
        for value in values
        if value is not None and not math.isinf(float(value))
    ]
    return statistics.mean(clean) if clean else None


def _write_csv(path: Path, rows: list[RasterVectorizationMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(RasterVectorizationMetrics.__dataclass_fields__)
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _print_summary(rows: list[RasterVectorizationMetrics]) -> None:
    valid_rows = [row for row in rows if row.valid]
    invalid_rows = [row for row in rows if not row.valid]
    summary = {
        "images": len(rows),
        "valid": len(valid_rows),
        "invalid": len(invalid_rows),
        "valid_rate": len(valid_rows) / len(rows) if rows else None,
        "mse": _mean([row.mse for row in valid_rows]),
        "mae": _mean([row.mae for row in valid_rows]),
        "psnr": _mean([row.psnr for row in valid_rows]),
        "ssim": _mean([row.ssim for row in valid_rows]),
        "mask_iou": _mean([row.mask_iou for row in valid_rows]),
        "boundary_f1_1px": _mean([row.boundary_f1_1px for row in valid_rows]),
        "boundary_f1_2px": _mean([row.boundary_f1_2px for row in valid_rows]),
        "boundary_f1_4px": _mean([row.boundary_f1_4px for row in valid_rows]),
        "chamfer_px": _mean([row.chamfer_px for row in valid_rows]),
        "hausdorff_px": _mean([row.hausdorff_px for row in valid_rows]),
        "svg_bytes": _mean([row.svg_bytes for row in valid_rows]),
        "svg_elements": _mean([row.svg_elements for row in valid_rows]),
        "svg_paths": _mean([row.svg_paths for row in valid_rows]),
        "svg_path_commands": _mean([row.svg_path_commands for row in valid_rows]),
        "render_time_ms": _mean([row.render_time_ms for row in valid_rows]),
    }

    for key, value in summary.items():
        print(f"{key}: {_format_float(value)}")

    if invalid_rows:
        missing = sum(row.error == "missing SVG" for row in invalid_rows)
        render_errors = len(invalid_rows) - missing
        print(f"missing_svg: {missing}")
        print(f"render_errors: {render_errors}")


@app.command()
def main(
    png_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    svg_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    output_csv: Path | None = typer.Option(
        None,
        "--output-csv",
        "-o",
        help="Optional path for per-image metrics. Invalid rows are included.",
    ),
    limit: int | None = typer.Option(
        None,
        min=1,
        help="Evaluate only the first N PNG images by filename.",
    ),
    raster_output_dir: Path | None = typer.Option(
        None,
        help="Optional directory where rendered SVG PNGs are written.",
    ),
    raster_size: int | None = typer.Option(
        None,
        min=1,
        help="Optional square size used for both reference PNGs and rendered SVGs.",
    ),
    foreground_threshold: int = typer.Option(
        250,
        min=0,
        max=255,
        help="Grayscale threshold for foreground masks; lower values are foreground.",
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
    """Evaluate SVG vectorizations against reference PNG raster images."""

    png_paths = _png_paths(png_dir)
    if not png_paths:
        raise typer.BadParameter(f"No PNG files found in folder: {png_dir}")
    if limit is not None:
        png_paths = png_paths[:limit]

    rows = [
        _evaluate_pair(
            png_path,
            svg_dir,
            raster_output_dir=raster_output_dir,
            raster_size=raster_size,
            foreground_threshold=foreground_threshold,
            edge_threshold=edge_threshold,
            max_edge_points=max_edge_points,
        )
        for png_path in tqdm(png_paths, desc="Evaluating raster/vector pairs")
    ]

    _print_summary(rows)
    if output_csv is not None:
        _write_csv(output_csv, rows)
        print(f"wrote_csv: {output_csv}")


if __name__ == "__main__":
    app()
