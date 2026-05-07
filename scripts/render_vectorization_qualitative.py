from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import typer
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from raster import render_svg_bg


app = typer.Typer()

SAMPLE_NAMES = tuple(f"{index:04d}" for index in range(6))
IMAGE_SIZE = 512
WHITE = (255, 255, 255)


@dataclass(frozen=True)
class Method:
    label: str
    source_dir: Path
    output_dir: str


VALIDATION_METHODS = (
    Method("Proposed", Path("model_outputs/0627_reference"), "proposed"),
    Method("OmniSVG 4B", Path("vectorization_results/omni/i2i_4b"), "omnisvg_4b"),
    Method("OmniSVG 8B", Path("vectorization_results/omni/i2i_8b"), "omnisvg_8b"),
    Method("StarVector 1B", Path("vectorization_results/starvector/1b"), "starvector_1b"),
    Method("StarVector 8B", Path("vectorization_results/starvector/8b"), "starvector_8b"),
)

SYNTHETIC_METHODS = (
    Method("Proposed", Path("model_outputs/0627_synthetic"), "proposed"),
    Method(
        "OmniSVG 4B",
        Path("vectorization_results/omni/i2i_4b_synthetic"),
        "omnisvg_4b",
    ),
    Method(
        "OmniSVG 8B",
        Path("vectorization_results/omni/i2i_8b_synthetic"),
        "omnisvg_8b",
    ),
    Method(
        "StarVector 1B",
        Path("vectorization_results/starvector/1b_synthetic"),
        "starvector_1b",
    ),
    Method(
        "StarVector 8B",
        Path("vectorization_results/starvector/8b_synthetic"),
        "starvector_8b",
    ),
)


def _white_image(size: int) -> Image.Image:
    return Image.new("RGB", (size, size), WHITE)


def _save_white(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _white_image(size).save(path)


def _render_svg_or_white(svg_path: Path, output_path: Path, size: int) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not svg_path.exists():
        _save_white(output_path, size)
        return False

    try:
        image = render_svg_bg(svg_path.read_bytes(), width=size, height=size).convert("RGB")
    except (OSError, RuntimeError, subprocess.CalledProcessError):
        _save_white(output_path, size)
        return False

    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    image.save(output_path)
    return True


def _copy_image_or_white(source_path: Path, output_path: Path, size: int) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        _save_white(output_path, size)
        return False

    try:
        image = Image.open(source_path).convert("RGB")
        image.load()
    except OSError:
        _save_white(output_path, size)
        return False

    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    image.save(output_path)
    return True


def _find_synthetic_reference_dir() -> Path:
    for candidate in (Path("exports/synthetic_images"), Path("exports/synthetic-samples")):
        if candidate.exists():
            return candidate
    return Path("exports/synthetic_images")


def _write_references(
    reference_dir: Path,
    output_dir: Path,
    *,
    size: int,
) -> tuple[int, int]:
    written = 0
    missing = 0
    for name in SAMPLE_NAMES:
        output_path = output_dir / "reference" / f"{name}.png"
        png_path = reference_dir / f"{name}.png"
        svg_path = reference_dir / f"{name}.svg"
        if png_path.exists():
            ok = _copy_image_or_white(png_path, output_path, size)
        else:
            ok = _render_svg_or_white(svg_path, output_path, size)
        written += int(ok)
        missing += int(not ok)
    return written, missing


def _write_methods(
    methods: tuple[Method, ...],
    output_dir: Path,
    *,
    size: int,
) -> tuple[int, int]:
    written = 0
    missing = 0
    for method in methods:
        for name in SAMPLE_NAMES:
            ok = _render_svg_or_white(
                method.source_dir / f"{name}.svg",
                output_dir / method.output_dir / f"{name}.png",
                size,
            )
            written += int(ok)
            missing += int(not ok)
    return written, missing


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path("text/assets/vectorization_qualitative"),
        help="Directory where rendered PNG assets are written.",
    ),
    size: int = typer.Option(
        IMAGE_SIZE,
        min=1,
        help="Square output image size in pixels.",
    ),
) -> None:
    """Render fixed qualitative vectorization samples for the thesis tables."""

    _reset_output_dir(output_dir)

    validation_dir = output_dir / "validation"
    synthetic_dir = output_dir / "synthetic"
    synthetic_reference_dir = _find_synthetic_reference_dir()

    val_ref_written, val_ref_missing = _write_references(
        Path("raster/reference"),
        validation_dir,
        size=size,
    )
    syn_ref_written, syn_ref_missing = _write_references(
        synthetic_reference_dir,
        synthetic_dir,
        size=size,
    )
    val_method_written, val_method_missing = _write_methods(
        VALIDATION_METHODS,
        validation_dir,
        size=size,
    )
    syn_method_written, syn_method_missing = _write_methods(
        SYNTHETIC_METHODS,
        synthetic_dir,
        size=size,
    )

    print(f"validation_reference_written: {val_ref_written}")
    print(f"validation_reference_missing_or_failed: {val_ref_missing}")
    print(f"synthetic_reference_dir: {synthetic_reference_dir}")
    print(f"synthetic_reference_written: {syn_ref_written}")
    print(f"synthetic_reference_missing_or_failed: {syn_ref_missing}")
    print(f"validation_methods_written: {val_method_written}")
    print(f"validation_methods_missing_or_failed: {val_method_missing}")
    print(f"synthetic_methods_written: {syn_method_written}")
    print(f"synthetic_methods_missing_or_failed: {syn_method_missing}")


if __name__ == "__main__":
    app()
