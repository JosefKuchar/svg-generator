import os
import subprocess
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def has_raster_tool(command: str) -> bool:
    return _resolve_raster_tool(command) is not None


def _resolve_raster_tool(command: str) -> str | None:
    command_path = Path(command).expanduser()
    if command_path.parent != Path("."):
        if command_path.is_file() and os.access(command_path, os.X_OK):
            return str(command_path)
        return None

    for raw_path in os.environ.get("PATH", "").split(os.pathsep):
        if not raw_path:
            continue

        candidate = Path(raw_path).expanduser() / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    return None


def _require_raster_tool(command: str) -> str:
    resolved_command = _resolve_raster_tool(command)
    if resolved_command is not None:
        return resolved_command

    raise RuntimeError(
        f"Required raster tool '{command}' was not found on PATH. "
        f"Install '{command}' or skip features that depend on it."
    )


def _render_svg(
    svg_content: str | bytes,
    width: int | None = None,
    height: int | None = None,
    background: str | None = None,
) -> Image.Image:
    resvg = _require_raster_tool("resvg")

    if isinstance(svg_content, str):
        svg_content = svg_content.encode("utf-8")

    command = [resvg, "-"]
    if width is not None:
        command.extend(["--width", str(width)])
    if height is not None:
        command.extend(["--height", str(height)])
    if background is not None:
        command.extend(["--background", background])
    command.append("-c")

    result = subprocess.run(
        command,
        input=svg_content,
        capture_output=True,
        check=True,
    )

    image = Image.open(BytesIO(result.stdout))
    image.load()

    return image


def render_svg(
    svg_content: str | bytes,
    width: int | None = None,
    height: int | None = None,
) -> Image.Image:
    return _render_svg(svg_content, width=width, height=height)


def render_svg_bg(
    svg_content: str | bytes,
    width: int | None = None,
    height: int | None = None,
) -> Image.Image:
    return _render_svg(svg_content, width=width, height=height, background="white")


def vectorize_image(image_path: str | Path, output_svg_path: str | Path) -> None:
    vtracer = _require_raster_tool("vtracer")

    subprocess.run(
        [
            vtracer,
            "--input",
            str(image_path),
            "--output",
            str(output_svg_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def calculate_mse(img1: Image.Image, img2: Image.Image) -> float:
    # Convert to RGB if needed and ensure same size
    if img1.size != img2.size:
        raise ValueError(
            f"Images must have the same size. Got {img1.size} and {img2.size}"
        )

    # Convert to RGB mode to ensure consistent format
    img1_rgb = img1.convert("RGB")
    img2_rgb = img2.convert("RGB")

    # Convert to numpy arrays
    arr1 = np.array(img1_rgb, dtype=np.float64)
    arr2 = np.array(img2_rgb, dtype=np.float64)

    # Calculate MSE: mean((img1 - img2)^2)
    mse = np.mean((arr1 - arr2) ** 2)

    return float(mse)
