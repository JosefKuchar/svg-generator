import subprocess
from io import BytesIO
import numpy as np
from PIL import Image


def render_svg(svg_content: str | bytes) -> Image.Image:
    # Convert string to bytes if needed
    if isinstance(svg_content, str):
        svg_content = svg_content.encode("utf-8")

    result = subprocess.run(
        ["resvg", "-", "-c"],
        input=svg_content,
        capture_output=True,
        check=True,
    )

    # Load the PNG bytes directly into PIL Image
    image = Image.open(BytesIO(result.stdout))
    # Ensure the image is loaded into memory
    image.load()

    return image


def render_svg_bg(svg_content: str | bytes) -> Image.Image:
    # Convert string to bytes if needed
    if isinstance(svg_content, str):
        svg_content = svg_content.encode("utf-8")

    result = subprocess.run(
        ["resvg", "-", "-c", "--background", "white"],
        input=svg_content,
        capture_output=True,
        check=True,
    )

    # Load the PNG bytes directly into PIL Image
    image = Image.open(BytesIO(result.stdout))
    # Ensure the image is loaded into memory
    image.load()

    return image


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
