"""Inference script for sampling from the FlowMatchingTransformer model."""

import typer
import torch
import glob
import os
from model import FlowMatchingTransformer
from dataset import BezierDataset
from representation import tensor_to_shapes
from parsing import save_bezier_shapes_to_svg
from raster import render_svg

app = typer.Typer()


@app.command()
def main(
    steps: int = typer.Option(50, help="Number of sampling steps"),
    cfg_scale: float = typer.Option(1.0, help="Classifier-free guidance scale"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    output_svg: str = typer.Option("output.svg", help="Output SVG file path"),
    output_png: str = typer.Option("output.png", help="Output PNG file path"),
    original_png: str = typer.Option(
        "original.png", help="Original conditioning image path"
    ),
    img_size: int = typer.Option(512, help="Output image size"),
    max_segments: int = typer.Option(256, help="Maximum number of segments"),
):
    """Sample from the FlowMatchingTransformer model using the first training image as conditioning."""

    torch.set_float32_matmul_precision("medium")

    # Find the latest checkpoint
    ckpt_files = glob.glob("./svg-generator/osp09uef/checkpoints/*.ckpt")
    if not ckpt_files:
        print("No checkpoint files found!")
        raise typer.Exit(1)

    ckpt_files.sort(key=os.path.getmtime)
    latest_ckpt = ckpt_files[-1]
    print(f"Loading checkpoint: {latest_ckpt}")

    # Load model from checkpoint
    module = FlowMatchingTransformer.load_from_checkpoint(latest_ckpt)
    module.eval()
    device = next(module.parameters()).device
    print(f"Model loaded on device: {device}")

    # Load the first image from the training dataset
    print("Loading training dataset...")
    dataset = BezierDataset(split="train", max_segments=max_segments)
    curve_tensor, image_tensor = dataset[2]
    print(
        f"Loaded first training image. Curve tensor shape: {curve_tensor.shape}, Image tensor shape: {image_tensor.shape}"
    )

    # Move image tensor to device and convert to patch conditioning.
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]

    print("Building 16x16 patch conditioning...")
    cond = module.patchify_images(image_tensor)
    print(f"Conditioning shape: {cond.shape}")

    # Sample from the model
    print(f"Sampling with {steps} steps, CFG scale {cfg_scale}, seed {seed}...")
    samples = module.sample(
        cond,
        steps=steps,
        cfg_scale=cfg_scale,
        shape=(1, max_segments, module.hparams.input_dim),
        seed=seed,
    )
    print(f"Generated samples shape: {samples.shape}")

    # Convert tensor to shapes
    sample_tensor = samples[0].cpu()
    shapes = tensor_to_shapes(sample_tensor, img_size, img_size)
    print(f"Converted to {len(shapes)} shapes")

    # Save SVG
    svg_content = save_bezier_shapes_to_svg(shapes, img_size, img_size)
    with open(output_svg, "w") as f:
        f.write(svg_content)
    print(f"Saved SVG to: {output_svg}")

    # Render and save PNG
    image = render_svg(svg_content)
    image.save(output_png)
    print(f"Saved PNG to: {output_png}")

    # Save the original conditioning image.
    original_img = image_tensor[0].cpu()
    original_img = ((original_img + 1.0) * 0.5).clamp(0, 1)
    original_img = (original_img * 255).to(torch.uint8)
    original_img = original_img.permute(1, 2, 0).numpy()
    from PIL import Image

    Image.fromarray(original_img).save(original_png)
    print(f"Saved original conditioning image to: {original_png}")

    print("Done!")


if __name__ == "__main__":
    app()
