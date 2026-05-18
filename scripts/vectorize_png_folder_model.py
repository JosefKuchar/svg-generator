from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import typer
from PIL import Image
from tqdm.auto import tqdm

import model as model_module
from flow_matching_hf import (
    CONFIG_NAME,
    DEFAULT_SUBFOLDER,
    DINO_MODEL_NAME,
    WEIGHTS_NAME,
    load_dino_encoder,
    load_flow_matching_from_files,
    load_flow_matching_from_hub,
)
from model import FlowMatchingTransformer
from parsing import save_bezier_shapes_to_svg
from raster import render_svg_bg
from representation import tensor_to_shapes


app = typer.Typer(pretty_exceptions_show_locals=False)

PNG_SUFFIXES = {".png"}


def _png_paths(folder: Path, recursive: bool) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in PNG_SUFFIXES
    )


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _disable_unsupported_flash_attention(device: torch.device) -> None:
    if device.type != "cuda" or not model_module.FLASH_ATTN_AVAILABLE:
        return

    major, _ = torch.cuda.get_device_capability(device)
    if major < 8:
        model_module.FLASH_ATTN_AVAILABLE = False
        model_module.flash_attn_func = None
        print("FlashAttention disabled: selected CUDA device is older than Ampere.")


def _output_path(input_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative = input_path.relative_to(input_dir)
    return output_dir / relative.with_suffix(".svg")


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_module(
    checkpoint: Path | None,
    model_repo_id: str | None,
    model_subfolder: str,
    device: torch.device,
) -> FlowMatchingTransformer:
    if checkpoint is not None and model_repo_id is not None:
        raise typer.BadParameter("Use either --checkpoint or --model-repo-id, not both")

    if model_repo_id is not None:
        print(f"Loading flow-matching model from Hub: {model_repo_id}/{model_subfolder}")
        return load_flow_matching_from_hub(
            repo_id=model_repo_id,
            subfolder=model_subfolder,
            device=device,
        )

    if checkpoint is None:
        raise typer.BadParameter("Either --checkpoint or --model-repo-id is required")

    print(f"Loading checkpoint: {checkpoint}")
    if checkpoint.suffix == ".safetensors":
        config_path = checkpoint.with_name(CONFIG_NAME)
        if not config_path.exists():
            raise typer.BadParameter(
                f"Missing config for safetensors checkpoint: {config_path}"
            )
        return load_flow_matching_from_files(checkpoint, config_path, device=device)

    module = FlowMatchingTransformer.load_from_checkpoint(
        str(checkpoint),
        map_location=device,
    )
    module.to(device)
    module.eval()
    return module


def _batched(items: list[Path], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@app.command()
def main(
    png_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    output_dir: Path = typer.Argument(..., file_okay=False, dir_okay=True),
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        "-c",
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=(
            "Local PyTorch Lightning .ckpt or exported "
            f"{WEIGHTS_NAME}. Required unless --model-repo-id is set."
        ),
    ),
    model_repo_id: str | None = typer.Option(
        None,
        help="Hugging Face model repo containing the exported flow-matching files.",
    ),
    model_subfolder: str = typer.Option(
        DEFAULT_SUBFOLDER,
        help="Subfolder in the Hugging Face repo containing config and weights.",
    ),
    batch_size: int = typer.Option(
        8,
        min=1,
        help="Number of PNG images to encode and sample at once.",
    ),
    steps: int = typer.Option(
        50,
        min=2,
        help="Number of RK4 sampling time steps.",
    ),
    cfg_scale: float = typer.Option(
        1.0,
        min=0.0,
        help="Classifier-free guidance scale. 1.0 uses conditional sampling only.",
    ),
    seed: int | None = typer.Option(
        42,
        help="Base random seed. Each batch uses seed + batch_index. Use none for random sampling.",
    ),
    max_segments: int = typer.Option(
        256,
        min=1,
        help="Maximum generated Bezier segments per SVG.",
    ),
    svg_size: int | None = typer.Option(
        None,
        min=1,
        help="Optional square SVG canvas size. Defaults to each input PNG size.",
    ),
    device: str = typer.Option(
        "auto",
        help="Torch device for inference, for example auto, cuda, cuda:0, or cpu.",
    ),
    recursive: bool = typer.Option(
        False,
        help="Process PNG files in subdirectories and mirror the folder structure.",
    ),
    overwrite: bool = typer.Option(
        False,
        help="Overwrite existing SVG outputs.",
    ),
    preview_png_dir: Path | None = typer.Option(
        None,
        help="Optional folder for rendered PNG previews of generated SVGs.",
    ),
):
    """Vectorize a folder of PNG images with the trained flow-matching model."""

    torch.set_float32_matmul_precision("medium")
    resolved_device = _resolve_device(device)
    _disable_unsupported_flash_attention(resolved_device)

    png_paths = _png_paths(png_dir, recursive=recursive)
    if not png_paths:
        raise typer.BadParameter(f"No PNG files found in folder: {png_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if preview_png_dir is not None:
        preview_png_dir.mkdir(parents=True, exist_ok=True)

    module = _load_module(
        checkpoint=checkpoint,
        model_repo_id=model_repo_id,
        model_subfolder=model_subfolder,
        device=resolved_device,
    )

    if module.image_encoder is None:
        processor, image_encoder = load_dino_encoder(
            device=resolved_device,
            model_name=DINO_MODEL_NAME,
        )
    else:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
        image_encoder = module.image_encoder

    written = 0
    skipped = 0

    with torch.inference_mode():
        for batch_index, batch_paths in enumerate(
            tqdm(list(_batched(png_paths, batch_size)), desc="Vectorizing PNGs")
        ):
            images = [_load_rgb_image(path) for path in batch_paths]
            pixel_values = processor(images=images, return_tensors="pt")[
                "pixel_values"
            ].to(resolved_device)

            cond = image_encoder(pixel_values=pixel_values).last_hidden_state
            samples = module.sample(
                cond.float(),
                steps=steps,
                cfg_scale=cfg_scale,
                shape=(len(batch_paths), max_segments, module.hparams.input_dim),
                seed=None if seed is None else seed + batch_index,
            )

            for path, image, sample in zip(batch_paths, images, samples):
                svg_path = _output_path(path, png_dir, output_dir)
                if svg_path.exists() and not overwrite:
                    skipped += 1
                    continue

                width, height = (svg_size, svg_size) if svg_size is not None else image.size
                shapes = tensor_to_shapes(sample.cpu(), width, height)
                svg_content = save_bezier_shapes_to_svg(shapes, width, height)

                svg_path.parent.mkdir(parents=True, exist_ok=True)
                svg_path.write_text(svg_content, encoding="utf-8")
                written += 1

                if preview_png_dir is not None:
                    preview_path = _output_path(path, png_dir, preview_png_dir).with_suffix(
                        ".png"
                    )
                    preview_path.parent.mkdir(parents=True, exist_ok=True)
                    render_svg_bg(svg_content, width=width, height=height).save(
                        preview_path
                    )

    print(f"images: {len(png_paths)}")
    print(f"written_svg: {written}")
    print(f"skipped_existing_svg: {skipped}")
    print(f"output_dir: {output_dir}")
    if preview_png_dir is not None:
        print(f"preview_png_dir: {preview_png_dir}")


if __name__ == "__main__":
    app()
