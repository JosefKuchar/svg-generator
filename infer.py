from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import typer
from diffusers import ZImagePipeline
from PIL import Image

from flow_matching_hf import (
    DEFAULT_SUBFOLDER,
    DINO_MODEL_NAME,
    load_dino_encoder,
    load_flow_matching_from_hub,
    resolve_device,
)
from parsing import save_bezier_shapes_to_svg
from raster import render_svg_bg, vectorize_image
from representation import tensor_to_shapes


app = typer.Typer(pretty_exceptions_show_locals=False)


class ZImageVariant(str, Enum):
    base = "base"
    turbo = "turbo"


class Vectorizer(str, Enum):
    vtracer = "vtracer"
    flow_matching = "flow-matching"


ZIMAGE_MODEL_IDS = {
    ZImageVariant.base: "Tongyi-MAI/Z-Image",
    ZImageVariant.turbo: "Tongyi-MAI/Z-Image-Turbo",
}

ZIMAGE_DEFAULTS = {
    ZImageVariant.base: {"steps": 50, "guidance_scale": 4.0},
    ZImageVariant.turbo: {"steps": 9, "guidance_scale": 0.0},
}


def _load_z_image_pipeline(
    variant: ZImageVariant,
    model_repo_id: str,
    lora_weight_name: str,
    lora_scale: float,
    device: torch.device,
) -> ZImagePipeline:
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    pipe = ZImagePipeline.from_pretrained(
        ZIMAGE_MODEL_IDS[variant],
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    pipe.load_lora_weights(
        model_repo_id,
        weight_name=lora_weight_name,
        adapter_name="svg",
    )
    pipe.set_adapters(["svg"], adapter_weights=[lora_scale])
    pipe.to(device)
    return pipe


def _generate_raster(
    pipe: ZImagePipeline,
    prompt: str,
    output_png: Path,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    seed: int | None,
    device: torch.device,
) -> Image.Image:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)
    return image


def _vectorize_with_vtracer(
    input_png: Path,
    output_svg: Path,
    vtracer_args: list[str] | None,
) -> None:
    if vtracer_args:
        import subprocess
        from raster import _require_raster_tool

        vtracer = _require_raster_tool("vtracer")
        subprocess.run(
            [
                vtracer,
                "--input",
                str(input_png),
                "--output",
                str(output_svg),
                *vtracer_args,
            ],
            check=True,
        )
        return

    vectorize_image(input_png, output_svg)


def _vectorize_with_flow_matching(
    image: Image.Image,
    output_svg: Path,
    model_repo_id: str,
    model_subfolder: str,
    steps: int,
    cfg_scale: float,
    seed: int | None,
    max_segments: int,
    svg_size: int | None,
    device: torch.device,
) -> None:
    vectorizer = load_flow_matching_from_hub(
        model_repo_id,
        subfolder=model_subfolder,
        device=device,
    )
    processor, image_encoder = load_dino_encoder(
        device=device,
        model_name=DINO_MODEL_NAME,
    )

    pixel_values = processor(images=[image.convert("RGB")], return_tensors="pt")[
        "pixel_values"
    ].to(device)

    with torch.inference_mode():
        cond = image_encoder(pixel_values=pixel_values).last_hidden_state
        samples = vectorizer.sample(
            cond.float(),
            steps=steps,
            cfg_scale=cfg_scale,
            shape=(1, max_segments, vectorizer.hparams.input_dim),
            seed=seed,
        )

    width, height = (svg_size, svg_size) if svg_size is not None else image.size
    shapes = tensor_to_shapes(samples[0].cpu(), width, height)
    svg_content = save_bezier_shapes_to_svg(shapes, width, height)
    output_svg.write_text(svg_content, encoding="utf-8")


@app.command()
def main(
    prompt: str = typer.Argument(..., help="Text prompt for SVG generation."),
    output_svg: Path = typer.Option(
        Path("output.svg"),
        "--output-svg",
        "-o",
        file_okay=True,
        dir_okay=False,
        help="Output SVG path.",
    ),
    output_png: Path | None = typer.Option(
        None,
        "--output-png",
        help="Optional path for the intermediate generated PNG.",
    ),
    preview_png: Path | None = typer.Option(
        None,
        "--preview-png",
        help="Optional rendered preview of the final SVG.",
    ),
    model_repo_id: str = typer.Option(
        "JosefKuchar/svg-generator",
        help="Hugging Face repo containing the SVG LoRA and vectorizer artifacts.",
    ),
    z_image_variant: ZImageVariant = typer.Option(
        ZImageVariant.base,
        "--z-image",
        help="Stage 1 Z-Image variant.",
    ),
    prompt_prefix: str = typer.Option(
        "SVG illustration with white background. ",
        help="Text prepended to the prompt before raster generation.",
    ),
    lora_weight_name: str = typer.Option(
        "zimage-svg-lora.safetensors",
        help="LoRA safetensors filename in the Hugging Face repo.",
    ),
    lora_scale: float = typer.Option(
        1.0,
        min=0.0,
        help="Diffusers adapter scale for the Z-Image SVG LoRA.",
    ),
    height: int = typer.Option(1024, min=64, help="Generated raster height."),
    width: int = typer.Option(1024, min=64, help="Generated raster width."),
    z_steps: int | None = typer.Option(
        None,
        min=1,
        help="Stage 1 diffusion steps. Defaults depend on --z-image.",
    ),
    z_guidance_scale: float | None = typer.Option(
        None,
        min=0.0,
        help="Stage 1 guidance scale. Defaults depend on --z-image.",
    ),
    seed: int | None = typer.Option(
        42,
        help="Random seed for stage 1 and, by default, flow-matching stage 2.",
    ),
    vectorizer: Vectorizer = typer.Option(
        Vectorizer.vtracer,
        help="Stage 2 raster-to-SVG vectorizer.",
    ),
    vtracer_arg: list[str] | None = typer.Option(
        None,
        "--vtracer-arg",
        help="Extra argument passed to vtracer. Repeat for multiple args.",
    ),
    flow_subfolder: str = typer.Option(
        DEFAULT_SUBFOLDER,
        help="Subfolder containing flow-matching config and weights.",
    ),
    flow_steps: int = typer.Option(
        50,
        min=2,
        help="Flow-matching RK4 sampling steps.",
    ),
    flow_cfg_scale: float = typer.Option(
        1.0,
        min=0.0,
        help="Flow-matching classifier-free guidance scale.",
    ),
    flow_seed: int | None = typer.Option(
        None,
        help="Optional separate flow-matching seed. Defaults to --seed.",
    ),
    max_segments: int = typer.Option(
        256,
        min=1,
        help="Maximum number of generated Bezier segments for flow matching.",
    ),
    svg_size: int | None = typer.Option(
        None,
        min=1,
        help="Optional square SVG canvas size for flow matching. Defaults to PNG size.",
    ),
    device: str = typer.Option(
        "auto",
        help="Torch device: auto, cuda, cuda:0, or cpu.",
    ),
):
    """Generate a raster image with Z-Image + LoRA, then convert it to SVG."""

    torch.set_float32_matmul_precision("medium")
    resolved_device = resolve_device(device)

    defaults = ZIMAGE_DEFAULTS[z_image_variant]
    resolved_z_steps = defaults["steps"] if z_steps is None else z_steps
    resolved_z_guidance = (
        defaults["guidance_scale"] if z_guidance_scale is None else z_guidance_scale
    )
    full_prompt = f"{prompt_prefix}{prompt.strip()}"

    output_svg.parent.mkdir(parents=True, exist_ok=True)

    pipe = _load_z_image_pipeline(
        variant=z_image_variant,
        model_repo_id=model_repo_id,
        lora_weight_name=lora_weight_name,
        lora_scale=lora_scale,
        device=resolved_device,
    )

    if output_png is not None:
        raster_path = output_png
        image = _generate_raster(
            pipe=pipe,
            prompt=full_prompt,
            output_png=raster_path,
            height=height,
            width=width,
            steps=resolved_z_steps,
            guidance_scale=resolved_z_guidance,
            seed=seed,
            device=resolved_device,
        )

        if vectorizer == Vectorizer.vtracer:
            _vectorize_with_vtracer(raster_path, output_svg, vtracer_arg)
        else:
            _vectorize_with_flow_matching(
                image=image,
                output_svg=output_svg,
                model_repo_id=model_repo_id,
                model_subfolder=flow_subfolder,
                steps=flow_steps,
                cfg_scale=flow_cfg_scale,
                seed=seed if flow_seed is None else flow_seed,
                max_segments=max_segments,
                svg_size=svg_size,
                device=resolved_device,
            )
    else:
        with TemporaryDirectory(prefix="svg-generator-infer-") as temp_dir:
            raster_path = Path(temp_dir) / "stage1.png"
            image = _generate_raster(
                pipe=pipe,
                prompt=full_prompt,
                output_png=raster_path,
                height=height,
                width=width,
                steps=resolved_z_steps,
                guidance_scale=resolved_z_guidance,
                seed=seed,
                device=resolved_device,
            )

            if vectorizer == Vectorizer.vtracer:
                _vectorize_with_vtracer(raster_path, output_svg, vtracer_arg)
            else:
                _vectorize_with_flow_matching(
                    image=image,
                    output_svg=output_svg,
                    model_repo_id=model_repo_id,
                    model_subfolder=flow_subfolder,
                    steps=flow_steps,
                    cfg_scale=flow_cfg_scale,
                    seed=seed if flow_seed is None else flow_seed,
                    max_segments=max_segments,
                    svg_size=svg_size,
                    device=resolved_device,
                )

    if preview_png is not None:
        preview_png.parent.mkdir(parents=True, exist_ok=True)
        svg_content = output_svg.read_bytes()
        render_svg_bg(svg_content, width=width, height=height).save(preview_png)

    typer.echo(f"Wrote SVG: {output_svg}")
    if output_png is not None:
        typer.echo(f"Wrote stage 1 PNG: {output_png}")
    if preview_png is not None:
        typer.echo(f"Wrote preview PNG: {preview_png}")


if __name__ == "__main__":
    app()
