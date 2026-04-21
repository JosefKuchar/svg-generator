import os
from pathlib import Path

import torch
import typer
from datasets import load_dataset
from diffusers import ZImagePipeline
from tqdm.auto import tqdm


app = typer.Typer(
    help=(
        "Render mikronai/svg-svgrepo valid captions into numbered PNG/TXT pairs.\n\n"
        "Example with a local safetensors LoRA:\n"
        "uv run python render_svg_svgrepo_valid.py --output-dir outputs/svg-svgrepo-valid "
        "--caption-index 2 --num-samples 100 --prompt-prefix 'SVG illustration. ' "
        "--lora-path ./lora/my-style.safetensors --lora-scale 0.8"
    )
)


def _load_lora(pipe: ZImagePipeline, lora_path: str, weight_name: str | None) -> None:
    if os.path.isfile(lora_path):
        pipe.load_lora_weights(
            os.path.dirname(lora_path) or ".",
            weight_name=weight_name or os.path.basename(lora_path),
        )
        return

    pipe.load_lora_weights(lora_path, weight_name=weight_name)


def _select_prompt(item: dict, caption_index: int) -> str:
    texts = item.get("caption_texts") or []
    if not texts:
        fallback = item.get("item_title") or item.get("item_slug") or ""
        if fallback:
            return fallback
        raise typer.BadParameter("Sample has no captions and no fallback title/slug")

    if not 0 <= caption_index < len(texts):
        raise typer.BadParameter(
            f"caption_index must be between 0 and {len(texts) - 1} for this dataset"
        )

    return texts[caption_index]


@app.command()
def main(
    output_dir: str = typer.Option(
        "outputs/svg-svgrepo-valid", help="Directory for numbered PNG/TXT pairs"
    ),
    caption_index: int = typer.Option(
        0, min=0, max=3, help="Caption index to use from caption_texts (0-3)"
    ),
    num_samples: int | None = typer.Option(
        None,
        min=1,
        help="Optional number of valid split samples to render",
    ),
    prompt_prefix: str = typer.Option(
        "", help="Arbitrary string prefixed to every selected caption"
    ),
    batch_size: int = typer.Option(1, min=1, help="Number of prompts to render at once"),
    height: int = typer.Option(1024, help="Output image height"),
    width: int = typer.Option(1024, help="Output image width"),
    num_inference_steps: int = typer.Option(8, help="Number of inference steps"),
    guidance_scale: float = typer.Option(0.0, help="Classifier-free guidance scale"),
    seed: int = typer.Option(42, help="Base random seed"),
    model_id: str = typer.Option(
        "Tongyi-MAI/Z-Image-Turbo", help="Base Z-Image model identifier"
    ),
    lora_path: str | None = typer.Option(
        None,
        help="Optional LoRA directory, repo id, or safetensors file",
    ),
    lora_weight_name: str | None = typer.Option(
        None,
        help="Optional explicit LoRA weight filename",
    ),
    lora_scale: float = typer.Option(1.0, help="LoRA scaling factor"),
):
    if width <= 0 or height <= 0:
        raise typer.BadParameter("width and height must be greater than 0")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("mikronai/svg-svgrepo", split="valid")
    render_count = len(dataset) if num_samples is None else min(num_samples, len(dataset))
    pad_width = max(4, len(str(render_count - 1)))

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = ZImagePipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )

    if lora_path is not None:
        typer.echo(f"Loading LoRA weights from: {lora_path}")
        _load_lora(pipe, lora_path, lora_weight_name)
        pipe.set_adapters(["default_0"], adapter_weights=[lora_scale])

    pipe.to(device)

    for batch_start in tqdm(
        range(0, render_count, batch_size), desc="Rendering valid split"
    ):
        batch_end = min(batch_start + batch_size, render_count)
        prompts: list[str] = []
        stems: list[str] = []
        generators: list[torch.Generator] = []

        for idx in range(batch_start, batch_end):
            item = dataset[idx]
            prompts.append(f"{prompt_prefix}{_select_prompt(item, caption_index).strip()}")
            stems.append(f"{idx:0{pad_width}d}")
            generators.append(torch.Generator(device=device).manual_seed(seed + idx))

        with torch.inference_mode():
            images = pipe(
                prompt=prompts,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generators,
            ).images

        for stem, prompt, image in zip(stems, prompts, images, strict=True):
            image.save(out_path / f"{stem}.png")
            (out_path / f"{stem}.txt").write_text(f"{prompt}\n", encoding="utf-8")


if __name__ == "__main__":
    app()
