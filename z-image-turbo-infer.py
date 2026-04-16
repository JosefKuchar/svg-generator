import os

import torch
import typer
from diffusers import ZImagePipeline


app = typer.Typer()


def _load_lora(pipe: ZImagePipeline, lora_path: str, weight_name: str | None) -> None:
    if os.path.isfile(lora_path):
        pipe.load_lora_weights(
            os.path.dirname(lora_path) or ".",
            weight_name=weight_name or os.path.basename(lora_path),
        )
        return

    pipe.load_lora_weights(lora_path, weight_name=weight_name)


@app.command()
def main(
    prompt: str = typer.Option(
        "SVG illustration with white background. A black silhouette of a person paddling a kayak, with simple waves and a paddle mid-motion, capturing the essence of water sports and recreation.",
        help="Text prompt for image generation",
    ),
    output: str = typer.Option("example.png", help="Output image path"),
    height: int = typer.Option(1024, help="Output image height"),
    width: int = typer.Option(1024, help="Output image width"),
    num_inference_steps: int = typer.Option(9, help="Number of inference steps"),
    guidance_scale: float = typer.Option(
        0.0, help="Classifier-free guidance scale for generation"
    ),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    model_id: str = typer.Option(
        "Tongyi-MAI/Z-Image-Turbo", help="Base Z-Image model identifier"
    ),
    lora_path: str | None = typer.Option(
        None, help="Optional LoRA directory, repo id, or weight file"
    ),
    lora_weight_name: str | None = typer.Option(
        None, help="Optional explicit LoRA weight filename"
    ),
    lora_scale: float = typer.Option(1.0, help="LoRA scaling factor"),
):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = ZImagePipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )

    if lora_path is not None:
        print(f"Loading LoRA weights from: {lora_path}")
        _load_lora(pipe, lora_path, lora_weight_name)
        pipe.set_adapters(["default_0"], adapter_weights=[lora_scale])

    pipe.to(device)

    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device).manual_seed(seed),
    ).images[0]

    image.save(output)
    print(f"Saved image to: {output}")


if __name__ == "__main__":
    app()
