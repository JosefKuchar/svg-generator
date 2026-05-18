import json
import sys
from pathlib import Path

import torch
import typer
from safetensors.torch import save_file

sys.path.append(str(Path(__file__).resolve().parents[1]))

from hub import CONFIG_NAME, DEFAULT_SUBFOLDER, WEIGHTS_NAME


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    checkpoint: Path = typer.Option(
        Path("svg-generator/bgno3qml/checkpoints/epoch0627.ckpt"),
        "--checkpoint",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Source PyTorch Lightning checkpoint.",
    ),
    output_dir: Path = typer.Option(
        Path("huggingface") / DEFAULT_SUBFOLDER,
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for exported Hugging Face files.",
    ),
) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = {
        key: value.contiguous()
        for key, value in ckpt["state_dict"].items()
        if not key.startswith("image_encoder.")
    }

    hparams = dict(ckpt["hyper_parameters"])
    hparams.pop("load_image_encoder", None)
    hparams.pop("image_encoder_model", None)

    config = {
        "architectures": ["FlowMatchingTransformer"],
        "model_type": "svg-flow-matching-vectorizer",
        "model_config": hparams,
        "conditioning_model": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "format": "safetensors",
        "source_checkpoint": str(checkpoint),
        "source_epoch": ckpt.get("epoch"),
        "source_global_step": ckpt.get("global_step"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / WEIGHTS_NAME
    config_path = output_dir / CONFIG_NAME

    save_file(
        state_dict,
        weights_path,
        metadata={
            "format": "pt",
            "source_checkpoint": str(checkpoint),
            "conditioning_model": config["conditioning_model"],
        },
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    source_size = checkpoint.stat().st_size / 1024 / 1024
    weights_size = weights_path.stat().st_size / 1024 / 1024
    typer.echo(f"Wrote {config_path}")
    typer.echo(f"Wrote {weights_path}")
    typer.echo(f"Source checkpoint: {source_size:.1f} MiB")
    typer.echo(f"Exported weights: {weights_size:.1f} MiB")
    typer.echo(f"Removed tensors: {len(ckpt['state_dict']) - len(state_dict)}")


if __name__ == "__main__":
    app()
