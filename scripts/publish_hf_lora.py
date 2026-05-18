from pathlib import Path

import typer
from huggingface_hub import HfApi


app = typer.Typer()


@app.command()
def main(
    repo_id: str = typer.Option(
        "JosefKuchar/svg-generator",
        help="Target Hugging Face model repository.",
    ),
    lora_path: Path = typer.Option(
        Path("zimage-svg-lora.safetensors"),
        exists=True,
        dir_okay=False,
        help="Local LoRA safetensors checkpoint.",
    ),
    readme_path: Path = typer.Option(
        Path("huggingface/README.md"),
        exists=True,
        dir_okay=False,
        help="Local model card to publish as README.md.",
    ),
) -> None:
    api = HfApi()

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add model card for SVG generator LoRA",
    )
    typer.echo("Uploaded README.md")

    api.upload_file(
        path_or_fileobj=lora_path,
        path_in_repo="zimage-svg-lora.safetensors",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add Z-Image SVG LoRA weights",
    )
    typer.echo("Uploaded zimage-svg-lora.safetensors")


if __name__ == "__main__":
    app()
