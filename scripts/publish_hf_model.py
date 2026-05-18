from pathlib import Path

import typer
from huggingface_hub import HfApi


app = typer.Typer(pretty_exceptions_show_locals=False)

FILES = {
    "huggingface/README.md": "README.md",
    "zimage-svg-lora.safetensors": "zimage-svg-lora.safetensors",
    "huggingface/flow-matching/config.json": "flow-matching/config.json",
    "huggingface/flow-matching/model.safetensors": "flow-matching/model.safetensors",
}


@app.command()
def main(
    repo_id: str = typer.Option(
        "JosefKuchar/svg-generator",
        help="Target Hugging Face model repository.",
    ),
) -> None:
    api = HfApi()

    for local_path, repo_path in FILES.items():
        path = Path(local_path)
        if not path.exists():
            raise typer.BadParameter(f"Missing required file: {path}")

        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {repo_path}",
        )
        typer.echo(f"Uploaded {repo_path}")


if __name__ == "__main__":
    app()
