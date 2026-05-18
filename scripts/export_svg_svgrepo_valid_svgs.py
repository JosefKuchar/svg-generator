from pathlib import Path

import typer
from datasets import load_dataset
from tqdm.auto import tqdm


app = typer.Typer(help="Export mikronai/svg-svgrepo valid SVGs into numbered SVG files.")


@app.command()
def main(
    output_dir: str = typer.Option(
        "outputs/svg-svgrepo-valid-svgs", help="Directory for numbered SVG files"
    ),
    num_samples: int | None = typer.Option(
        None,
        min=1,
        help="Optional number of valid split samples to export",
    ),
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("mikronai/svg-svgrepo", split="valid")
    export_count = len(dataset) if num_samples is None else min(num_samples, len(dataset))
    pad_width = max(4, len(str(export_count - 1)))

    for idx in tqdm(range(export_count), desc="Exporting valid SVGs"):
        item = dataset[idx]
        (out_path / f"{idx:0{pad_width}d}.svg").write_text(
            item["item_svg"], encoding="utf-8"
        )

    typer.echo(f"Wrote {export_count} SVGs to {out_path}")


if __name__ == "__main__":
    app()
