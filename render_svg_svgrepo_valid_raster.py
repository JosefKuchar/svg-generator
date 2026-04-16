from pathlib import Path

import typer
from datasets import load_dataset
from tqdm.auto import tqdm

from raster import render_svg_bg


app = typer.Typer(
    help="Rasterize mikronai/svg-svgrepo valid SVGs into numbered PNG files."
)


@app.command()
def main(
    output_dir: str = typer.Option(
        "outputs/svg-svgrepo-valid-raster", help="Directory for numbered PNG files"
    ),
    num_samples: int | None = typer.Option(
        None,
        min=1,
        help="Optional number of valid split samples to rasterize",
    ),
    width: int = typer.Option(1024, help="Rendered PNG width"),
    height: int = typer.Option(1024, help="Rendered PNG height"),
):
    if width <= 0 or height <= 0:
        raise typer.BadParameter("width and height must be greater than 0")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("mikronai/svg-svgrepo", split="valid")
    render_count = len(dataset) if num_samples is None else min(num_samples, len(dataset))
    pad_width = max(4, len(str(render_count - 1)))

    for idx in tqdm(range(render_count), desc="Rasterizing valid split"):
        item = dataset[idx]
        image = render_svg_bg(item["item_svg"], width=width, height=height).convert("RGB")
        image.save(out_path / f"{idx:0{pad_width}d}.png")


if __name__ == "__main__":
    app()
