from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import typer
from tqdm.auto import tqdm

from parsing import save_bezier_shapes_to_svg
from raster import render_svg_bg
from synthetic import generate_random_scene

app = typer.Typer()


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path("exports/synthetic-samples"),
        help="Directory where numbered SVG and PNG files will be written.",
    ),
    num_samples: int = typer.Option(1000, help="Number of samples to export."),
    seed: int = typer.Option(
        9_223_372_036_854_775_000,
        help="Base seed. Sample i uses seed + i.",
    ),
    canvas_size: int = typer.Option(256, help="Canvas width and height."),
    png_size: int = typer.Option(
        1024,
        help="Rendered PNG width and height.",
    ),
    min_shapes: int = typer.Option(1, help="Minimum shapes per scene."),
    max_shapes: int = typer.Option(10, help="Maximum shapes per scene."),
    max_segments: int = typer.Option(256, help="Maximum Bezier segments per scene."),
):
    """Export reproducible synthetic generator samples as numbered SVG/PNG pairs."""

    if num_samples <= 0:
        raise typer.BadParameter("num_samples must be greater than 0")
    if canvas_size <= 0:
        raise typer.BadParameter("canvas_size must be greater than 0")
    if png_size <= 0:
        raise typer.BadParameter("png_size must be greater than 0")
    if min_shapes <= 0:
        raise typer.BadParameter("min_shapes must be greater than 0")
    if max_shapes < min_shapes:
        raise typer.BadParameter("max_shapes must be greater than or equal to min_shapes")
    if max_segments <= 0:
        raise typer.BadParameter("max_segments must be greater than 0")

    render_size = png_size
    pad_width = max(4, len(str(num_samples - 1)))
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in tqdm(range(num_samples), desc="Exporting synthetic samples"):
        sample_seed = seed + index
        stem = f"{index:0{pad_width}d}"

        shapes = generate_random_scene(
            canvas_w=canvas_size,
            canvas_h=canvas_size,
            min_shapes=min_shapes,
            max_shapes=max_shapes,
            max_segments=max_segments,
            seed=sample_seed,
        )
        svg_content = save_bezier_shapes_to_svg(shapes, canvas_size, canvas_size)

        (output_dir / f"{stem}.svg").write_text(svg_content, encoding="utf-8")

        image = render_svg_bg(
            svg_content,
            width=render_size,
            height=render_size,
        ).convert("RGB")
        image.save(output_dir / f"{stem}.png")

    typer.echo(
        f"Exported {num_samples} synthetic SVG/PNG pairs to {output_dir} "
        f"with base seed {seed}"
    )


if __name__ == "__main__":
    app()
