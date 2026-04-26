"""
Temporary exporter for thesis figures from the synthetic generator.

Generates PNG rasters under text/assets/syntetic_generator.
"""

from pathlib import Path

import typer

from parsing import save_bezier_shapes_to_svg
from raster import render_svg_bg
from synthetic import generate_random_scene

app = typer.Typer()


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path("text/assets/syntetic_generator"),
        help="Directory where PNG rasters will be written.",
    ),
    count: int = typer.Option(16, help="Number of samples to generate."),
    seed: int = typer.Option(42, help="Base random seed."),
    canvas_size: int = typer.Option(256, help="Canvas width and height."),
    min_shapes: int = typer.Option(1, help="Minimum shapes per scene."),
    max_shapes: int = typer.Option(10, help="Maximum shapes per scene."),
    max_segments: int = typer.Option(256, help="Maximum Bezier segments per scene."),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        sample_seed = seed + index
        shapes = generate_random_scene(
            canvas_w=canvas_size,
            canvas_h=canvas_size,
            min_shapes=min_shapes,
            max_shapes=max_shapes,
            max_segments=max_segments,
            seed=sample_seed,
        )
        svg_content = save_bezier_shapes_to_svg(shapes, canvas_size, canvas_size)
        image = render_svg_bg(
            svg_content,
            width=canvas_size,
            height=canvas_size,
        ).convert("RGB")

        output_path = output_dir / f"synthetic_generator_{index + 1:02d}.png"
        image.save(output_path)
        typer.echo(f"Saved {output_path}")


if __name__ == "__main__":
    app()
