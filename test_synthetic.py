"""
Test script for the synthetic dataset generator.
Generates a random scene, renders it, and saves the output.
"""

import typer
from synthetic import generate_random_scene
from representation import shapes_to_tensor, tensor_to_shapes
from parsing import save_bezier_shapes_to_svg
from raster import render_svg_bg

app = typer.Typer()


@app.command()
def main(
    seed: int = typer.Option(42, help="Random seed"),
    canvas_size: int = typer.Option(256, help="Canvas width and height"),
    min_shapes: int = typer.Option(1, help="Minimum shapes per scene"),
    max_shapes: int = typer.Option(10, help="Maximum shapes per scene"),
    max_segments: int = typer.Option(256, help="Maximum bezier segments"),
    output_svg: str = typer.Option("synthetic_test.svg", help="Output SVG path"),
    output_png: str = typer.Option("synthetic_test.png", help="Output PNG path"),
):
    """Generate a synthetic scene and save as SVG + PNG."""

    print(f"Generating scene (seed={seed}, canvas={canvas_size}x{canvas_size})...")
    shapes = generate_random_scene(
        canvas_w=canvas_size,
        canvas_h=canvas_size,
        min_shapes=min_shapes,
        max_shapes=max_shapes,
        max_segments=max_segments,
        seed=seed,
    )

    total_segments = sum(len(p.curves) for s in shapes for p in s.paths)
    print(f"Generated {len(shapes)} shapes, {total_segments} bezier segments:")
    for i, s in enumerate(shapes):
        n_curves = sum(len(p.curves) for p in s.paths)
        print(f"  [{i}] color=#{s.color[0]:02x}{s.color[1]:02x}{s.color[2]:02x} "
              f"opacity={s.opacity:.2f} curves={n_curves}")

    # Verify tensor roundtrip
    tensor = shapes_to_tensor(shapes, canvas_size, canvas_size, max_segments=max_segments)
    shapes_rt = tensor_to_shapes(tensor, canvas_size, canvas_size)
    print(f"\nTensor roundtrip: {len(shapes)} -> {len(shapes_rt)} shapes")

    # Save SVG
    svg_content = save_bezier_shapes_to_svg(shapes, canvas_size, canvas_size)
    with open(output_svg, "w") as f:
        f.write(svg_content)
    print(f"Saved SVG: {output_svg}")

    # Render and save PNG
    image = render_svg_bg(svg_content).convert("RGB")
    image.save(output_png)
    print(f"Saved PNG: {output_png} ({image.size[0]}x{image.size[1]})")

    print("Done!")


if __name__ == "__main__":
    app()
