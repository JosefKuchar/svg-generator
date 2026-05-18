from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import typer
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from raster import has_raster_tool, render_svg_bg, vectorize_image


app = typer.Typer(
    help="Rasterize SVG files to PNG and vectorize the rasters back to SVG with vtracer."
)


def _svg_paths(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.svg" if recursive else "*.svg"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def _process_svg(
    svg_path: Path,
    input_dir: Path,
    output_dir: Path,
    size: int,
    overwrite: bool,
) -> Path:
    relative_path = svg_path.relative_to(input_dir)
    output_path = output_dir / relative_path

    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = render_svg_bg(svg_path.read_bytes(), width=size, height=size).convert("RGB")

    with TemporaryDirectory(prefix="svg-vtracer-") as temp_dir:
        raster_path = Path(temp_dir) / f"{svg_path.stem}.png"
        image.save(raster_path)
        vectorize_image(raster_path, output_path)

    return output_path


@app.command()
def main(
    input_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Folder containing SVG files.",
    ),
    output_dir: Path = typer.Argument(
        ...,
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Folder where vtracer SVG outputs will be written.",
    ),
    size: int = typer.Option(1024, min=1, help="Raster width and height in pixels."),
    recursive: bool = typer.Option(False, help="Process SVG files recursively."),
    overwrite: bool = typer.Option(False, help="Overwrite existing output SVG files."),
    workers: int = typer.Option(1, min=1, help="Number of SVG files to process in parallel."),
):
    if not has_raster_tool("resvg"):
        typer.echo("Required raster tool 'resvg' was not found on PATH.", err=True)
        raise typer.Exit(1)
    if not has_raster_tool("vtracer"):
        typer.echo("Required raster tool 'vtracer' was not found on PATH.", err=True)
        raise typer.Exit(1)

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _svg_paths(input_dir, recursive=recursive)
    if not paths:
        scope = "recursively " if recursive else ""
        typer.echo(f"No SVG files found {scope}in {input_dir}.", err=True)
        raise typer.Exit(1)

    process_kwargs = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "size": size,
        "overwrite": overwrite,
    }

    if workers == 1 or len(paths) == 1:
        for path in tqdm(paths, desc="Vectorizing SVGs"):
            _process_svg(path, **process_kwargs)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as executor:
            futures = [
                executor.submit(_process_svg, path, **process_kwargs) for path in paths
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Vectorizing SVGs"):
                future.result()

    typer.echo(f"Wrote {len(paths)} SVG files to {output_dir}")


if __name__ == "__main__":
    app()
