from pathlib import Path

import typer
from datasets import load_dataset
from tqdm.auto import tqdm

from raster import render_svg_bg

app = typer.Typer()


def _select_largest_prompt(item: dict) -> str:
    texts = item.get("caption_texts") or []
    if not texts:
        return item.get("item_title") or item.get("item_slug") or ""

    token_counts = item.get("caption_num_tokens") or []
    if len(token_counts) == len(texts):
        best_idx = max(range(len(texts)), key=lambda idx: token_counts[idx])
        return texts[best_idx]

    return max(texts, key=len)


@app.command()
def main(
    output_dir: str = typer.Option("exports/svg-svgrepo", help="Output directory"),
    num_samples: int = typer.Option(500, help="Number of samples to export"),
    width: int = typer.Option(1024, help="Rendered PNG width"),
    height: int = typer.Option(1024, help="Rendered PNG height"),
    start_index: int = typer.Option(0, help="Train split start index"),
    seed: int = typer.Option(42, help="Shuffle seed"),
    caption_prefix: str = typer.Option("", help="String prefixed to each caption"),
):
    """Export mikronai/svg-svgrepo train samples to numbered PNG/TXT pairs."""

    if num_samples <= 0:
        raise typer.BadParameter("num_samples must be greater than 0")
    if width <= 0 or height <= 0:
        raise typer.BadParameter("width and height must be greater than 0")
    if start_index < 0:
        raise typer.BadParameter("start_index must be 0 or greater")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("mikronai/svg-svgrepo", split="train").shuffle(seed=seed)
    available = len(dataset) - start_index
    if available <= 0:
        raise typer.BadParameter("start_index is past the end of the train split")

    export_count = min(num_samples, available)
    pad_width = max(4, len(str(export_count - 1)))

    for output_idx, item_idx in enumerate(
        tqdm(range(start_index, start_index + export_count), desc="Exporting samples")
    ):
        item = dataset[item_idx]
        stem = f"{output_idx:0{pad_width}d}"

        prompt = f"{caption_prefix}{_select_largest_prompt(item).strip()}"
        image = render_svg_bg(item["item_svg"], width=width, height=height).convert("RGB")

        image.save(out_path / f"{stem}.png")
        (out_path / f"{stem}.txt").write_text(f"{prompt}\n", encoding="utf-8")


if __name__ == "__main__":
    app()
