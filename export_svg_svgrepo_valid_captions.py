from pathlib import Path

import typer
from datasets import load_dataset
from tqdm.auto import tqdm


app = typer.Typer(
    help="Export captions from the mikronai/svg-svgrepo valid split to a text file."
)


def _select_caption(item: dict, caption_index: int) -> str:
    texts = item.get("caption_texts") or []
    if not texts:
        fallback = item.get("item_title") or item.get("item_slug") or ""
        if fallback:
            return fallback
        raise typer.BadParameter("Sample has no captions and no fallback title/slug")

    if not 0 <= caption_index < len(texts):
        raise typer.BadParameter(
            f"caption_index must be between 0 and {len(texts) - 1} for this dataset"
        )

    return texts[caption_index]


@app.command()
def main(
    output_file: str = typer.Option(
        "outputs/svg-svgrepo-valid-captions.txt",
        help="Path to the output text file",
    ),
    caption_index: int = typer.Option(
        1, min=0, max=3, help="Caption index to export from caption_texts (0-3)"
    ),
    num_samples: int | None = typer.Option(
        None,
        min=1,
        help="Optional number of valid split samples to export",
    ),
):
    """Write one caption per valid split item, using caption_texts[caption_index]."""

    dataset = load_dataset("mikronai/svg-svgrepo", split="valid")
    export_count = len(dataset) if num_samples is None else min(num_samples, len(dataset))

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for idx in tqdm(range(export_count), desc="Exporting valid captions"):
            item = dataset[idx]
            caption = _select_caption(item, caption_index).strip()
            handle.write(f"{caption}\n")

    typer.echo(f"Wrote {export_count} captions to {out_path}")


if __name__ == "__main__":
    app()
