import argparse
import json

import torch
from datasets import DatasetDict, get_dataset_split_names, load_dataset

from text_encoder import encode_prompts, load_text_encoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute text embeddings for bezier dataset captions."
    )
    parser.add_argument(
        "--dataset",
        default="JosefKuchar/bezier-dataset",
        help="Hugging Face dataset id or local dataset path",
    )
    parser.add_argument(
        "--splits",
        default="all",
        help="Comma-separated list of splits, or 'all'",
    )
    parser.add_argument(
        "--caption-column",
        default="caption_texts",
        help="Column that contains multiple captions per item",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size used for text encoding",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Tokenizer max length",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Stored dtype for text_embeddings",
    )
    parser.add_argument(
        "--output",
        default="bezier_dataset_with_text_embeddings",
        help="Output directory for save_to_disk",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit per split (useful for quick tests)",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=256,
        help="Keep only items with this many or fewer Bezier segments",
    )
    return parser.parse_args()


def resolve_splits(dataset_name: str, splits_arg: str):
    if splits_arg == "all":
        return get_dataset_split_names(dataset_name)
    return [split.strip() for split in splits_arg.split(",") if split.strip()]


def first_caption(captions):
    if isinstance(captions, list) and captions:
        first = captions[0]
        return first if isinstance(first, str) else ""
    return ""


def count_curves(shapes_json: str) -> int:
    shapes_data = json.loads(shapes_json)
    total_curves = 0
    for shape_data in shapes_data:
        for path_data in shape_data["paths"]:
            total_curves += len(path_data["curves"])
    return total_curves


def main():
    args = parse_args()
    target_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    tokenizer, model, device = load_text_encoder()

    split_names = resolve_splits(args.dataset, args.splits)
    processed_splits = {}

    for split in split_names:
        print(f"Loading split: {split}")
        ds = load_dataset(args.dataset, split=split)

        if "shapes" not in ds.column_names:
            raise ValueError(
                f"Column 'shapes' not found in split '{split}'. "
                f"Available columns: {ds.column_names}"
            )

        print(f"Filtering split: {split} to <= {args.max_segments} segments")
        ds = ds.filter(
            lambda item: count_curves(item["shapes"]) <= args.max_segments,
            num_proc=4,
            desc=f"Filtering items with <= {args.max_segments} curves",
        )

        if args.max_samples is not None:
            ds = ds.select(range(min(args.max_samples, len(ds))))

        if args.caption_column not in ds.column_names:
            raise ValueError(
                f"Column '{args.caption_column}' not found in split '{split}'. "
                f"Available columns: {ds.column_names}"
            )

        def embed_batch(batch):
            prompts = [first_caption(captions) for captions in batch[args.caption_column]]
            embeddings, attention_mask = encode_prompts(
                prompts,
                tokenizer,
                model,
                device,
                max_length=args.max_length,
            )
            embeddings = embeddings.to(dtype=target_dtype).cpu().numpy()
            attention_mask = attention_mask.cpu().numpy()
            return {
                "text_embeddings": embeddings,
                "text_attention_mask": attention_mask,
            }

        print(f"Encoding split: {split}")
        ds_with_embeddings = ds.map(
            embed_batch,
            batched=True,
            batch_size=args.batch_size,
            desc=f"Precomputing text embeddings for {split}",
        )
        processed_splits[split] = ds_with_embeddings

    out = DatasetDict(processed_splits)
    out.save_to_disk(args.output)
    print(f"Saved dataset with text embeddings to: {args.output}")


if __name__ == "__main__":
    main()
