import argparse
import json
import math

import numpy as np
import torch
from datasets import DatasetDict, get_dataset_split_names, load_dataset

from representation import BezierPath, BezierShape, shapes_to_tensor
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
        default=300,
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
    parser.add_argument(
        "--curve-abs-threshold",
        type=float,
        default=10.0,
        help=(
            "Drop rows whose max abs value in shapes_to_tensor output exceeds this. "
            "Set negative to disable this filter."
        ),
    )
    parser.add_argument(
        "--filter-num-proc",
        type=int,
        default=4,
        help="Number of processes for dataset filtering",
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
    try:
        shapes_data = json.loads(shapes_json)
    except Exception:
        return -1

    total_curves = 0
    try:
        for shape_data in shapes_data:
            for path_data in shape_data["paths"]:
                total_curves += len(path_data["curves"])
    except Exception:
        return -1

    return total_curves


def has_valid_size(item) -> bool:
    width = item.get("width")
    height = item.get("height")
    try:
        width = float(width)
        height = float(height)
    except Exception:
        return False

    if not math.isfinite(width) or not math.isfinite(height):
        return False
    return width > 0 and height > 0


def shapes_from_json(shapes_json: str):
    shapes_data = json.loads(shapes_json)
    bezier_shapes = []
    for shape_data in shapes_data:
        bezier_paths = []
        for path_data in shape_data["paths"]:
            curves = [
                tuple(tuple(point) for point in curve)
                for curve in path_data["curves"]
            ]
            bezier_paths.append(BezierPath(curves))

        bezier_shapes.append(
            BezierShape(
                paths=bezier_paths,
                color=tuple(shape_data["color"]) if shape_data["color"] else (0, 0, 0),
                opacity=shape_data["opacity"],
            )
        )
    return bezier_shapes


def has_valid_curve_tensor(item, max_segments: int, curve_abs_threshold: float | None) -> bool:
    try:
        width = float(item["width"])
        height = float(item["height"])
        shapes = shapes_from_json(item["shapes"])
        curve_tensor = shapes_to_tensor(
            shapes,
            width=width,
            height=height,
            max_segments=max_segments,
        )
    except Exception:
        return False

    if not torch.isfinite(curve_tensor).all().item():
        return False

    if curve_abs_threshold is not None:
        max_abs_curve = float(curve_tensor.abs().max().item())
        if max_abs_curve > curve_abs_threshold:
            return False

    return True


def is_valid_item(item, max_segments: int, curve_abs_threshold: float | None) -> bool:
    if not has_valid_size(item):
        return False
    curve_count = count_curves(item["shapes"])
    if curve_count < 0:
        return False
    if curve_count > max_segments:
        return False
    return has_valid_curve_tensor(item, max_segments, curve_abs_threshold)


def main():
    args = parse_args()
    target_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    curve_abs_threshold = (
        None if args.curve_abs_threshold is None or args.curve_abs_threshold < 0 else args.curve_abs_threshold
    )

    if args.filter_num_proc < 1:
        raise ValueError("--filter-num-proc must be >= 1")

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

        print(
            f"Filtering split: {split} to <= {args.max_segments} segments, "
            f"finite curve tensors, curve_abs_threshold={curve_abs_threshold}"
        )
        before_count = len(ds)
        ds = ds.filter(
            is_valid_item,
            fn_kwargs={
                "max_segments": args.max_segments,
                "curve_abs_threshold": curve_abs_threshold,
            },
            num_proc=args.filter_num_proc,
            desc=(
                "Filtering invalid rows, oversized curve counts, and unstable curve tensors"
            ),
        )
        after_count = len(ds)
        print(f"Kept {after_count}/{before_count} rows after validity filtering")

        if args.max_samples is not None:
            ds = ds.select(range(min(args.max_samples, len(ds))))

        if args.caption_column not in ds.column_names:
            raise ValueError(
                f"Column '{args.caption_column}' not found in split '{split}'. "
                f"Available columns: {ds.column_names}"
            )

        def embed_batch(batch, indices):
            prompts = [first_caption(captions) for captions in batch[args.caption_column]]
            embeddings, attention_mask = encode_prompts(
                prompts,
                tokenizer,
                model,
                device,
                max_length=args.max_length,
            )

            if not torch.isfinite(embeddings).all():
                raise ValueError(
                    f"Non-finite values in text embeddings for indices sample: {indices[:8]}"
                )

            if not torch.isfinite(attention_mask).all():
                raise ValueError(
                    f"Non-finite values in attention mask for indices sample: {indices[:8]}"
                )

            if embeddings.shape[:2] != attention_mask.shape:
                raise ValueError(
                    "Embedding/attention_mask shape mismatch for indices sample: "
                    f"{indices[:8]} | embeddings={tuple(embeddings.shape)} "
                    f"attention_mask={tuple(attention_mask.shape)}"
                )

            valid_mask = (attention_mask == 0) | (attention_mask == 1)
            if not valid_mask.all():
                raise ValueError(
                    f"Attention mask contains values outside {{0,1}} for indices sample: {indices[:8]}"
                )

            embeddings = embeddings.to(dtype=target_dtype).cpu().numpy()
            attention_mask = attention_mask.cpu().numpy()

            if not np.isfinite(embeddings).all():
                raise ValueError(
                    f"Non-finite values after dtype conversion for indices sample: {indices[:8]}"
                )

            return {
                "text_embeddings": embeddings,
                "text_attention_mask": attention_mask,
            }

        print(f"Encoding split: {split}")
        ds_with_embeddings = ds.map(
            embed_batch,
            batched=True,
            with_indices=True,
            batch_size=args.batch_size,
            desc=f"Precomputing text embeddings for {split}",
        )
        processed_splits[split] = ds_with_embeddings

    out = DatasetDict(processed_splits)
    out.save_to_disk(args.output)
    print(f"Saved dataset with text embeddings to: {args.output}")


if __name__ == "__main__":
    main()
