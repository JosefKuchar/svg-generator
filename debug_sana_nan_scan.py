#!/usr/bin/env python3
import argparse
import json
import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from representation import BezierPath, BezierShape, shapes_to_tensor
from sana_dataset import SanaBezierDataset, sana_bezier_collate_fn


@dataclass
class ScanStats:
    total_rows: int = 0
    bad_width_height: int = 0
    bad_shapes_json: int = 0
    bad_curve_tensor: int = 0
    bad_embeddings: int = 0
    bad_attention_mask: int = 0
    bad_embed_mask_len: int = 0
    attention_mask_not_binary: int = 0
    max_abs_curve_value: float = 0.0
    max_abs_embedding_value: float = 0.0
    bad_row_indices: list[int] = field(default_factory=list)

    def mark_bad(self, idx: int):
        if len(self.bad_row_indices) < 25:
            self.bad_row_indices.append(idx)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan Sana dataset for NaN/Inf and malformed rows."
    )
    parser.add_argument(
        "--dataset",
        default="bezier_dataset_with_text_embeddings",
        help="HF dataset id or local save_to_disk directory",
    )
    parser.add_argument("--split", default="train", help="Dataset split to scan")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of scanned rows",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Randomly sample this many rows (without replacement)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for row sampling",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=256,
        help="max_segments for shapes_to_tensor sanity check",
    )
    parser.add_argument(
        "--dataloader-batches",
        type=int,
        default=0,
        help="If > 0, also iterate this many dataloader batches and check finiteness",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for dataloader sanity check",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Num workers for dataloader sanity check",
    )
    return parser.parse_args()


def _load_split(dataset_name_or_path: str, split: str):
    try:
        return load_from_disk(dataset_name_or_path)[split]
    except Exception:
        return load_dataset(dataset_name_or_path, split=split)


def _shapes_from_json(shapes_data):
    bezier_shapes = []
    for shape_data in shapes_data:
        bezier_paths = []
        for path_data in shape_data["paths"]:
            curves = [tuple(tuple(point) for point in curve) for curve in path_data["curves"]]
            bezier_paths.append(BezierPath(curves))

        bezier_shape = BezierShape(
            paths=bezier_paths,
            color=tuple(shape_data["color"]) if shape_data["color"] else (0.0, 0.0, 0.0),
            opacity=shape_data["opacity"],
        )
        bezier_shapes.append(bezier_shape)
    return bezier_shapes


def _is_finite_scalar(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def run_row_scan(args):
    ds = _load_split(args.dataset, args.split)
    n = len(ds)

    indices = list(range(n))
    if args.sample_rows is not None:
        k = min(args.sample_rows, n)
        rnd = random.Random(args.seed)
        indices = rnd.sample(indices, k)

    if args.max_rows is not None:
        indices = indices[: args.max_rows]

    stats = ScanStats(total_rows=len(indices))

    pbar = tqdm(indices, desc=f"row scan [{args.split}]", unit="row")
    for idx in pbar:
        row = ds[idx]
        row_bad = False

        width = row.get("width")
        height = row.get("height")
        if not _is_finite_scalar(width) or not _is_finite_scalar(height):
            stats.bad_width_height += 1
            row_bad = True
        else:
            if float(width) <= 0 or float(height) <= 0:
                stats.bad_width_height += 1
                row_bad = True

        # text embeddings
        emb = np.asarray(row.get("text_embeddings", []))
        if emb.size == 0 or not np.isfinite(emb).all():
            stats.bad_embeddings += 1
            row_bad = True
        else:
            stats.max_abs_embedding_value = max(
                stats.max_abs_embedding_value, float(np.max(np.abs(emb)))
            )

        # attention mask
        mask = np.asarray(row.get("text_attention_mask", []))
        if mask.size == 0 or not np.isfinite(mask).all():
            stats.bad_attention_mask += 1
            row_bad = True
        else:
            mask_unique = np.unique(mask)
            if not np.all(np.isin(mask_unique, [0, 1])):
                stats.attention_mask_not_binary += 1
                row_bad = True

        if emb.ndim >= 1 and mask.ndim >= 1:
            if emb.shape[0] != mask.shape[0]:
                stats.bad_embed_mask_len += 1
                row_bad = True

        # shapes -> curve tensor
        try:
            shapes_data = json.loads(row["shapes"])
            shapes = _shapes_from_json(shapes_data)
            curve_tensor = shapes_to_tensor(
                shapes,
                width=float(width),
                height=float(height),
                max_segments=args.max_segments,
            )
            if not torch.isfinite(curve_tensor).all():
                stats.bad_curve_tensor += 1
                row_bad = True
            else:
                stats.max_abs_curve_value = max(
                    stats.max_abs_curve_value,
                    float(curve_tensor.abs().max().item()),
                )
        except Exception:
            stats.bad_shapes_json += 1
            stats.bad_curve_tensor += 1
            row_bad = True

        if row_bad:
            stats.mark_bad(idx)

        pbar.set_postfix(
            bad=(
                stats.bad_width_height
                + stats.bad_shapes_json
                + stats.bad_curve_tensor
                + stats.bad_embeddings
                + stats.bad_attention_mask
                + stats.bad_embed_mask_len
                + stats.attention_mask_not_binary
            )
        )

    return stats


def run_dataloader_scan(args):
    ds = SanaBezierDataset(
        split=args.split,
        max_segments=args.max_segments,
        dataset_name_or_path=args.dataset,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=sana_bezier_collate_fn,
    )

    bad_batches = 0
    checked = 0
    for batch in tqdm(loader, desc="dataloader scan", total=args.dataloader_batches):
        curves, embeds, masks, _captions = batch
        checks = [curves, embeds, masks]
        if not all(torch.isfinite(x).all().item() for x in checks):
            bad_batches += 1
        checked += 1
        if checked >= args.dataloader_batches:
            break

    return checked, bad_batches


def print_report(stats: ScanStats):
    print("\n=== Row Scan Report ===")
    print(f"rows_scanned: {stats.total_rows}")
    print(f"bad_width_height: {stats.bad_width_height}")
    print(f"bad_shapes_json: {stats.bad_shapes_json}")
    print(f"bad_curve_tensor: {stats.bad_curve_tensor}")
    print(f"bad_embeddings: {stats.bad_embeddings}")
    print(f"bad_attention_mask: {stats.bad_attention_mask}")
    print(f"bad_embed_mask_len: {stats.bad_embed_mask_len}")
    print(f"attention_mask_not_binary: {stats.attention_mask_not_binary}")
    print(f"max_abs_curve_value: {stats.max_abs_curve_value:.6f}")
    print(f"max_abs_embedding_value: {stats.max_abs_embedding_value:.6f}")
    print(f"example_bad_indices (up to 25): {stats.bad_row_indices}")


def main():
    args = parse_args()
    stats = run_row_scan(args)
    print_report(stats)

    if args.dataloader_batches > 0:
        checked, bad_batches = run_dataloader_scan(args)
        print("\n=== Dataloader Scan Report ===")
        print(f"batches_checked: {checked}")
        print(f"bad_batches: {bad_batches}")


if __name__ == "__main__":
    main()
