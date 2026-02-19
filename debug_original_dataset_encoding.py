#!/usr/bin/env python3
import argparse
import json
import math
import random
from dataclasses import dataclass, field

import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from representation import BezierPath, BezierShape, shapes_to_tensor


@dataclass
class ScanStats:
    total_rows: int = 0
    bad_width_height: int = 0
    bad_shapes_json: int = 0
    bad_encoding: int = 0
    bad_tensor_shape: int = 0
    bad_tensor_finite: int = 0
    bad_real_count: int = 0
    bad_padding: int = 0
    truncated_rows: int = 0
    out_of_range_values: int = 0
    max_abs_encoded_value: float = 0.0
    total_curves_raw: int = 0
    total_curves_encoded: int = 0
    bad_row_indices: list[int] = field(default_factory=list)

    def mark_bad(self, idx: int):
        if len(self.bad_row_indices) < 25:
            self.bad_row_indices.append(idx)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan the original bezier dataset and verify that shapes can be encoded "
            "with shapes_to_tensor."
        )
    )
    parser.add_argument(
        "--dataset",
        default="JosefKuchar/bezier-dataset",
        help="HF dataset id or local save_to_disk directory",
    )
    parser.add_argument("--split", default="train", help="Dataset split to scan")
    parser.add_argument(
        "--max-segments",
        type=int,
        default=256,
        help="max_segments used for encoding",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows to scan",
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
        "--range-eps",
        type=float,
        default=1e-5,
        help="Tolerance for [-1, 1] normalized range checks",
    )
    return parser.parse_args()


def _load_split(dataset_name_or_path: str, split: str):
    try:
        return load_from_disk(dataset_name_or_path)[split]
    except Exception:
        return load_dataset(dataset_name_or_path, split=split)


def _shapes_from_json(shapes_data):
    bezier_shapes = []
    total_curves = 0

    for shape_data in shapes_data:
        bezier_paths = []
        for path_data in shape_data["paths"]:
            curves = [tuple(tuple(point) for point in curve) for curve in path_data["curves"]]
            total_curves += len(curves)
            bezier_paths.append(BezierPath(curves))

        color = tuple(shape_data["color"]) if shape_data["color"] else (0.0, 0.0, 0.0)
        opacity = float(shape_data["opacity"])
        bezier_shapes.append(BezierShape(paths=bezier_paths, color=color, opacity=opacity))

    return bezier_shapes, total_curves


def _is_finite_positive_scalar(x) -> bool:
    try:
        value = float(x)
        return math.isfinite(value) and value > 0
    except Exception:
        return False


def run_scan(args):
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

    for idx in tqdm(indices, desc=f"encoding scan [{args.split}]", unit="row"):
        row = ds[idx]
        row_bad = False

        width = row.get("width")
        height = row.get("height")
        if not _is_finite_positive_scalar(width) or not _is_finite_positive_scalar(height):
            stats.bad_width_height += 1
            stats.bad_encoding += 1
            stats.mark_bad(idx)
            continue

        try:
            shapes_data = json.loads(row["shapes"])
            shapes, raw_curve_count = _shapes_from_json(shapes_data)
        except Exception:
            stats.bad_shapes_json += 1
            stats.bad_encoding += 1
            stats.mark_bad(idx)
            continue

        stats.total_curves_raw += raw_curve_count
        expected_encoded_curves = min(raw_curve_count, args.max_segments)
        stats.total_curves_encoded += expected_encoded_curves
        if raw_curve_count > args.max_segments:
            stats.truncated_rows += 1

        try:
            encoded = shapes_to_tensor(
                shapes,
                width=float(width),
                height=float(height),
                max_segments=args.max_segments,
            )
        except Exception:
            stats.bad_encoding += 1
            stats.mark_bad(idx)
            continue

        if encoded.shape != (args.max_segments, 13):
            stats.bad_tensor_shape += 1
            row_bad = True

        if not torch.isfinite(encoded).all().item():
            stats.bad_tensor_finite += 1
            row_bad = True
        else:
            stats.max_abs_encoded_value = max(
                stats.max_abs_encoded_value,
                float(encoded.abs().max().item()),
            )

        real_mask = encoded[:, 12] > 0
        real_count = int(real_mask.sum().item())
        if real_count != expected_encoded_curves:
            stats.bad_real_count += 1
            row_bad = True

        if expected_encoded_curves < args.max_segments:
            padded = encoded[expected_encoded_curves:]
            if padded.numel() > 0:
                padding_real_ok = torch.all(padded[:, 12] < 0).item()
                padding_zero_ok = torch.all(padded[:, :12] == 0).item()
                if not (padding_real_ok and padding_zero_ok):
                    stats.bad_padding += 1
                    row_bad = True

        values_to_check = encoded[:, :12]
        out_of_range = (values_to_check.abs() > (1.0 + args.range_eps)).any().item()
        if out_of_range:
            stats.out_of_range_values += 1

        if row_bad:
            stats.bad_encoding += 1
            stats.mark_bad(idx)

    return stats


def print_report(stats: ScanStats):
    print("\n=== Original Dataset Encoding Report ===")
    print(f"rows_scanned: {stats.total_rows}")
    print(f"bad_width_height: {stats.bad_width_height}")
    print(f"bad_shapes_json: {stats.bad_shapes_json}")
    print(f"bad_encoding_total: {stats.bad_encoding}")
    print(f"bad_tensor_shape: {stats.bad_tensor_shape}")
    print(f"bad_tensor_finite: {stats.bad_tensor_finite}")
    print(f"bad_real_count: {stats.bad_real_count}")
    print(f"bad_padding: {stats.bad_padding}")
    print(f"rows_with_out_of_range_values: {stats.out_of_range_values}")
    print(f"rows_truncated_by_max_segments: {stats.truncated_rows}")
    print(f"total_curves_raw: {stats.total_curves_raw}")
    print(f"total_curves_encoded: {stats.total_curves_encoded}")
    print(f"max_abs_encoded_value: {stats.max_abs_encoded_value:.6f}")
    print(f"example_bad_indices (up to 25): {stats.bad_row_indices}")


def main():
    args = parse_args()
    stats = run_scan(args)
    print_report(stats)


if __name__ == "__main__":
    main()
