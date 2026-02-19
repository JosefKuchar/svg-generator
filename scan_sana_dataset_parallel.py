#!/usr/bin/env python3
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import heapq

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from representation import BezierPath, BezierShape, shapes_to_tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan the full Sana dataset split for malformed text embeddings "
            "and non-finite bezier curve tensors, using multiple threads."
        )
    )
    parser.add_argument(
        "--dataset",
        default="bezier_dataset_with_text_embeddings",
        help="HF dataset id or local save_to_disk directory",
    )
    parser.add_argument("--split", default="train", help="Dataset split to scan")
    parser.add_argument(
        "--max-segments",
        type=int,
        default=256,
        help="max_segments passed to shapes_to_tensor",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of worker threads",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows to scan",
    )
    parser.add_argument(
        "--show-bad-limit",
        type=int,
        default=50,
        help="How many bad row indices to keep in the report",
    )
    parser.add_argument(
        "--curve-abs-threshold",
        type=float,
        default=10.0,
        help=(
            "Mark row bad if max abs(curve tensor) exceeds this value. "
            "Set negative to disable."
        ),
    )
    parser.add_argument(
        "--embedding-abs-threshold",
        type=float,
        default=-1.0,
        help=(
            "Mark row bad if max abs(text embedding) exceeds this value. "
            "Set negative to disable."
        ),
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="How many largest-curve rows to print",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write bad rows and top-k rows as JSON",
    )
    return parser.parse_args()


@dataclass
class ScanStats:
    rows_scanned: int = 0
    bad_text_embeddings: int = 0
    bad_curve_tensor: int = 0
    bad_shapes_json: int = 0
    bad_width_height: int = 0
    curve_out_of_range: int = 0
    embedding_out_of_range: int = 0
    max_abs_embedding: float = 0.0
    max_abs_curve: float = 0.0
    bad_indices: list[int] = field(default_factory=list)
    bad_reasons: list[str] = field(default_factory=list)

    def mark_bad(self, idx: int, reason: str, limit: int):
        if len(self.bad_indices) < limit:
            self.bad_indices.append(idx)
            self.bad_reasons.append(reason)


def _load_split(dataset_name_or_path: str, split: str):
    path = Path(dataset_name_or_path)
    if path.exists() and path.is_dir():
        return load_from_disk(str(path))[split]
    return load_dataset(dataset_name_or_path, split=split)


def _shapes_from_json(shapes_data):
    bezier_shapes = []
    for shape_data in shapes_data:
        bezier_paths = []
        for path_data in shape_data["paths"]:
            curves = [tuple(tuple(point) for point in curve) for curve in path_data["curves"]]
            bezier_paths.append(BezierPath(curves))

        bezier_shapes.append(
            BezierShape(
                paths=bezier_paths,
                color=tuple(shape_data["color"]) if shape_data["color"] else (0.0, 0.0, 0.0),
                opacity=shape_data["opacity"],
            )
        )
    return bezier_shapes


def _check_row(
    idx: int,
    ds,
    max_segments: int,
    curve_abs_threshold: float | None,
    embedding_abs_threshold: float | None,
):
    row = ds[idx]
    row_bad = False
    reason_parts = []

    max_abs_embedding = 0.0
    max_abs_curve = 0.0

    emb = np.asarray(row.get("text_embeddings", []))
    bad_text_embeddings = emb.size == 0 or not np.isfinite(emb).all()
    if bad_text_embeddings:
        row_bad = True
        reason_parts.append("text_embeddings")
    else:
        max_abs_embedding = float(np.max(np.abs(emb)))
        if (
            embedding_abs_threshold is not None
            and max_abs_embedding > embedding_abs_threshold
        ):
            row_bad = True
            reason_parts.append("embedding_out_of_range")

    width = row.get("width", None)
    height = row.get("height", None)
    bad_width_height = True
    try:
        width_f = float(width)
        height_f = float(height)
        bad_width_height = (
            not np.isfinite(width_f)
            or not np.isfinite(height_f)
            or width_f <= 0
            or height_f <= 0
        )
    except Exception:
        bad_width_height = True

    if bad_width_height:
        row_bad = True
        reason_parts.append("width_height")

    bad_shapes_json = False
    bad_curve_tensor = False
    try:
        shapes_data = json.loads(row["shapes"])
        shapes = _shapes_from_json(shapes_data)
        curve_tensor = shapes_to_tensor(
            shapes,
            width=width_f,
            height=height_f,
            max_segments=max_segments,
        )
        if not torch.isfinite(curve_tensor).all().item():
            bad_curve_tensor = True
            row_bad = True
            reason_parts.append("curve_non_finite")
        else:
            max_abs_curve = float(curve_tensor.abs().max().item())
            if curve_abs_threshold is not None and max_abs_curve > curve_abs_threshold:
                row_bad = True
                reason_parts.append("curve_out_of_range")
    except Exception:
        bad_shapes_json = True
        bad_curve_tensor = True
        row_bad = True
        reason_parts.append("shapes_or_curve_exception")

    return {
        "idx": idx,
        "row_bad": row_bad,
        "reason": ",".join(reason_parts),
        "bad_text_embeddings": int(bad_text_embeddings),
        "bad_curve_tensor": int(bad_curve_tensor),
        "bad_shapes_json": int(bad_shapes_json),
        "bad_width_height": int(bad_width_height),
        "curve_out_of_range": int(
            curve_abs_threshold is not None and max_abs_curve > curve_abs_threshold
        ),
        "embedding_out_of_range": int(
            embedding_abs_threshold is not None
            and max_abs_embedding > embedding_abs_threshold
        ),
        "max_abs_embedding": max_abs_embedding,
        "max_abs_curve": max_abs_curve,
    }


def main():
    args = parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.topk < 1:
        raise ValueError("--topk must be >= 1")

    curve_abs_threshold = (
        None if args.curve_abs_threshold is None or args.curve_abs_threshold < 0 else args.curve_abs_threshold
    )
    embedding_abs_threshold = (
        None
        if args.embedding_abs_threshold is None or args.embedding_abs_threshold < 0
        else args.embedding_abs_threshold
    )

    ds = _load_split(args.dataset, args.split)
    total_len = len(ds)
    if args.max_rows is None:
        indices = list(range(total_len))
    else:
        indices = list(range(min(args.max_rows, total_len)))

    stats = ScanStats(rows_scanned=len(indices))

    top_curve_rows: list[tuple[float, int, float]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = pool.map(
            lambda i: _check_row(
                i,
                ds,
                args.max_segments,
                curve_abs_threshold,
                embedding_abs_threshold,
            ),
            indices,
        )
        for result in tqdm(iterator, total=len(indices), desc=f"scan [{args.split}]", unit="row"):
            stats.bad_text_embeddings += result["bad_text_embeddings"]
            stats.bad_curve_tensor += result["bad_curve_tensor"]
            stats.bad_shapes_json += result["bad_shapes_json"]
            stats.bad_width_height += result["bad_width_height"]
            stats.curve_out_of_range += result["curve_out_of_range"]
            stats.embedding_out_of_range += result["embedding_out_of_range"]
            stats.max_abs_embedding = max(stats.max_abs_embedding, result["max_abs_embedding"])
            stats.max_abs_curve = max(stats.max_abs_curve, result["max_abs_curve"])

            item = (result["max_abs_curve"], result["idx"], result["max_abs_embedding"])
            if len(top_curve_rows) < args.topk:
                heapq.heappush(top_curve_rows, item)
            else:
                if item[0] > top_curve_rows[0][0]:
                    heapq.heapreplace(top_curve_rows, item)

            if result["row_bad"]:
                stats.mark_bad(result["idx"], result["reason"], args.show_bad_limit)

    print("\n=== Parallel Dataset Scan Report ===")
    print(f"dataset: {args.dataset}")
    print(f"split: {args.split}")
    print(f"rows_scanned: {stats.rows_scanned}")
    print(f"workers: {args.workers}")
    print(f"bad_text_embeddings: {stats.bad_text_embeddings}")
    print(f"bad_curve_tensor: {stats.bad_curve_tensor}")
    print(f"bad_shapes_json: {stats.bad_shapes_json}")
    print(f"bad_width_height: {stats.bad_width_height}")
    print(f"curve_out_of_range: {stats.curve_out_of_range}")
    print(f"embedding_out_of_range: {stats.embedding_out_of_range}")
    print(f"max_abs_embedding: {stats.max_abs_embedding:.6f}")
    print(f"max_abs_curve: {stats.max_abs_curve:.6f}")
    print(f"curve_abs_threshold: {curve_abs_threshold}")
    print(f"embedding_abs_threshold: {embedding_abs_threshold}")

    if stats.bad_indices:
        print("\nexample_bad_rows:")
        for idx, reason in zip(stats.bad_indices, stats.bad_reasons):
            print(f"  idx={idx} reason={reason}")
    else:
        print("\nexample_bad_rows: []")

    top_sorted = sorted(top_curve_rows, key=lambda t: t[0], reverse=True)
    print(f"\ntop_{len(top_sorted)}_rows_by_max_abs_curve:")
    for curve_max, idx, emb_max in top_sorted:
        print(f"  idx={idx} max_abs_curve={curve_max:.6f} max_abs_embedding={emb_max:.6f}")

    if args.output_json:
        output = {
            "dataset": args.dataset,
            "split": args.split,
            "rows_scanned": stats.rows_scanned,
            "workers": args.workers,
            "bad_text_embeddings": stats.bad_text_embeddings,
            "bad_curve_tensor": stats.bad_curve_tensor,
            "bad_shapes_json": stats.bad_shapes_json,
            "bad_width_height": stats.bad_width_height,
            "curve_out_of_range": stats.curve_out_of_range,
            "embedding_out_of_range": stats.embedding_out_of_range,
            "max_abs_embedding": stats.max_abs_embedding,
            "max_abs_curve": stats.max_abs_curve,
            "curve_abs_threshold": curve_abs_threshold,
            "embedding_abs_threshold": embedding_abs_threshold,
            "example_bad_rows": [
                {"idx": idx, "reason": reason}
                for idx, reason in zip(stats.bad_indices, stats.bad_reasons)
            ],
            "top_rows_by_max_abs_curve": [
                {
                    "idx": idx,
                    "max_abs_curve": curve_max,
                    "max_abs_embedding": emb_max,
                }
                for curve_max, idx, emb_max in top_sorted
            ],
        }
        Path(args.output_json).write_text(json.dumps(output, indent=2))
        print(f"\nWrote report JSON: {args.output_json}")


if __name__ == "__main__":
    main()
