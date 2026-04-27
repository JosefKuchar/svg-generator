import csv
import json
import re
from pathlib import Path
from typing import Optional

import typer
import wandb


def _parse_csv(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def _parse_json_object(value: Optional[str], option_name: str) -> Optional[dict]:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"{option_name} must be valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise typer.BadParameter(f"{option_name} must decode to a JSON object")
    return parsed


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "run"


def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _run_attr(run, *names: str):
    for name in names:
        try:
            return getattr(run, name)
        except AttributeError:
            continue
    return None


def _scan_history(run, keys: Optional[list[str]], page_size: int) -> list[dict]:
    if keys is None:
        return list(run.scan_history(page_size=page_size))

    # W&B returns only rows where all requested keys are present together. Some
    # Lightning metrics are logged at different steps, so export each metric
    # independently and merge by step to avoid silently producing an empty CSV.
    rows_by_step: dict[int, dict] = {}
    rows_without_step: list[dict] = []

    for key in keys:
        for row in run.scan_history(keys=["_step", key], page_size=page_size):
            if "_step" not in row:
                rows_without_step.append(row)
                continue

            step = row["_step"]
            merged = rows_by_step.setdefault(step, {"_step": step})
            if key in row:
                merged[key] = row[key]
            if "_timestamp" in row:
                merged.setdefault("_timestamp", row["_timestamp"])
            if "_runtime" in row:
                merged.setdefault("_runtime", row["_runtime"])

    return [
        *[rows_by_step[step] for step in sorted(rows_by_step)],
        *rows_without_step,
    ]


def _write_history_csv(path: Path, rows: list[dict]) -> None:
    columns: list[str] = []
    seen = set()
    for preferred in ("_step", "_timestamp", "_runtime"):
        for row in rows:
            if preferred in row and preferred not in seen:
                columns.append(preferred)
                seen.add(preferred)
                break
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)

    if not columns:
        columns = ["_step"]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")


def _export_run(
    run,
    output_dir: Path,
    keys: Optional[list[str]],
    page_size: int,
    include_metadata: bool,
) -> tuple[Path, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = _safe_filename(f"{run.name}_{run.id}")

    rows = _scan_history(run, keys=keys, page_size=page_size)
    history_path = output_dir / f"{run_prefix}.csv"
    _write_history_csv(history_path, rows)

    if include_metadata:
        metadata = {
            "id": run.id,
            "name": run.name,
            "path": list(run.path),
            "url": run.url,
            "state": run.state,
            "created_at": str(_run_attr(run, "created_at", "createdAt")),
            "updated_at": str(_run_attr(run, "updated_at", "updatedAt")),
            "tags": list(run.tags),
            "config": _json_safe(dict(run.config)),
            "summary": _json_safe(dict(run.summary)),
        }
        _write_json(output_dir / f"{run_prefix}.json", metadata)

    return history_path, len(rows)


def export(
    entity: str = typer.Option(..., help="W&B entity or username"),
    project: str = typer.Option("svg-generator", help="W&B project name"),
    output_dir: Path = typer.Option(
        Path("results/wandb"),
        file_okay=False,
        help="Directory where CSV histories and JSON metadata are written",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        help="Export one run by ID. If omitted, exports runs selected by filters.",
    ),
    keys: Optional[str] = typer.Option(
        None,
        help="Comma-separated history keys to export, e.g. val/loss,val/image_mse",
    ),
    filters: Optional[str] = typer.Option(
        None,
        help='JSON filters for project runs, e.g. \'{"state":"finished"}\'',
    ),
    order: str = typer.Option("-created_at", help="Ordering used for project runs"),
    limit: Optional[int] = typer.Option(
        None, min=1, help="Maximum number of project runs to export"
    ),
    page_size: int = typer.Option(
        1000, min=1, help="W&B scan_history page size"
    ),
    include_metadata: bool = typer.Option(
        True, help="Write run config and summary JSON next to each CSV"
    ),
) -> None:
    metric_keys = _parse_csv(keys)
    api = wandb.Api()

    if run_id is not None:
        runs = [api.run(f"{entity}/{project}/{run_id}")]
    else:
        run_filters = _parse_json_object(filters, "filters") or {}
        runs = list(
            api.runs(
                f"{entity}/{project}",
                filters=run_filters,
                order=order,
                per_page=50,
            )
        )
        if limit is not None:
            runs = runs[:limit]

    if not runs:
        typer.echo("No matching runs found.")
        raise typer.Exit(code=1)

    typer.echo(f"Exporting {len(runs)} run(s) to {output_dir}")
    for run in runs:
        history_path, row_count = _export_run(
            run=run,
            output_dir=output_dir,
            keys=metric_keys,
            page_size=page_size,
            include_metadata=include_metadata,
        )
        typer.echo(f"{run.name} ({run.id}): {row_count} rows -> {history_path}")


if __name__ == "__main__":
    typer.run(export)
