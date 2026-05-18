#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_benchmark_image_folders.sh FIRST_FOLDER SECOND_FOLDER_PREFIX OUTPUT_DIR [BENCHMARK_ARGS...]

Example:
  ./scripts/run_benchmark_image_folders.sh ./targets ./runs/render- ./benchmark-results --metric clip_similarity --batch-size 8

This script finds directories matching SECOND_FOLDER_PREFIX* and runs
scripts/benchmark_image_folders.py once per matching folder:

  scripts/benchmark_image_folders.py FIRST_FOLDER MATCHING_FOLDER [BENCHMARK_ARGS...]

The stdout/stderr for each run is saved to OUTPUT_DIR/<matching-folder-name>.txt.
EOF
}

if [[ $# -lt 3 ]]; then
  usage >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required but was not found in PATH." >&2
  exit 1
fi

first_folder=$1
second_folder_prefix=$2
output_dir=$3
shift 3

if [[ ! -d "$first_folder" ]]; then
  echo "Error: first folder does not exist: $first_folder" >&2
  exit 1
fi

mkdir -p "$output_dir"

shopt -s nullglob
matching_paths=("${second_folder_prefix}"*)
shopt -u nullglob

declare -a matching_dirs=()

for path in "${matching_paths[@]}"; do
  if [[ -d "$path" ]]; then
    matching_dirs+=("$path")
  fi
done

if [[ ${#matching_dirs[@]} -eq 0 ]]; then
  echo "Error: no directories matched prefix: ${second_folder_prefix}*" >&2
  exit 1
fi

IFS=$'\n' matching_dirs=($(printf '%s\n' "${matching_dirs[@]}" | sort))
unset IFS

for second_folder in "${matching_dirs[@]}"; do
  output_file="$output_dir/$(basename "$second_folder").txt"

  echo "Benchmarking:"
  echo "  folder_a: $first_folder"
  echo "  folder_b: $second_folder"
  echo "  output:   $output_file"

  uv run python scripts/benchmark_image_folders.py "$first_folder" "$second_folder" "$@" \
    >"$output_file" 2>&1
done
