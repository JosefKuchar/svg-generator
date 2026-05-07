#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./vectorize_png_folder_vtracer.sh [OPTIONS] INPUT_DIR OUTPUT_DIR [-- VTRACER_ARGS...]

Options:
  -r, --recursive   Process PNG files recursively and preserve subdirectories.
  -f, --overwrite   Overwrite existing SVG outputs.
  -h, --help        Show this help message.

Examples:
  ./vectorize_png_folder_vtracer.sh ./pngs ./svgs
  ./vectorize_png_folder_vtracer.sh --recursive ./pngs ./svgs
  ./vectorize_png_folder_vtracer.sh ./pngs ./svgs -- --colormode color --hierarchical stacked

Each input PNG is written to OUTPUT_DIR with the same base name and a .svg suffix.
Additional arguments after -- are passed directly to vtracer.
EOF
}

recursive=0
overwrite=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--recursive)
      recursive=1
      shift
      ;;
    -f|--overwrite)
      overwrite=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Error: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 1
fi

input_dir=$1
output_dir=$2
shift 2

if [[ ! -d "$input_dir" ]]; then
  echo "Error: input directory does not exist: $input_dir" >&2
  exit 1
fi

if ! command -v vtracer >/dev/null 2>&1; then
  echo "Error: vtracer is required but was not found in PATH." >&2
  exit 1
fi

mkdir -p "$output_dir"

declare -a png_paths=()

if [[ "$recursive" -eq 1 ]]; then
  while IFS= read -r -d '' path; do
    png_paths+=("$path")
  done < <(find "$input_dir" -type f \( -iname '*.png' \) -print0 | sort -z)
else
  shopt -s nullglob nocaseglob
  png_paths=("$input_dir"/*.png)
  shopt -u nullglob nocaseglob
fi

if [[ ${#png_paths[@]} -eq 0 ]]; then
  scope="in"
  if [[ "$recursive" -eq 1 ]]; then
    scope="recursively in"
  fi
  echo "Error: no PNG files found $scope $input_dir." >&2
  exit 1
fi

converted=0
skipped=0

for input_path in "${png_paths[@]}"; do
  if [[ "$recursive" -eq 1 ]]; then
    relative_path=${input_path#"$input_dir"/}
    output_path="$output_dir/${relative_path%.*}.svg"
  else
    filename=$(basename "$input_path")
    output_path="$output_dir/${filename%.*}.svg"
  fi

  if [[ -e "$output_path" && "$overwrite" -ne 1 ]]; then
    echo "Skipping existing output: $output_path"
    skipped=$((skipped + 1))
    continue
  fi

  mkdir -p "$(dirname "$output_path")"

  echo "Vectorizing: $input_path -> $output_path"
  vtracer --input "$input_path" --output "$output_path" "$@"
  converted=$((converted + 1))
done

echo "Done. Converted $converted PNG file(s), skipped $skipped existing output(s)."
