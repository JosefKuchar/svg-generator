#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_z_image_loras.sh OUTPUT_ROOT LORA_DIR [LORA_DIR ...]

Example:
  ./scripts/run_z_image_loras.sh ./z-image-renders ./lora-runs ./more-loras

This script scans each provided directory recursively for .safetensors files and
runs scripts/z-image-dataset.py once per LoRA.
EOF
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required but was not found in PATH." >&2
  exit 1
fi

output_root=$1
shift

mkdir -p "$output_root"

declare -a lora_files=()

for lora_dir in "$@"; do
  if [[ ! -d "$lora_dir" ]]; then
    echo "Warning: skipping missing directory: $lora_dir" >&2
    continue
  fi

  while IFS= read -r -d '' lora_file; do
    lora_files+=("$lora_file")
  done < <(find "$lora_dir" -type f -name '*.safetensors' -print0 | sort -z)
done

if [[ ${#lora_files[@]} -eq 0 ]]; then
  echo "Error: no .safetensors files found in the provided directories." >&2
  exit 1
fi

declare -A used_names=()

for lora_file in "${lora_files[@]}"; do
  lora_name=$(basename "$lora_file")
  lora_stem=${lora_name%.safetensors}
  output_dir="$output_root/$lora_stem"

  if [[ -v "used_names[$lora_stem]" ]]; then
    used_names["$lora_stem"]=$((used_names["$lora_stem"] + 1))
    output_dir="${output_dir}-${used_names[$lora_stem]}"
  else
    used_names["$lora_stem"]=1
  fi

  echo "Running LoRA: $lora_file"
  echo "Output dir:   $output_dir"

  uv run python scripts/z-image-dataset.py \
    --output-dir "$output_dir" \
    --caption-index 1 \
    --num-samples 100 \
    --prompt-prefix "SVG illustration with white background. " \
    --batch-size 8 \
    --num-inference-steps 50 \
    --guidance-scale 4.0 \
    --model-id "Tongyi-MAI/Z-Image" \
    --lora-path "$lora_file"
done
