#!/usr/bin/env bash
set -euo pipefail

mkdir -p generated

rsvg-convert -f pdf -o generated/pipeline.pdf ../text/assets/pipeline.svg
rsvg-convert -f pdf -o generated/architecture.pdf ../text/assets/architecture.svg
rsvg-convert -f pdf -o generated/svg_primitives_example.pdf ../text/assets/svg_primitives_example.svg
rsvg-convert -f pdf -o generated/svg_bezier_example.pdf ../text/assets/svg_bezier_example.svg

for name in lighthouse rocket cat potion cabin steampunk; do
  rsvg-convert -f pdf -o "generated/${name}.pdf" "../text/assets/text_svg_pipeline/${name}.svg"
  rsvg-convert -f pdf -o "generated/omnisvg4_${name}.pdf" "../text/assets/text_svg_pipeline/omnisvg_4b/${name}.svg"
  rsvg-convert -f pdf -o "generated/omnisvg8_${name}.pdf" "../text/assets/text_svg_pipeline/omnisvg_8b/${name}.svg"
done

latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error slides.tex
