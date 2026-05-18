# SVG Generator

## Generative Neural Models for Scalable Vector Graphics

<table>
  <tr>
    <td align="center" width="16.66%"><img src="huggingface/assets/lighthouse.svg" width="120" alt="A simple lighthouse icon"></td>
    <td align="center" width="16.66%"><img src="huggingface/assets/rocket.svg" width="120" alt="A minimal rocket icon"></td>
    <td align="center" width="16.66%"><img src="huggingface/assets/cat.svg" width="120" alt="A cute cat face icon"></td>
    <td align="center" width="16.66%"><img src="huggingface/assets/potion.svg" width="120" alt="A magic potion bottle with bubbles, sparkles, cork, and label"></td>
    <td align="center" width="16.66%"><img src="huggingface/assets/cabin.svg" width="120" alt="A cozy cabin in the woods with a chimney, pine trees, and a crescent moon"></td>
    <td align="center" width="16.66%"><img src="huggingface/assets/steampunk.svg" width="120" alt="A steampunk hot air balloon with gears, ropes, basket, and brass details"></td>
  </tr>
  <tr>
    <td align="center" width="16.66%"><sub>a simple lighthouse icon</sub></td>
    <td align="center" width="16.66%"><sub>A minimal rocket icon</sub></td>
    <td align="center" width="16.66%"><sub>A cute cat face icon</sub></td>
    <td align="center" width="16.66%"><sub>A magic potion bottle with bubbles, sparkles, cork, and label</sub></td>
    <td align="center" width="16.66%"><sub>A cozy cabin in the woods with a chimney, pine trees, and a crescent moon</sub></td>
    <td align="center" width="16.66%"><sub>A steampunk hot air balloon with gears, ropes, basket, and brass details</sub></td>
  </tr>
</table>

This repository contains a two-stage neural pipeline for generating SVG images
from text prompts. The first stage uses Z-Image with an SVG-domain LoRA to
generate a raster image. The second stage converts the raster image to SVG,
either with `vtracer` or with a conditional flow-matching vectorizer trained to
predict cubic Bezier curves.

The project was developed for the thesis *Generative Neural Models for Scalable
Vector Graphics*. The main idea is to separate semantic image generation from
geometric reconstruction: a large text-to-image model handles prompt grounding,
while a smaller vectorizer focuses on producing editable vector geometry.

## Links

- Rendered thesis text: [thesis.pdf](thesis.pdf)
- Model artifacts: <https://huggingface.co/JosefKuchar/svg-generator>
- Bezier training dataset: <https://huggingface.co/datasets/JosefKuchar/bezier-dataset>
- Source SVG/caption dataset used by the data pipeline: <https://huggingface.co/datasets/mikronai/svg-svgrepo>

## Setup

The project uses `uv` and Python 3.13.

```bash
uv sync
```

For faster transformer attention on compatible CUDA systems, install the
prebuilt FlashAttention wheel after `uv sync`:

```bash
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp313-cp313-linux_x86_64.whl
```

For the default inference path, install `vtracer` and make sure it is available
on `PATH`. `resvg` is also used for SVG raster previews and several evaluation
or data-generation utilities.

CUDA is strongly recommended for Z-Image and flow-matching inference.

## Inference

`infer.py` is the main end-to-end entry point. It downloads the published LoRA
and vectorizer artifacts from Hugging Face by default.

```bash
uv run python infer.py "a black silhouette of a person paddling a kayak" \
  --output-svg kayak.svg \
  --output-png kayak.png \
  --preview-png kayak-preview.png
```

The default prompt prefix is:

```text
SVG illustration with white background. 
```

You can override it when needed:

```bash
uv run python infer.py "a mountain cabin icon" \
  --prompt-prefix "Flat vector icon on white background. " \
  --output-svg cabin.svg
```

### Stage 1: Z-Image

The first stage can use either Z-Image Base or Z-Image Turbo:

```bash
# Default: Tongyi-MAI/Z-Image, 50 steps, guidance 4.0
uv run python infer.py "a simple lighthouse icon" --z-image base

# Turbo: Tongyi-MAI/Z-Image-Turbo, 9 steps, guidance 0.0
uv run python infer.py "a simple lighthouse icon" --z-image turbo
```

The stage-1 defaults can be overridden with `--z-steps`,
`--z-guidance-scale`, `--lora-scale`, `--height`, `--width`, and `--seed`.

### Stage 2: Vectorization

The default vectorizer is `vtracer`, which is usually the most reliable choice
for direct raster fidelity:

```bash
uv run python infer.py "a simple tree icon" \
  --vectorizer vtracer \
  --output-svg tree.svg
```

Extra `vtracer` flags can be passed by repeating `--vtracer-arg`:

```bash
uv run python infer.py "a simple tree icon" \
  --vtracer-arg=--colormode \
  --vtracer-arg=color
```

The neural flow-matching vectorizer is also available:

```bash
uv run python infer.py "a simple tree icon" \
  --vectorizer flow-matching \
  --flow-steps 50 \
  --flow-cfg-scale 1.0 \
  --max-segments 256 \
  --output-svg tree.svg
```

## Training

`train.py` trains the flow-matching vectorizer. It uses PyTorch Lightning and
logs to Weights & Biases.

Training on the Hugging Face Bezier dataset:

```bash
uv run python train.py \
  --batch-size 256 \
  --learning-rate 1e-4 \
  --warmup-steps 1000
```

Training on procedural synthetic data:

```bash
uv run python train.py \
  --synthetic \
  --synthetic-length 100000 \
  --synthetic-min-shapes 1 \
  --synthetic-max-shapes 10
```

For quick debugging, limit the dataset:

```bash
uv run python train.py --max-samples 32 --batch-size 8
```

Checkpoints are selected by validation image MSE and stored by PyTorch
Lightning/W&B in the run directory.

## Repository Layout

Root files are the reusable implementation and primary entry points:

- `infer.py` - end-to-end text-to-SVG inference with Z-Image + LoRA and either
  `vtracer` or the flow-matching vectorizer.
- `train.py` - training entry point for the flow-matching vectorizer.
- `model.py` - conditional flow-matching transformer and sampling code.
- `dataset.py` - Lightning data module and Hugging Face dataset loading.
- `representation.py` - fixed-size Bezier tensor representation and conversion
  utilities.
- `parsing.py` - SVG parsing and Bezier-shape serialization helpers.
- `raster.py` - SVG rendering, image vectorization, and raster metrics.
- `synthetic.py` - procedural Bezier scene generator used for synthetic
  training data.
- `hub.py` - small helpers for loading the published Hugging Face artifacts and
  the DINOv3 image encoder.
- `text/` - thesis source written in Typst.

The `scripts/` directory contains one-off and supporting workflows: benchmark
and evaluation scripts, dataset export utilities, plotting scripts, Hugging
Face publishing helpers, and older standalone inference/vectorization tools.

## Published Artifacts

The Hugging Face model repository contains:

- `zimage-svg-lora.safetensors` - LoRA adapter for Z-Image.
- `flow-matching/config.json` - architecture/configuration for the neural
  vectorizer.
- `flow-matching/model.safetensors` - flow-matching vectorizer weights without
  the frozen DINOv3 encoder or trainer state.

The flow-matching vectorizer is conditioned on
`facebook/dinov3-vits16-pretrain-lvd1689m`, which is loaded separately.

## Main Dependencies

- PyTorch - model implementation, tensor operations, and inference.
- PyTorch Lightning - training loop, checkpointing, and validation logging.
- Diffusers - Z-Image pipeline and LoRA loading.
- Transformers - DINOv3 image encoder and image preprocessing.
- Hugging Face Hub/Datasets - published model artifacts and training datasets.
- Typer - command-line interfaces for inference, training, and utilities.
- `vtracer` - default classical raster-to-SVG vectorizer.
- `resvg` - SVG rasterization for previews, data generation, and evaluation.
- FlashAttention - optional faster CUDA attention for supported systems.

## License

This repository is licensed under the MIT License.
