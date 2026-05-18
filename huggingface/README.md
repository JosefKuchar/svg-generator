---
license: mit
library_name: diffusers
pipeline_tag: text-to-image
base_model: Tongyi-MAI/Z-Image
tags:
- z-image
- diffusers
- lora
- text-to-image
- svg
- vector-graphics
- image-to-vector
- flow-matching
---

# SVG Generator

This repository contains models for a two-stage text-to-SVG generation pipeline.
It includes a LoRA adapter for Z-Image that biases image generation toward clean
SVG-style illustrations and a flow-matching vectorizer that converts raster
images into Bezier-curve SVGs.

## Z-Image SVG LoRA

Adapter weights:

- `zimage-svg-lora.safetensors`

The adapter can be loaded with the standard Diffusers `ZImagePipeline`:

```python
import torch
from diffusers import ZImagePipeline

repo_id = "JosefKuchar/svg-generator"
base_model = "Tongyi-MAI/Z-Image"

pipe = ZImagePipeline.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
)
pipe.load_lora_weights(repo_id, weight_name="zimage-svg-lora.safetensors")
pipe.to("cuda")

prompt = (
    "SVG illustration with white background. "
    "A simple icon of a mountain cabin surrounded by pine trees."
)

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=4.0,
).images[0]

image.save("zimage-svg-lora-example.png")
```

The adapter was trained with `ai-toolkit` and stores its weights in bfloat16.
The published checkpoint metadata reports training step 3000 and LoRA rank 16.

## Intended Use

This LoRA is intended for generating bitmap illustrations that are easier to
vectorize in the second stage of the pipeline. Prompts should describe simple,
flat, clean illustrations and can use a prefix such as:

```text
SVG illustration with white background.
```

## Limitations

This is only the bitmap-generation stage of the full pipeline. The output of
the LoRA is still a raster image and must be converted to SVG by a separate
vectorization model or tool.

## Flow-Matching Vectorizer

Vectorizer files:

- `flow-matching/config.json`
- `flow-matching/model.safetensors`

The vectorizer is conditioned on DINOv3 image features from
`facebook/dinov3-vits16-pretrain-lvd1689m`. The DINOv3 encoder is not stored in
this repository; it is loaded separately from its original Hugging Face
repository.

The model can be loaded with the helper code from the project repository:

```python
import torch
from flow_matching_hf import load_dino_encoder, load_flow_matching_from_hub

device = "cuda" if torch.cuda.is_available() else "cpu"

vectorizer = load_flow_matching_from_hub(
    "JosefKuchar/svg-generator",
    subfolder="flow-matching",
    device=device,
)
processor, image_encoder = load_dino_encoder(device=device)
```

For folder-based PNG to SVG inference:

```bash
uv run python vectorize_png_folder_model.py ./pngs ./svgs \
  --model-repo-id JosefKuchar/svg-generator \
  --batch-size 1 \
  --steps 50 \
  --max-segments 256
```

The exported vectorizer was created from the original PyTorch Lightning
checkpoint by removing the frozen DINOv3 encoder tensors and trainer state. The
original checkpoint is not required for inference.
