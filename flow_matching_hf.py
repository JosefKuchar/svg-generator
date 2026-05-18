from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoImageProcessor, AutoModel

from model import FlowMatchingTransformer


DINO_MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_SUBFOLDER = "flow-matching"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_flow_matching_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def build_flow_matching_model(config: dict[str, Any]) -> FlowMatchingTransformer:
    model_config = dict(config["model_config"])
    model_config["load_image_encoder"] = False
    return FlowMatchingTransformer(**model_config)


def load_flow_matching_from_files(
    weights_path: str | Path,
    config_path: str | Path,
    device: str | torch.device = "auto",
) -> FlowMatchingTransformer:
    resolved_device = resolve_device(device) if isinstance(device, str) else device
    config = load_flow_matching_config(config_path)
    module = build_flow_matching_model(config)
    state_dict = load_file(str(weights_path), device=str(resolved_device))
    module.load_state_dict(state_dict, strict=True)
    module.to(resolved_device)
    module.eval()
    return module


def load_flow_matching_from_hub(
    repo_id: str = "JosefKuchar/svg-generator",
    subfolder: str = DEFAULT_SUBFOLDER,
    revision: str | None = None,
    device: str | torch.device = "auto",
) -> FlowMatchingTransformer:
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=WEIGHTS_NAME,
        subfolder=subfolder,
        revision=revision,
        repo_type="model",
    )
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=CONFIG_NAME,
        subfolder=subfolder,
        revision=revision,
        repo_type="model",
    )
    return load_flow_matching_from_files(weights_path, config_path, device=device)


def load_dino_encoder(
    device: str | torch.device = "auto",
    model_name: str = DINO_MODEL_NAME,
) -> tuple[AutoImageProcessor, AutoModel]:
    resolved_device = resolve_device(device) if isinstance(device, str) else device
    dtype = torch.bfloat16 if resolved_device.type == "cuda" else torch.float32
    processor = AutoImageProcessor.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name, dtype=dtype)
    encoder.requires_grad_(False)
    encoder.eval()
    encoder.to(resolved_device)
    return processor, encoder
