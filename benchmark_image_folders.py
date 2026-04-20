from abc import ABC, abstractmethod
from itertools import islice
from pathlib import Path
import tempfile

import numpy as np
import torch
import typer
from PIL import Image
from tqdm.auto import tqdm

from raster import has_raster_tool, render_svg_bg, vectorize_image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VECTORIZATION_METRIC = "vectorization_mse"
DEFAULT_METRICS = ["clip_similarity", "dino_similarity", VECTORIZATION_METRIC]

app = typer.Typer()


class Metric(ABC):
    name: str
    input_kind = "pair"

    def update(self, image_a: Path, image_b: Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support pair updates")

    def update_batch(self, image_pairs: list[tuple[Path, Path]]) -> None:
        for image_a, image_b in image_pairs:
            self.update(image_a, image_b)

    def update_folder_image(self, image_path: Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support folder-only updates")

    def update_folder_batch(self, image_paths: list[Path]) -> None:
        for image_path in image_paths:
            self.update_folder_image(image_path)

    def compute(self) -> float:
        raise NotImplementedError

class ClipSimilarityMetric(Metric):
    name = "clip_similarity"

    def __init__(self, device: str, model_name: str):
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.total = 0.0
        self.count = 0
        self.model.eval()

    def update(self, image_a: Path, image_b: Path) -> None:
        pil_a = _open_rgb_image(image_a)
        pil_b = _open_rgb_image(image_b)
        inputs = self.processor(images=[pil_a, pil_b], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            image_features = self.model.get_image_features(**inputs)
            image_features = _as_feature_tensor(image_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = torch.sum(image_features[0] * image_features[1]).item()

        self.total += similarity
        self.count += 1

    def update_batch(self, image_pairs: list[tuple[Path, Path]]) -> None:
        images = []
        for image_a, image_b in image_pairs:
            images.extend([_open_rgb_image(image_a), _open_rgb_image(image_b)])

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            image_features = self.model.get_image_features(**inputs)
            image_features = _as_feature_tensor(image_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            paired_features = image_features.view(len(image_pairs), 2, -1)
            similarities = torch.sum(paired_features[:, 0] * paired_features[:, 1], dim=-1)

        self.total += similarities.sum().item()
        self.count += len(image_pairs)

    def compute(self) -> float:
        return self.total / self.count if self.count else 0.0


class DinoV3SimilarityMetric(Metric):
    name = "dino_similarity"

    def __init__(self, device: str, model_name: str):
        from transformers import AutoImageProcessor, AutoModel

        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.total = 0.0
        self.count = 0
        self.model.eval()

    def update(self, image_a: Path, image_b: Path) -> None:
        pil_a = _open_rgb_image(image_a)
        pil_b = _open_rgb_image(image_b)
        inputs = self.processor(images=[pil_a, pil_b], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            image_features = self.model(**inputs)
            image_features = _as_feature_tensor(image_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = torch.sum(image_features[0] * image_features[1]).item()

        self.total += similarity
        self.count += 1

    def update_batch(self, image_pairs: list[tuple[Path, Path]]) -> None:
        images = []
        for image_a, image_b in image_pairs:
            images.extend([_open_rgb_image(image_a), _open_rgb_image(image_b)])

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            image_features = self.model(**inputs)
            image_features = _as_feature_tensor(image_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            paired_features = image_features.view(len(image_pairs), 2, -1)
            similarities = torch.sum(paired_features[:, 0] * paired_features[:, 1], dim=-1)

        self.total += similarities.sum().item()
        self.count += len(image_pairs)

    def compute(self) -> float:
        return self.total / self.count if self.count else 0.0


class VectorizationMSEMetric(Metric):
    name = "vectorization_mse"
    input_kind = "folder_b"

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update_folder_image(self, image_path: Path) -> None:
        original = _open_rgb_image(image_path)

        with tempfile.TemporaryDirectory(prefix="vtracer-") as temp_dir:
            svg_path = Path(temp_dir) / f"{image_path.stem}.svg"
            vectorize_image(image_path, svg_path)
            rerendered = _rasterize_svg(svg_path, width=original.width, height=original.height)

        self.total += _calculate_mse(original, rerendered)
        self.count += 1

    def compute(self) -> float:
        return self.total / self.count if self.count else 0.0


def _open_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")

def _rasterize_svg(svg_path: Path, width: int, height: int) -> Image.Image:
    svg_content = svg_path.read_bytes()
    return render_svg_bg(svg_content, width=width, height=height).convert("RGB")


def _calculate_mse(image_a: Image.Image, image_b: Image.Image) -> float:
    if image_a.size != image_b.size:
        raise ValueError(
            f"MSE requires matching image sizes, got {image_a.size} and {image_b.size}"
        )

    arr_a = np.asarray(image_a.convert("RGB"), dtype=np.float32)
    arr_b = np.asarray(image_b.convert("RGB"), dtype=np.float32)
    return float(np.mean((arr_a - arr_b) ** 2))


def _as_feature_tensor(image_features: torch.Tensor) -> torch.Tensor:
    if isinstance(image_features, torch.Tensor):
        return image_features

    image_embeds = getattr(image_features, "image_embeds", None)
    if isinstance(image_embeds, torch.Tensor):
        return image_embeds

    pooler_output = getattr(image_features, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output

    last_hidden_state = getattr(image_features, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state[:, 0]

    raise TypeError(
        f"Unsupported image feature output type: {type(image_features).__name__}"
    )


def _collect_images(folder: Path) -> dict[str, Path]:
    images = {
        path.name: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not images:
        raise ValueError(f"No supported images found in {folder}")
    return images


def _build_metrics(
    metric_names: list[str],
    device: str,
    clip_model: str,
    dino_model: str,
    *,
    using_default_metrics: bool,
) -> list[Metric]:
    requested_vectorization_metric = VECTORIZATION_METRIC in metric_names
    if requested_vectorization_metric and not using_default_metrics and not has_raster_tool("vtracer"):
        raise typer.BadParameter(
            "Metric 'vectorization_mse' requires the 'vtracer' binary on PATH."
        )

    effective_metric_names = list(metric_names)
    if using_default_metrics and not has_raster_tool("vtracer"):
        effective_metric_names = [
            metric_name
            for metric_name in effective_metric_names
            if metric_name != VECTORIZATION_METRIC
        ]

    metrics: list[Metric] = []
    for metric_name in effective_metric_names:
        if metric_name == "clip_similarity":
            metrics.append(ClipSimilarityMetric(device=device, model_name=clip_model))
        elif metric_name == "dino_similarity":
            metrics.append(DinoV3SimilarityMetric(device=device, model_name=dino_model))
        elif metric_name == VECTORIZATION_METRIC:
            metrics.append(VectorizationMSEMetric())
        else:
            raise typer.BadParameter(f"Unsupported metric: {metric_name}")
    return metrics


def _batched_image_pairs(
    common_names: list[str], images_a: dict[str, Path], images_b: dict[str, Path], batch_size: int
) -> list[list[tuple[Path, Path]]]:
    iterator = iter(common_names)
    batches: list[list[tuple[Path, Path]]] = []
    while batch_names := list(islice(iterator, batch_size)):
        batches.append([(images_a[name], images_b[name]) for name in batch_names])
    return batches


def _batched_image_paths(images: dict[str, Path], batch_size: int) -> list[list[Path]]:
    iterator = iter(sorted(images))
    batches: list[list[Path]] = []
    while batch_names := list(islice(iterator, batch_size)):
        batches.append([images[name] for name in batch_names])
    return batches


@app.command()
def main(
    folder_a: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    folder_b: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    metrics: list[str] | None = typer.Option(
        None,
        "--metric",
        help="Metric to compute. Defaults to all metrics; repeat the option to choose a subset.",
    ),
    clip_model: str = typer.Option(
        "openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model name for image similarity.",
    ),
    dino_model: str = typer.Option(
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Hugging Face DINOv3 model name for image similarity.",
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use for image similarity metrics.",
    ),
    batch_size: int = typer.Option(
        16,
        min=1,
        help="How many matching image pairs to process per batch.",
    ),
):
    """Compare matching images in two folders and print average metrics."""

    using_default_metrics = metrics is None
    selected_metrics = list(DEFAULT_METRICS if metrics is None else metrics)
    if using_default_metrics and not has_raster_tool("vtracer"):
        print("Skipping vectorization_mse: 'vtracer' is not installed or not on PATH")

    metric_instances = _build_metrics(
        selected_metrics,
        device=device,
        clip_model=clip_model,
        dino_model=dino_model,
        using_default_metrics=using_default_metrics,
    )
    pair_metrics = [metric for metric in metric_instances if metric.input_kind == "pair"]
    folder_b_metrics = [metric for metric in metric_instances if metric.input_kind == "folder_b"]

    images_a = _collect_images(folder_a)
    images_b = _collect_images(folder_b)

    if pair_metrics:
        common_names = sorted(set(images_a) & set(images_b))
        only_a = sorted(set(images_a) - set(images_b))
        only_b = sorted(set(images_b) - set(images_a))

        if not common_names:
            raise typer.BadParameter("No matching filenames found between the folders")

        if only_a:
            print(f"Skipping {len(only_a)} files only in {folder_a}: {', '.join(only_a[:5])}")
        if only_b:
            print(f"Skipping {len(only_b)} files only in {folder_b}: {', '.join(only_b[:5])}")

        image_pair_batches = _batched_image_pairs(common_names, images_a, images_b, batch_size)
        for image_pairs in tqdm(image_pair_batches, desc="Comparing image batches"):
            for metric in pair_metrics:
                metric.update_batch(image_pairs)

        print(f"Compared {len(common_names)} matching image pairs")

    if folder_b_metrics:
        image_batches = _batched_image_paths(images_b, batch_size)
        for image_paths in tqdm(image_batches, desc="Vectorizing folder_b images"):
            for metric in folder_b_metrics:
                metric.update_folder_batch(image_paths)

        print(f"Processed {len(images_b)} images from {folder_b} for folder_b-only metrics")

    for metric in metric_instances:
        print(f"{metric.name}: {metric.compute():.6f}")


if __name__ == "__main__":
    app()
