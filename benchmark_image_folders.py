from abc import ABC, abstractmethod
from itertools import islice
from pathlib import Path

import torch
import typer
from PIL import Image
from tqdm.auto import tqdm
from torchvision.transforms.functional import pil_to_tensor


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

app = typer.Typer()


class Metric(ABC):
    name: str

    @abstractmethod
    def update(self, image_a: Path, image_b: Path) -> None:
        pass

    def update_batch(self, image_pairs: list[tuple[Path, Path]]) -> None:
        for image_a, image_b in image_pairs:
            self.update(image_a, image_b)

    @abstractmethod
    def compute(self) -> float:
        pass


class MeanSquaredErrorMetric(Metric):
    name = "mse"

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, image_a: Path, image_b: Path) -> None:
        tensor_a = _load_image_tensor(image_a)
        tensor_b = _load_image_tensor(image_b)

        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f"MSE requires matching image shapes, got {image_a.name}: {tuple(tensor_a.shape)} "
                f"and {image_b.name}: {tuple(tensor_b.shape)}"
            )

        self.total += torch.mean((tensor_a - tensor_b) ** 2).item()
        self.count += 1

    def update_batch(self, image_pairs: list[tuple[Path, Path]]) -> None:
        tensors_a = [_load_image_tensor(image_a) for image_a, _ in image_pairs]
        tensors_b = [_load_image_tensor(image_b) for _, image_b in image_pairs]

        for (image_a, image_b), tensor_a, tensor_b in zip(image_pairs, tensors_a, tensors_b):
            if tensor_a.shape != tensor_b.shape:
                raise ValueError(
                    f"MSE requires matching image shapes, got {image_a.name}: {tuple(tensor_a.shape)} "
                    f"and {image_b.name}: {tuple(tensor_b.shape)}"
                )

        batch_a = torch.stack(tensors_a)
        batch_b = torch.stack(tensors_b)
        self.total += torch.mean((batch_a - batch_b) ** 2, dim=(1, 2, 3)).sum().item()
        self.count += len(image_pairs)

    def compute(self) -> float:
        return self.total / self.count if self.count else 0.0


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


def _open_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _load_image_tensor(image_path: Path) -> torch.Tensor:
    image = _open_rgb_image(image_path)
    return pil_to_tensor(image).float() / 255.0


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


def _build_metrics(metric_names: list[str], device: str, clip_model: str) -> list[Metric]:
    metrics: list[Metric] = []
    for metric_name in metric_names:
        if metric_name == "mse":
            metrics.append(MeanSquaredErrorMetric())
        elif metric_name == "clip_similarity":
            metrics.append(ClipSimilarityMetric(device=device, model_name=clip_model))
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


@app.command()
def main(
    folder_a: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    folder_b: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    metrics: list[str] = typer.Option(
        ["mse", "clip_similarity"],
        "--metric",
        help="Metric to compute. Repeat the option to choose a subset.",
    ),
    clip_model: str = typer.Option(
        "openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model name for image similarity.",
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use for CLIP similarity.",
    ),
    batch_size: int = typer.Option(
        16,
        min=1,
        help="How many matching image pairs to process per batch.",
    ),
):
    """Compare matching images in two folders and print average metrics."""

    images_a = _collect_images(folder_a)
    images_b = _collect_images(folder_b)

    common_names = sorted(set(images_a) & set(images_b))
    only_a = sorted(set(images_a) - set(images_b))
    only_b = sorted(set(images_b) - set(images_a))

    if not common_names:
        raise typer.BadParameter("No matching filenames found between the folders")

    if only_a:
        print(f"Skipping {len(only_a)} files only in {folder_a}: {', '.join(only_a[:5])}")
    if only_b:
        print(f"Skipping {len(only_b)} files only in {folder_b}: {', '.join(only_b[:5])}")

    metric_instances = _build_metrics(metrics, device=device, clip_model=clip_model)
    image_pair_batches = _batched_image_pairs(common_names, images_a, images_b, batch_size)

    for image_pairs in tqdm(image_pair_batches, desc="Comparing image batches"):
        for metric in metric_instances:
            metric.update_batch(image_pairs)

    print(f"Compared {len(common_names)} matching image pairs")
    for metric in metric_instances:
        print(f"{metric.name}: {metric.compute():.6f}")


if __name__ == "__main__":
    app()
