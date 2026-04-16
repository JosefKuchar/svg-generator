from abc import ABC, abstractmethod
from pathlib import Path

import torch
import typer
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

app = typer.Typer()


class Metric(ABC):
    name: str

    @abstractmethod
    def update(self, image_a: Path, image_b: Path) -> None:
        pass

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
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = torch.sum(image_features[0] * image_features[1]).item()

        self.total += similarity
        self.count += 1

    def compute(self) -> float:
        return self.total / self.count if self.count else 0.0


def _open_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _load_image_tensor(image_path: Path) -> torch.Tensor:
    image = _open_rgb_image(image_path)
    return pil_to_tensor(image).float() / 255.0


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

    for name in common_names:
        image_a = images_a[name]
        image_b = images_b[name]
        for metric in metric_instances:
            metric.update(image_a, image_b)

    print(f"Compared {len(common_names)} matching image pairs")
    for metric in metric_instances:
        print(f"{metric.name}: {metric.compute():.6f}")


if __name__ == "__main__":
    app()
