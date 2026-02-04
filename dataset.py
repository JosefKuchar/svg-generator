import json
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pytorch_lightning as pl

from representation import BezierPath, BezierShape, shapes_to_tensor
from transformers import AutoImageProcessor
from raster import render_svg


class BezierDataset(Dataset):
    def __init__(self, split="train", max_segments=100):
        self.max_segments = max_segments
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov3-vits16-pretrain-lvd1689m"
        )
        raw_dataset = load_dataset("JosefKuchar/bezier-dataset", split=split)

        # Pre-filter dataset to only include items within max_segments
        self.dataset = raw_dataset.filter(
            lambda item: self._count_curves(item) <= max_segments,
            num_proc=4,
            desc=f"Filtering items with <= {max_segments} curves",
        )

    @staticmethod
    def _count_curves(item):
        """Count total number of curves across all shapes in an item."""
        shapes_data = json.loads(item["shapes"])
        total_curves = 0
        for shape_data in shapes_data:
            for path_data in shape_data["paths"]:
                total_curves += len(path_data["curves"])
        return total_curves

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        shapes_data = json.loads(item["shapes"])
        width = item["width"]
        height = item["height"]

        # Convert JSON shapes back to BezierShape/BezierPath objects
        bezier_shapes = []
        for shape_data in shapes_data:
            # Reconstruct BezierPath objects from serialized paths
            bezier_paths = []
            for path_data in shape_data["paths"]:
                # Convert curves from list format to tuple format
                curves = [
                    tuple(tuple(point) for point in curve)
                    for curve in path_data["curves"]
                ]
                bezier_paths.append(BezierPath(curves))

            # Reconstruct BezierShape with paths, color, and opacity
            bezier_shape = BezierShape(
                paths=bezier_paths,
                color=(
                    tuple(shape_data["color"])
                    if shape_data["color"]
                    else (0.0, 0.0, 0.0)
                ),
                opacity=shape_data["opacity"],
            )
            bezier_shapes.append(bezier_shape)

        # Use shapes_to_tensor from representation.py for conversion
        curve_tensor = shapes_to_tensor(
            bezier_shapes, width, height, max_segments=self.max_segments
        )

        # Render the original svg
        image = render_svg(item["item_svg"]).convert("RGB")

        # Process the image using the DINO image processor
        # torch.Size([1, 3, 224, 224])
        image_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"]

        return curve_tensor, image_tensor


class ValidationSamplingDataset(Dataset):
    """
    A minimal dataset that provides fixed conditioning for validation sampling.
    Uses a fixed seed to ensure reproducible samples across epochs.
    """

    def __init__(self, num_samples=4, cond_dim=384, seed=42):
        self.num_samples = num_samples
        self.cond_dim = cond_dim
        self.seed = seed
        # Generate fixed conditioning using the seed
        generator = torch.Generator().manual_seed(seed)
        self.fixed_cond = torch.zeros(num_samples, 1, cond_dim)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return the fixed conditioning for this sample
        return self.fixed_cond[idx]


class DataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size=256,
        num_workers=10,
        max_segments=100,
        val_num_samples=4,
        cond_dim=384,
        val_seed=42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_segments = max_segments
        self.val_num_samples = val_num_samples
        self.cond_dim = cond_dim
        self.val_seed = val_seed

    def train_dataloader(self):
        dataset = BezierDataset(split="train", max_segments=self.max_segments)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        dataset = ValidationSamplingDataset(
            num_samples=self.val_num_samples,
            cond_dim=self.cond_dim,
            seed=self.val_seed,
        )
        return DataLoader(
            dataset,
            batch_size=self.val_num_samples,
            num_workers=0,
            shuffle=False,
        )
