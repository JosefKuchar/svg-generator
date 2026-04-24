import json
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from datasets import load_dataset
import pytorch_lightning as pl

from representation import BezierPath, BezierShape, shapes_to_tensor
from transformers import AutoImageProcessor
from raster import render_svg_bg
from synthetic import SyntheticBezierDataset, SyntheticSamplingDataset


class BezierDataset(Dataset):
    def __init__(self, split="train", max_segments=100, max_samples=None):
        self.max_segments = max_segments
        self.max_samples = max_samples
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov3-vits16-pretrain-lvd1689m"
        )
        raw_dataset = load_dataset("JosefKuchar/bezier-dataset", split=split)

        # Pre-filter dataset to only include items within max_segments and
        # with normalized coordinates in a reasonable range.
        filtered = raw_dataset.filter(
            lambda item: self._is_valid_item(item, max_segments),
            num_proc=4,
            desc=f"Filtering items with <= {max_segments} curves and |coord| <= 2",
        )
        # Limit to max_samples if specified
        if max_samples is not None:
            filtered = filtered.select(range(min(max_samples, len(filtered))))
        self.dataset = filtered

    @staticmethod
    def _count_curves(item):
        """Count total number of curves across all shapes in an item."""
        shapes_data = json.loads(item["shapes"])
        total_curves = 0
        for shape_data in shapes_data:
            for path_data in shape_data["paths"]:
                total_curves += len(path_data["curves"])
        return total_curves

    @staticmethod
    def _normalized_coords_in_range(item, max_abs_value=2.0):
        """Check that all normalized curve coordinates stay within bounds."""
        width = item["width"]
        height = item["height"]
        scale = 2.0 / max(width, height)
        cx = width / 2.0
        cy = height / 2.0

        shapes_data = json.loads(item["shapes"])
        for shape_data in shapes_data:
            for path_data in shape_data["paths"]:
                for curve in path_data["curves"]:
                    for point in curve:
                        x_norm = (point[0] - cx) * scale
                        y_norm = (point[1] - cy) * scale
                        if abs(x_norm) > max_abs_value or abs(y_norm) > max_abs_value:
                            return False
        return True

    @classmethod
    def _is_valid_item(cls, item, max_segments):
        return cls._count_curves(item) <= max_segments and cls._normalized_coords_in_range(
            item
        )

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
        image = render_svg_bg(item["item_svg"]).convert("RGB")

        # Process the image using the DINO image processor
        # torch.Size([1, 3, 224, 224])
        image_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"]

        return curve_tensor, image_tensor


class ValidationSamplingDataset(BezierDataset):
    """
    A dataset for validation sampling that uses the valid split of BezierDataset.
    Limited to a fixed number of samples.
    """

    def __init__(self, num_samples=8, max_segments=100):
        super().__init__(split="valid", max_segments=max_segments)
        self.num_samples = min(num_samples, len(self.dataset))

    def __len__(self):
        return self.num_samples


class TrainSamplingDataset(BezierDataset):
    """
    A dataset for inference sampling on training data.
    Uses the train split, limited to a fixed number of samples.
    """

    def __init__(self, num_samples=8, max_segments=100):
        super().__init__(split="train", max_segments=max_segments)
        self.num_samples = min(num_samples, len(self.dataset))

    def __len__(self):
        return self.num_samples


class DataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size=256,
        num_workers=20,
        max_segments=100,
        val_num_samples=8,
        max_samples=None,
        train_samples_per_epoch=None,
        synthetic=False,
        synthetic_length=100_000,
        synthetic_min_shapes=1,
        synthetic_max_shapes=10,
        synthetic_canvas_size=256,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_segments = max_segments
        self.val_num_samples = val_num_samples
        self.max_samples = max_samples
        self.train_samples_per_epoch = train_samples_per_epoch
        self.synthetic = synthetic
        self.synthetic_length = synthetic_length
        self.synthetic_min_shapes = synthetic_min_shapes
        self.synthetic_max_shapes = synthetic_max_shapes
        self.synthetic_canvas_size = synthetic_canvas_size
        self.synthetic_epoch = 0
        self._train_synthetic_dataset = None

    def set_synthetic_epoch(self, epoch: int):
        self.synthetic_epoch = epoch
        if self._train_synthetic_dataset is not None:
            self._train_synthetic_dataset.set_epoch(epoch)

    def train_dataloader(self):
        if self.synthetic:
            if self._train_synthetic_dataset is None:
                self._train_synthetic_dataset = SyntheticBezierDataset(
                    length=self.synthetic_length,
                    canvas_size=self.synthetic_canvas_size,
                    max_segments=self.max_segments,
                    min_shapes=self.synthetic_min_shapes,
                    max_shapes=self.synthetic_max_shapes,
                )
            self._train_synthetic_dataset.set_epoch(self.synthetic_epoch)
            dataset = self._train_synthetic_dataset
        else:
            dataset = BezierDataset(
                split="train",
                max_segments=self.max_segments,
                max_samples=self.max_samples,
            )
        if self.train_samples_per_epoch is not None:
            sampler = RandomSampler(
                dataset,
                replacement=True,
                num_samples=self.train_samples_per_epoch,
            )
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                sampler=sampler,
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        if self.synthetic:
            val_dataset = ValidationSamplingDataset(
                num_samples=self.val_num_samples,
                max_segments=self.max_segments,
            )
            train_sample_dataset = SyntheticSamplingDataset(
                num_samples=self.val_num_samples,
                canvas_size=self.synthetic_canvas_size,
                max_segments=self.max_segments,
                min_shapes=self.synthetic_min_shapes,
                max_shapes=self.synthetic_max_shapes,
                base_seed=888_888,
            )
        else:
            val_dataset = ValidationSamplingDataset(
                num_samples=self.val_num_samples,
                max_segments=self.max_segments,
            )
            train_sample_dataset = TrainSamplingDataset(
                num_samples=self.val_num_samples,
                max_segments=self.max_segments,
            )
        loader_kwargs = dict(
            batch_size=self.val_num_samples,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return [
            DataLoader(val_dataset, **loader_kwargs),
            DataLoader(train_sample_dataset, **loader_kwargs),
        ]
