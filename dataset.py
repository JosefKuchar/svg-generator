import json
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pytorch_lightning as pl


class BezierDataset(Dataset):
    def __init__(self, split="train", max_segments=100):
        self.dataset = load_dataset("JosefKuchar/bezier-dataset", split=split)
        self.max_segments = max_segments

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        shapes = json.loads(item["shapes"])
        width = item["width"]
        height = item["height"]

        # Normalization parameters (same as parsing.py)
        cx = width / 2.0
        cy = height / 2.0
        scale = 2.0 / max(width, height)

        def norm_point(x, y):
            return (x - cx) * scale, (y - cy) * scale

        # Convert shapes to normalized curves
        all_curves = []
        for shape in shapes:
            curves = shape["curves"]
            normalized_curve = []
            for curve in curves:
                # curve is [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
                p0 = norm_point(curve[0][0], curve[0][1])
                p1 = norm_point(curve[1][0], curve[1][1])
                p2 = norm_point(curve[2][0], curve[2][1])
                p3 = norm_point(curve[3][0], curve[3][1])
                normalized_curve.append((p0, p1, p2, p3))
            if normalized_curve:
                all_curves.append(normalized_curve)

        curve_tensor = curves_to_tensor(all_curves, max_segments=self.max_segments)

        # Conditioning tensor (placeholder for now)
        cond_tensor = torch.tensor([[0.0, 0.0]]).float()

        return curve_tensor, cond_tensor


class ValidationSamplingDataset(Dataset):
    """
    A minimal dataset that provides fixed conditioning for validation sampling.
    Uses a fixed seed to ensure reproducible samples across epochs.
    """

    def __init__(self, num_samples=4, cond_dim=2, seed=42):
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
        batch_size=512,
        num_workers=10,
        max_segments=100,
        val_num_samples=4,
        cond_dim=2,
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
