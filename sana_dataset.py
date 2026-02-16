import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset, RandomSampler

from representation import BezierPath, BezierShape, shapes_to_tensor


class SanaBezierDataset(Dataset):
    def __init__(
        self,
        split="train",
        max_segments=100,
        max_samples=None,
        dataset_name_or_path="bezier_dataset_with_text_embeddings",
        caption_column="caption_texts",
    ):
        self.max_segments = max_segments
        self.max_samples = max_samples
        self.caption_column = caption_column

        dataset_path = Path(dataset_name_or_path)
        if dataset_path.exists() and dataset_path.is_dir():
            raw_dataset = load_from_disk(str(dataset_path))[split]
        else:
            raw_dataset = load_dataset(dataset_name_or_path, split=split)

        required_columns = {"text_embeddings", "text_attention_mask", "shapes"}
        missing_columns = required_columns.difference(raw_dataset.column_names)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                "Dataset is missing required precomputed text-conditioning columns: "
                f"{missing}."
            )
        if max_samples is not None:
            raw_dataset = raw_dataset.select(range(min(max_samples, len(raw_dataset))))
        self.dataset = raw_dataset

    @staticmethod
    def _shapes_from_json(shapes_data):
        bezier_shapes = []
        for shape_data in shapes_data:
            bezier_paths = []
            for path_data in shape_data["paths"]:
                curves = [
                    tuple(tuple(point) for point in curve)
                    for curve in path_data["curves"]
                ]
                bezier_paths.append(BezierPath(curves))

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
        return bezier_shapes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        shapes_data = json.loads(item["shapes"])
        width = item["width"]
        height = item["height"]

        bezier_shapes = self._shapes_from_json(shapes_data)
        curve_tensor = shapes_to_tensor(
            bezier_shapes,
            width,
            height,
            max_segments=self.max_segments,
        )

        text_embeddings = torch.as_tensor(item["text_embeddings"], dtype=torch.float32)
        text_attention_mask = torch.as_tensor(
            item["text_attention_mask"], dtype=torch.long
        )

        caption_text = ""
        if self.caption_column in item:
            captions = item[self.caption_column]
            if isinstance(captions, list) and captions:
                first = captions[0]
                caption_text = first if isinstance(first, str) else ""
            elif isinstance(captions, str):
                caption_text = captions

        return curve_tensor, text_embeddings, text_attention_mask, caption_text


def sana_bezier_collate_fn(batch):
    curve_tensors, text_embeddings, text_attention_masks, captions = zip(*batch)

    curves = torch.stack(curve_tensors)
    max_text_len = max(embedding.shape[0] for embedding in text_embeddings)
    embed_dim = text_embeddings[0].shape[1]

    embeddings = torch.zeros(
        len(text_embeddings),
        max_text_len,
        embed_dim,
        dtype=text_embeddings[0].dtype,
    )
    attention_mask = torch.zeros(
        len(text_attention_masks),
        max_text_len,
        dtype=text_attention_masks[0].dtype,
    )

    for i, (embedding, mask) in enumerate(zip(text_embeddings, text_attention_masks)):
        seq_len = embedding.shape[0]
        embeddings[i, -seq_len:] = embedding
        attention_mask[i, -seq_len:] = mask

    return curves, embeddings, attention_mask, list(captions)


class SanaValidationSamplingDataset(SanaBezierDataset):
    def __init__(
        self,
        num_samples=8,
        max_segments=100,
        dataset_name_or_path="bezier_dataset_with_text_embeddings",
    ):
        super().__init__(
            split="valid",
            max_segments=max_segments,
            dataset_name_or_path=dataset_name_or_path,
        )
        self.num_samples = min(num_samples, len(self.dataset))

    def __len__(self):
        return self.num_samples


class SanaTrainSamplingDataset(SanaBezierDataset):
    def __init__(
        self,
        num_samples=8,
        max_segments=100,
        dataset_name_or_path="bezier_dataset_with_text_embeddings",
    ):
        super().__init__(
            split="train",
            max_segments=max_segments,
            dataset_name_or_path=dataset_name_or_path,
        )
        self.num_samples = min(num_samples, len(self.dataset))

    def __len__(self):
        return self.num_samples


class SanaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=256,
        num_workers=20,
        max_segments=100,
        val_num_samples=8,
        max_samples=None,
        train_samples_per_epoch=None,
        dataset_name_or_path="bezier_dataset_with_text_embeddings",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_segments = max_segments
        self.val_num_samples = val_num_samples
        self.max_samples = max_samples
        self.train_samples_per_epoch = train_samples_per_epoch
        self.dataset_name_or_path = dataset_name_or_path

    def train_dataloader(self):
        dataset = SanaBezierDataset(
            split="train",
            max_segments=self.max_segments,
            max_samples=self.max_samples,
            dataset_name_or_path=self.dataset_name_or_path,
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
                collate_fn=sana_bezier_collate_fn,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=sana_bezier_collate_fn,
        )

    def val_dataloader(self):
        val_dataset = SanaValidationSamplingDataset(
            num_samples=self.val_num_samples,
            max_segments=self.max_segments,
            dataset_name_or_path=self.dataset_name_or_path,
        )
        train_sample_dataset = SanaTrainSamplingDataset(
            num_samples=self.val_num_samples,
            max_segments=self.max_segments,
            dataset_name_or_path=self.dataset_name_or_path,
        )
        loader_kwargs = dict(
            batch_size=self.val_num_samples,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=sana_bezier_collate_fn,
        )
        return [
            DataLoader(val_dataset, **loader_kwargs),
            DataLoader(train_sample_dataset, **loader_kwargs),
        ]
