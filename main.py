import typer
import torch
from typing import Optional
from model import FlowMatchingTransformer
from dataset import DataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


app = typer.Typer()


@app.command()
def train(
    max_samples: Optional[int] = typer.Option(
        None, help="Limit training dataset size (e.g. 32 for overfit test)"
    ),
    train_samples_per_epoch: Optional[int] = typer.Option(
        None,
        min=1,
        help="If set, sample with replacement and use this many train samples per epoch",
    ),
    synthetic: bool = typer.Option(
        False, help="Use synthetic dataset of random geometric shapes"
    ),
    synthetic_length: int = typer.Option(
        100_000, help="Number of samples per epoch for synthetic dataset"
    ),
    synthetic_min_shapes: int = typer.Option(
        1, help="Minimum shapes per synthetic scene"
    ),
    synthetic_max_shapes: int = typer.Option(
        10, help="Maximum shapes per synthetic scene"
    ),
    keep_n_checkpoints: int = typer.Option(
        5,
        min=1,
        help="Keep best N checkpoints by train inference image MSE",
    ),
):
    torch.set_float32_matmul_precision("medium")

    module = FlowMatchingTransformer(
        input_dim=13,
        cond_dim=384,
        hidden_size=768,
        num_layers=16,
        num_heads=12,
        max_len=256,
    )

    wandb_logger = WandbLogger(project="svg-generator")
    wandb_logger.watch(module)

    checkpoint_callback = ModelCheckpoint(
        monitor="train_inference/image_mse",
        mode="min",
        save_top_k=keep_n_checkpoints,
        filename="epoch{epoch:04d}",
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=-1,
        accelerator="auto",
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        module,
        datamodule=DataModule(
            max_segments=256,
            max_samples=max_samples,
            train_samples_per_epoch=train_samples_per_epoch,
            synthetic=synthetic,
            synthetic_length=synthetic_length,
            synthetic_min_shapes=synthetic_min_shapes,
            synthetic_max_shapes=synthetic_max_shapes,
        ),
    )


if __name__ == "__main__":
    app()
