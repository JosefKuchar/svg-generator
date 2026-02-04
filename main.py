import typer
import torch
from model import FlowMatchingTransformer
from dataset import DataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


app = typer.Typer()


@app.command()
def app():
    torch.set_float32_matmul_precision("medium")

    module = FlowMatchingTransformer(
        input_dim=15,
        cond_dim=384,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        max_len=256,
    )

    wandb_logger = WandbLogger(project="svg-generator")
    wandb_logger.watch(module)

    trainer = pl.Trainer(
        max_epochs=-1,
        accelerator="auto",
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        logger=wandb_logger,
    )
    trainer.fit(
        module,
        datamodule=DataModule(max_segments=256),
    )


if __name__ == "__main__":
    app()
