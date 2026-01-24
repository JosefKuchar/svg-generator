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
        input_dim=11, cond_dim=2, hidden_size=512, num_layers=6, num_heads=8
    )

    wandb_logger = WandbLogger(project="svg-generator")
    wandb_logger.watch(module)

    trainer = pl.Trainer(
        max_epochs=500,
        accelerator="auto",
        gradient_clip_val=1.0,
        logger=wandb_logger,
    )
    trainer.fit(
        module,
        datamodule=DataModule(),
    )


if __name__ == "__main__":
    app()
