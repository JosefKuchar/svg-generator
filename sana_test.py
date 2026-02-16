import typer
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from sana_dataset import SanaDataModule
from sana_transformer import SanaTransformer2DModel


app = typer.Typer()


class SanaFlowMatching(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 13,
        learning_rate: float = 1e-4,
        cond_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.cond_drop_prob = cond_drop_prob

        self.transformer = SanaTransformer2DModel.from_pretrained(
            "Efficient-Large-Model/Sana_600M_512px_diffusers",
            subfolder="transformer",
            in_channels=input_dim,
            out_channels=input_dim,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
        )

    def forward(self, x, t, cond, cond_attention_mask, mask_cond=None):  # type: ignore[override]
        x = x.float()
        t = t.float()
        cond = cond.float()
        cond_attention_mask = cond_attention_mask.long()

        if mask_cond is not None:
            cond_attention_mask = cond_attention_mask.clone()
            cond_attention_mask[mask_cond] = 0

        return self.transformer(
            hidden_states=x,
            encoder_hidden_states=cond,
            timestep=t,
            encoder_attention_mask=cond_attention_mask,
            return_dict=False,
        )[0]

    def _flow_matching_loss(self, curves, cond_embeddings, cond_attention_mask, cond_dropout):
        t = torch.sigmoid(torch.randn_like(curves[:, 0, 0]))
        x_0 = torch.randn_like(curves)
        t_reshaped = t.view(-1, 1, 1)
        x_t = t_reshaped * curves + (1 - t_reshaped) * x_0
        target_v = curves - x_0

        if cond_dropout:
            mask_cond = torch.rand_like(t) < self.cond_drop_prob
        else:
            mask_cond = None

        pred_v = self(
            x_t,
            t,
            cond_embeddings,
            cond_attention_mask,
            mask_cond=mask_cond,
        )

        return F.mse_loss(pred_v, target_v)

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        curves, cond_embeddings, cond_attention_mask = batch
        loss = self._flow_matching_loss(
            curves,
            cond_embeddings,
            cond_attention_mask,
            cond_dropout=True,
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):  # type: ignore[override]
        curves, cond_embeddings, cond_attention_mask = batch
        loss = self._flow_matching_loss(
            curves,
            cond_embeddings,
            cond_attention_mask,
            cond_dropout=False,
        )
        prefix = "val" if dataloader_idx == 0 else "train_inference"
        self.log(f"{prefix}/loss", loss, add_dataloader_idx=False, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, eps=1e-5)


@app.command()
def train(
    dataset_name_or_path = typer.Option(
        "bezier_dataset_with_text_embeddings",
        help="HF dataset id or local save_to_disk directory with text embeddings",
    ),
    max_samples = typer.Option(
        None,
        help="Limit training dataset size (useful for overfit/debug)",
    ),
    train_samples_per_epoch = typer.Option(
        None,
        min=1,
        help="If set, sample with replacement and use this many train samples per epoch",
    ),
):
    torch.set_float32_matmul_precision("medium")

    module = SanaFlowMatching(input_dim=13)

    wandb_logger = WandbLogger(project="svg-generator-sana")
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
        datamodule=SanaDataModule(
            max_segments=256,
            max_samples=max_samples,
            train_samples_per_epoch=train_samples_per_epoch,
            dataset_name_or_path=str(dataset_name_or_path),
        ),
    )


if __name__ == "__main__":
    app()
