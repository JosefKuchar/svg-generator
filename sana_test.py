import typer
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from parsing import save_bezier_shapes_to_svg
from raster import render_svg_bg
from representation import tensor_to_shapes
from sana_dataset import SanaDataModule
from sana_transformer import SanaTransformer2DModel
from text_encoder import encode_prompts, load_text_encoder


app = typer.Typer()


class StopOnNaNCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, torch.Tensor) and not torch.isfinite(outputs).all():
            trainer.should_stop = True
            raise RuntimeError(
                f"Stopping training: non-finite train loss at step {trainer.global_step}."
            )

    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val/loss")
        if isinstance(val_loss, torch.Tensor) and not torch.isfinite(val_loss).all():
            trainer.should_stop = True
            raise RuntimeError(
                f"Stopping training: non-finite val/loss at step {trainer.global_step}."
            )


class SanaFlowMatching(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 13,
        learning_rate: float = 5e-5,
        warmup_steps: float = 1000.0,
        cond_drop_prob: float = 0.1,
        text_max_length: int = 300,
        sample_steps: int = 30,
        validation_num_images: int = 8,
        validation_seed: int = 42,
        render_size: int = 512,
        reset_model: bool = False,
        train_new_layers_only: bool = False,
        train_first_n_layers: int = 0,
        train_last_n_layers: int = 0,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.cond_drop_prob = cond_drop_prob
        self.text_max_length = text_max_length
        self.sample_steps = sample_steps
        self.validation_num_images = validation_num_images
        self.validation_seed = validation_seed
        self.render_size = render_size
        self.reset_model = reset_model
        self.train_new_layers_only = train_new_layers_only
        self.train_first_n_layers = max(0, int(train_first_n_layers))
        self.train_last_n_layers = max(0, int(train_last_n_layers))

        self.transformer = SanaTransformer2DModel.from_pretrained(
            "Efficient-Large-Model/Sana_600M_512px_diffusers",
            subfolder="transformer",
            in_channels=input_dim,
            out_channels=input_dim,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
        )

        if self.reset_model:
            self._reset_transformer_parameters()
        else:
            self.transformer.x_embedder.reset_parameters()
            self.transformer.proj_out.reset_parameters()

        if (
            self.train_new_layers_only
            or self.train_first_n_layers > 0
            or self.train_last_n_layers > 0
        ):
            self._freeze_for_targeted_training(
                train_first_n_layers=self.train_first_n_layers,
                train_last_n_layers=self.train_last_n_layers,
                include_new_layers=True,
            )

        self._init_unconditional_conditioning()

        self.transformer.train()
        self._logged_image_steps: set[tuple[str, int]] = set()

    @torch.no_grad()
    def _init_unconditional_conditioning(self):
        tokenizer, text_encoder, text_device = load_text_encoder(device="cpu")
        empty_embeds, empty_mask = encode_prompts(
            [""],
            tokenizer,
            text_encoder,
            text_device,
            max_length=self.text_max_length,
        )
        self.register_buffer(
            "empty_prompt_embedding",
            empty_embeds[0].detach().to(dtype=torch.float32, device="cpu"),
            persistent=False,
        )
        self.register_buffer(
            "empty_prompt_attention_mask",
            empty_mask[0].detach().to(dtype=torch.long, device="cpu"),
            persistent=False,
        )
        del text_encoder

    def _get_empty_conditioning(
        self, seq_len: int, embed_dim: int, device: torch.device
    ):
        empty_embedding = self.empty_prompt_embedding
        empty_mask = self.empty_prompt_attention_mask

        if empty_embedding.shape[1] != embed_dim:
            raise ValueError(
                f"Empty prompt embedding dim mismatch: expected {embed_dim}, got {empty_embedding.shape[1]}"
            )

        if empty_embedding.shape[0] != seq_len:
            adjusted_embedding = torch.zeros(
                seq_len,
                embed_dim,
                dtype=empty_embedding.dtype,
                device=empty_embedding.device,
            )
            adjusted_mask = torch.zeros(
                seq_len,
                dtype=empty_mask.dtype,
                device=empty_mask.device,
            )
            copy_len = min(seq_len, empty_embedding.shape[0])
            adjusted_embedding[:copy_len] = empty_embedding[:copy_len]
            adjusted_mask[:copy_len] = empty_mask[:copy_len]
            empty_embedding = adjusted_embedding
            empty_mask = adjusted_mask

        return empty_embedding.to(device=device), empty_mask.to(device=device)

    def _reset_transformer_parameters(self):
        def _reset(module):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.transformer.apply(_reset)

    def _freeze_for_targeted_training(
        self,
        train_first_n_layers: int,
        train_last_n_layers: int,
        include_new_layers: bool,
    ):
        for parameter in self.transformer.parameters():
            parameter.requires_grad = False

        # norm_parameter_keywords = ("norm", "scale_shift_table")
        # for parameter_name, parameter in self.transformer.named_parameters():
        #     if any(keyword in parameter_name.lower() for keyword in norm_parameter_keywords):
        #         parameter.requires_grad = True

        if include_new_layers:
            for parameter in self.transformer.x_embedder.parameters():
                parameter.requires_grad = True

            for parameter in self.transformer.proj_out.parameters():
                parameter.requires_grad = True

        num_blocks = len(self.transformer.transformer_blocks)
        first_n = min(max(0, train_first_n_layers), num_blocks)
        last_n = min(max(0, train_last_n_layers), num_blocks)

        selected_indices = set(range(first_n))
        selected_indices.update(range(max(0, num_blocks - last_n), num_blocks))

        for block_idx, block in enumerate(self.transformer.transformer_blocks):
            if block_idx in selected_indices:
                for parameter in block.parameters():
                    parameter.requires_grad = True

    def forward(self, x, t, cond, cond_attention_mask, mask_cond=None):  # type: ignore[override]
        x = x.float()
        t = t.float()
        cond = cond.float()
        cond_attention_mask = cond_attention_mask.long()

        if mask_cond is not None:
            cond = cond.clone()
            empty_embedding, empty_mask = self._get_empty_conditioning(
                seq_len=cond.shape[1],
                embed_dim=cond.shape[2],
                device=cond.device,
            )
            cond[mask_cond] = empty_embedding
            cond_attention_mask = cond_attention_mask.clone()
            cond_attention_mask[mask_cond] = empty_mask

        return self.transformer(
            hidden_states=x,
            encoder_hidden_states=cond,
            timestep=t,
            encoder_attention_mask=cond_attention_mask,
            return_dict=False,
        )[0]

    def _flow_matching_loss(
        self, curves, cond_embeddings, cond_attention_mask, cond_dropout
    ):
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
        curves, cond_embeddings, cond_attention_mask, _captions = batch
        loss = self._flow_matching_loss(
            curves,
            cond_embeddings,
            cond_attention_mask,
            cond_dropout=True,
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):  # type: ignore[override]
        curves, cond_embeddings, cond_attention_mask, captions = batch
        loss = self._flow_matching_loss(
            curves,
            cond_embeddings,
            cond_attention_mask,
            cond_dropout=False,
        )
        prefix = "val" if dataloader_idx == 0 else "train_inference"
        self.log(f"{prefix}/loss", loss, add_dataloader_idx=False, prog_bar=True)

        should_log = (
            batch_idx == 0
            and (prefix, self.global_step) not in self._logged_image_steps
        )
        if should_log:
            self._logged_image_steps.add((prefix, self.global_step))
            self._log_validation_samples(
                curves,
                cond_embeddings,
                cond_attention_mask,
                captions,
                prefix,
            )

    @torch.no_grad()
    def sample(self, cond_embeddings, cond_attention_mask, shape):
        batch_size = shape[0]
        device = cond_embeddings.device
        generator = torch.Generator(device=device).manual_seed(self.validation_seed)
        x = torch.randn(shape, device=device, generator=generator)

        ts = torch.linspace(0, 1, self.sample_steps + 1, device=device)
        for i in range(self.sample_steps):
            t = ts[i].expand(batch_size)
            dt = ts[i + 1] - ts[i]
            velocity = self(
                x,
                t,
                cond_embeddings,
                cond_attention_mask,
                mask_cond=None,
            )
            x = x + dt * velocity

        return x

    @torch.no_grad()
    def _log_validation_samples(
        self, curves, cond_embeddings, cond_attention_mask, captions, prefix
    ):
        if not isinstance(self.logger, WandbLogger):
            return

        num_images = min(self.validation_num_images, curves.shape[0])
        samples = self.sample(
            cond_embeddings=cond_embeddings,
            cond_attention_mask=cond_attention_mask,
            shape=curves.shape,
        )

        generated_images = []
        target_images = []
        for i in range(num_images):
            try:
                caption_text = str(captions[i]) if i < len(captions) else ""
                sample_shapes = tensor_to_shapes(
                    samples[i].detach().cpu(),
                    self.render_size,
                    self.render_size,
                )
                sample_svg = save_bezier_shapes_to_svg(
                    sample_shapes,
                    self.render_size,
                    self.render_size,
                )
                sample_image = render_svg_bg(sample_svg)
                generated_images.append(
                    wandb.Image(
                        sample_image,
                        caption=f"{prefix} sample {i} | {caption_text}",
                    )
                )

                target_shapes = tensor_to_shapes(
                    curves[i].detach().cpu(),
                    self.render_size,
                    self.render_size,
                )
                target_svg = save_bezier_shapes_to_svg(
                    target_shapes,
                    self.render_size,
                    self.render_size,
                )
                target_image = render_svg_bg(target_svg)
                target_images.append(
                    wandb.Image(
                        target_image,
                        caption=f"{prefix} target {i} | {caption_text}",
                    )
                )
            except Exception as exc:
                print(f"Warning: failed to render {prefix} sample {i}: {exc}")

        if generated_images or target_images:
            log_payload = {"epoch": self.current_epoch}
            if generated_images:
                log_payload[f"{prefix}_samples"] = generated_images
            if target_images:
                log_payload[f"{prefix}_targets"] = target_images
            self.logger.experiment.log(log_payload, step=self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, eps=1e-5
        )

        def lr_lambda(step: int) -> float:
            if self.warmup_steps <= 0:
                return 1.0
            warmup_progress = float(step + 1) / float(self.warmup_steps)
            if warmup_progress > 1.0:
                return 1.0
            return warmup_progress

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


@app.command()
def train(
    dataset_name_or_path=typer.Option(
        "bezier_dataset_with_text_embeddings",
        help="HF dataset id or local save_to_disk directory with text embeddings",
    ),
    max_samples: int | None = typer.Option(
        None,
        min=1,
        help="Limit training dataset size (useful for overfit/debug)",
    ),
    train_samples_per_epoch: int | None = typer.Option(
        None,
        min=1,
        help="If set, sample with replacement and use this many train samples per epoch",
    ),
    learning_rate=typer.Option(5e-5, help="Peak learning rate after warmup"),
    warmup_steps=typer.Option(
        1000,
        min=0,
        help="Number of optimizer steps to linearly ramp learning rate",
    ),
    sample_steps: int = typer.Option(
        30, min=1, help="Sampling ODE steps for validation previews"
    ),
    validation_num_images: int = typer.Option(
        4, min=1, help="How many images to log per validation preview"
    ),
    batch_size: int = typer.Option(64, min=1, help="Per-device train batch size"),
    accumulate_grad_batches: int = typer.Option(
        4, min=1, help="Gradient accumulation steps"
    ),
    validation_interval_steps: int = typer.Option(
        200,
        min=1,
        help="Run validation every N training steps",
    ),
    max_epochs: int = typer.Option(
        -1,
        help="Maximum number of training epochs (-1 for no limit)",
    ),
    reset_model: bool = typer.Option(
        False,
        help="Reset all transformer weights before training (ignore pretrained init)",
    ),
    train_new_layers_only: bool = typer.Option(
        False,
        help="Train only newly initialized layers (x_embedder and proj_out)",
    ),
    train_first_n_layers: int = typer.Option(
        0,
        min=0,
        help="Train first N transformer blocks (in addition to x_embedder/proj_out when targeted training is enabled)",
    ),
    train_last_n_layers: int = typer.Option(
        0,
        min=0,
        help="Train last N transformer blocks (in addition to x_embedder/proj_out when targeted training is enabled)",
    ),
    checkpoint_dir: str = typer.Option(
        "checkpoints/sana",
        help="Directory to save model checkpoints",
    ),
    checkpoint_top_k: int = typer.Option(
        3,
        min=1,
        help="How many best checkpoints to keep based on val/loss",
    ),
    resume_from_ckpt: str | None = typer.Option(
        None,
        help="Path to a Lightning checkpoint (.ckpt) to resume training",
    ),
    init_from_ckpt: str | None = typer.Option(
        None,
        help="Path to checkpoint (.ckpt) to load weights only and start from epoch 0",
    ),
):
    torch.set_float32_matmul_precision("medium")

    learning_rate_value = float(learning_rate)
    warmup_steps_value = float(warmup_steps)

    module = SanaFlowMatching(
        input_dim=13,
        learning_rate=learning_rate_value,
        warmup_steps=warmup_steps_value,
        sample_steps=sample_steps,
        validation_num_images=validation_num_images,
        reset_model=reset_model,
        train_new_layers_only=train_new_layers_only,
        train_first_n_layers=train_first_n_layers,
        train_last_n_layers=train_last_n_layers,
    )

    if init_from_ckpt is not None:
        checkpoint = torch.load(init_from_ckpt, map_location="cpu")
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )
        missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)
        print(
            f"Initialized weights from {init_from_ckpt}. "
            f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}"
        )

    wandb_logger = WandbLogger(project="svg-generator-sana")
    wandb_logger.watch(module)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch{epoch:03d}-step{step:08d}-val_loss{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=checkpoint_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    nan_callback = StopOnNaNCallback()

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        precision="32",
        gradient_clip_val=1.0,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=validation_interval_steps,
        callbacks=[checkpoint_callback, nan_callback],
        logger=wandb_logger,
    )

    trainer.fit(
        module,
        datamodule=SanaDataModule(
            batch_size=batch_size,
            max_segments=256,
            max_samples=max_samples,
            train_samples_per_epoch=train_samples_per_epoch,
            dataset_name_or_path=str(dataset_name_or_path),
        ),
        ckpt_path=resume_from_ckpt,
    )


if __name__ == "__main__":
    app()
