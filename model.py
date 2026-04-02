"""Autoregressive image-conditioned bezier transformer."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import wandb
from PIL import Image
from transformers import AutoModel

from parsing import save_bezier_shapes_to_svg
from raster import calculate_mse, render_svg_bg
from representation import tensor_to_shapes


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 256, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.size(1):
            self._build_cache(seq_len)
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim > mask.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        batch_size, seq_len, hidden_size = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.out_proj(attn)


class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        batch_size, seq_len, hidden_size = x.shape
        cond_len = cond.shape[1]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(cond).view(batch_size, cond_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(cond).view(batch_size, cond_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.out_proj(attn)


class DecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = SelfAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.cross_attn = CrossAttention(hidden_size, num_heads, dropout)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        x = x + self.self_attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.cross_attn(self.norm2(x), cond)
        x = x + self.mlp(self.norm3(x))
        return x


class AutoregressiveTransformer(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_size: int = 512,
        max_len: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.image_encoder = AutoModel.from_pretrained(
            "facebook/dinov3-vits16-pretrain-lvd1689m",
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()

        self.token_embedder = nn.Linear(input_dim, hidden_size)
        self.cond_embedder = nn.Linear(cond_dim, hidden_size)
        self.bos_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.rope = RotaryPositionEmbedding(hidden_size // num_heads, max_len=max_len)
        self.blocks = nn.ModuleList(
            [DecoderBlock(hidden_size, num_heads, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.continuous_head = nn.Linear(hidden_size, 10)
        self.flag_head = nn.Linear(hidden_size, 3)

        self._val_cond_images_logged = False
        self._train_inference_cond_images_logged = False

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.squeeze(1).to(self.device)
        with torch.no_grad():
            cond = self.image_encoder(pixel_values=images).last_hidden_state
        return cond.float()

    def _decoder_inputs(self, target_tokens: torch.Tensor) -> torch.Tensor:
        bos = self.bos_token.expand(target_tokens.shape[0], -1, -1)
        embedded = self.token_embedder(target_tokens.float())
        return torch.cat([bos, embedded[:, :-1]], dim=1)

    def forward(self, decoder_inputs: torch.Tensor, cond: torch.Tensor):
        x = decoder_inputs.float()
        cond = self.cond_embedder(cond.float())
        rope_cos, rope_sin = self.rope(x.shape[1])
        for block in self.blocks:
            x = block(x, cond, rope_cos, rope_sin)
        x = self.final_norm(x)
        continuous = torch.tanh(self.continuous_head(x))
        flags = self.flag_head(x)
        return continuous, flags

    def _loss_and_outputs(self, batch):
        target_tokens, images = batch
        cond = self._encode_images(images)
        decoder_inputs = self._decoder_inputs(target_tokens)
        continuous_pred, flag_logits = self(decoder_inputs, cond)

        continuous_target = target_tokens[..., :10]
        flag_target = (target_tokens[..., 10:] > 0).float()
        real_mask = flag_target[..., 2] > 0.5

        coord_loss = masked_mean(
            (continuous_pred[..., :6] - continuous_target[..., :6]) ** 2,
            real_mask,
        )
        style_loss = masked_mean(
            (continuous_pred[..., 6:10] - continuous_target[..., 6:10]) ** 2,
            real_mask,
        )
        real_loss = F.binary_cross_entropy_with_logits(
            flag_logits[..., 2],
            flag_target[..., 2],
        )

        structure_mask = real_mask.unsqueeze(-1).expand(-1, -1, 2)
        structure_loss = masked_mean(
            F.binary_cross_entropy_with_logits(
                flag_logits[..., :2],
                flag_target[..., :2],
                reduction="none",
            ),
            structure_mask,
        )

        loss = coord_loss + style_loss + real_loss + structure_loss
        return {
            "loss": loss,
            "coord_loss": coord_loss,
            "style_loss": style_loss,
            "real_loss": real_loss,
            "structure_loss": structure_loss,
            "continuous_pred": continuous_pred,
            "flag_logits": flag_logits,
            "target_tokens": target_tokens,
        }

    def _compose_predictions(self, continuous_pred: torch.Tensor, flag_logits: torch.Tensor) -> torch.Tensor:
        binary_flags = torch.where(flag_logits > 0, torch.ones_like(flag_logits), -torch.ones_like(flag_logits))
        return torch.cat([continuous_pred, binary_flags], dim=-1)

    def training_step(self, batch, batch_idx):
        outputs = self._loss_and_outputs(batch)
        self.log("train/loss", outputs["loss"])
        self.log("train/coord_loss", outputs["coord_loss"])
        self.log("train/style_loss", outputs["style_loss"])
        self.log("train/real_loss", outputs["real_loss"])
        self.log("train/structure_loss", outputs["structure_loss"])
        return outputs["loss"]

    def on_train_epoch_start(self):
        datamodule = self.trainer.datamodule
        if hasattr(datamodule, "set_synthetic_epoch"):
            datamodule.set_synthetic_epoch(self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, eps=1e-5)

    def _compute_generation_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        target_real = targets[..., 12] > 0
        pred_real = predictions[..., 12] > 0
        real_union = target_real | pred_real
        real_intersection = target_real & pred_real

        metrics = {
            "segment_real_accuracy": (pred_real == target_real).float().mean().item(),
            "segment_real_iou": (real_intersection.float().sum() / real_union.float().sum().clamp_min(1.0)).item(),
            "length_mae": (pred_real.sum(dim=1).float() - target_real.sum(dim=1).float()).abs().mean().item(),
        }

        if target_real.any():
            metrics["coord_mse"] = ((predictions[..., :6] - targets[..., :6]) ** 2)[target_real].mean().item()
            metrics["style_mse"] = ((predictions[..., 6:10] - targets[..., 6:10]) ** 2)[target_real].mean().item()
            metrics["path_start_accuracy"] = ((predictions[..., 10] > 0) == (targets[..., 10] > 0))[target_real].float().mean().item()
            metrics["subpath_start_accuracy"] = ((predictions[..., 11] > 0) == (targets[..., 11] > 0))[target_real].float().mean().item()
        else:
            metrics["coord_mse"] = 0.0
            metrics["style_mse"] = 0.0
            metrics["path_start_accuracy"] = 0.0
            metrics["subpath_start_accuracy"] = 0.0

        return metrics

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        prefix = "val" if dataloader_idx == 0 else "train_inference"
        outputs = self._loss_and_outputs(batch)
        self.log(f"{prefix}/loss", outputs["loss"], add_dataloader_idx=False)
        self.log(f"{prefix}/coord_loss", outputs["coord_loss"], add_dataloader_idx=False)
        self.log(f"{prefix}/style_loss", outputs["style_loss"], add_dataloader_idx=False)
        self.log(f"{prefix}/real_loss", outputs["real_loss"], add_dataloader_idx=False)
        self.log(f"{prefix}/structure_loss", outputs["structure_loss"], add_dataloader_idx=False)
        self._run_inference_step(batch, prefix)

    def _run_inference_step(self, batch, prefix: str):
        validation_seed = 42
        img_size = 512
        target_curves, images_input = batch
        images_input = images_input.squeeze(1)
        num_samples = images_input.shape[0]
        cond = self._encode_images(images_input)
        samples = self.sample(
            cond,
            shape=(num_samples, target_curves.shape[1], self.hparams.input_dim),
            seed=validation_seed,
        )

        curve_l2_mse = F.mse_loss(samples, target_curves)
        self.log(f"{prefix}/curve_l2_mse", curve_l2_mse, add_dataloader_idx=False)

        metrics = self._compute_generation_metrics(samples, target_curves)
        for metric_name, metric_value in metrics.items():
            self.log(f"{prefix}/{metric_name}", metric_value, add_dataloader_idx=False)

        generated_images = []
        cond_images = []
        mse_values = []
        cond_logged_attr = f"_{prefix}_cond_images_logged"
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for i in range(num_samples):
            sample_tensor = samples[i]
            cond_img = images_input[i].cpu()
            cond_img = cond_img * std + mean
            cond_img = (cond_img.clamp(0, 1) * 255).to(torch.uint8)
            cond_img_np = cond_img.permute(1, 2, 0).numpy()
            cond_pil = Image.fromarray(cond_img_np)

            if not getattr(self, cond_logged_attr, False):
                cond_images.append(wandb.Image(cond_img_np, caption=f"Conditioning {i}"))

            try:
                shapes = tensor_to_shapes(sample_tensor, img_size, img_size)
                svg_content = save_bezier_shapes_to_svg(shapes, img_size, img_size)
                image = render_svg_bg(svg_content)
                generated_images.append(wandb.Image(image, caption=f"Sample {i}"))
                mse_values.append(calculate_mse(cond_pil, image.resize(cond_pil.size, Image.LANCZOS)))
            except Exception as exc:
                print(f"Warning: Failed to render sample {i} at epoch {self.current_epoch}: {exc}")

        if mse_values:
            self.log(f"{prefix}/image_mse", sum(mse_values) / len(mse_values), add_dataloader_idx=False)

        log_dict = {"epoch": self.current_epoch}
        if generated_images:
            log_dict[f"{prefix}_samples"] = generated_images
        if cond_images:
            log_dict[f"{prefix}_conditioning"] = cond_images
            setattr(self, cond_logged_attr, True)
        if (generated_images or cond_images) and self.logger is not None:
            self.logger.experiment.log(log_dict, step=self.global_step)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps=50, cfg_scale=1.0, shape=None, seed=None):
        del steps, cfg_scale, seed
        self.eval()
        batch_size = cond.shape[0]
        if shape is None:
            max_len = self.hparams.max_len
        else:
            max_len = shape[1]

        generated_tokens = torch.zeros(
            batch_size,
            max_len,
            self.hparams.input_dim,
            device=cond.device,
            dtype=cond.dtype,
        )
        finished = torch.zeros(batch_size, device=cond.device, dtype=torch.bool)

        for position in range(max_len):
            decoder_inputs = self._decoder_inputs(generated_tokens)
            continuous_pred, flag_logits = self(decoder_inputs[:, : position + 1], cond)
            next_token = self._compose_predictions(
                continuous_pred[:, -1:],
                flag_logits[:, -1:],
            )
            next_is_padding = next_token[:, 0, 12] <= 0
            next_token[:, 0, :10] = torch.where(
                next_is_padding.unsqueeze(-1),
                torch.zeros_like(next_token[:, 0, :10]),
                next_token[:, 0, :10],
            )
            next_token[:, 0, 10:12] = torch.where(
                next_is_padding.unsqueeze(-1),
                -torch.ones_like(next_token[:, 0, 10:12]),
                next_token[:, 0, 10:12],
            )
            next_token[:, 0] = torch.where(
                finished.unsqueeze(-1),
                torch.tensor([0.0] * 12 + [-1.0], device=cond.device, dtype=cond.dtype),
                next_token[:, 0],
            )
            generated_tokens[:, position : position + 1] = next_token
            finished |= next_is_padding

        return generated_tokens


FlowMatchingTransformer = AutoregressiveTransformer
