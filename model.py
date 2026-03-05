"""
Model implementation
"""

import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import wandb
from transformers import AutoModel, BitsAndBytesConfig
from representation import tensor_to_shapes
from parsing import save_bezier_shapes_to_svg
from PIL import Image
from raster import render_svg, render_svg_bg, calculate_mse

try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None  # type: ignore


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for transformers.
    Applies rotation to query/key pairs based on position.
    """

    def __init__(self, dim, max_len=256, base=10000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin cache
        self._build_cache(max_len)

    def _build_cache(self, seq_len):
        t = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        # Stack cos and sin for interleaved application
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0), persistent=False)

    def forward(self, seq_len):
        """Return cos and sin for the given sequence length."""
        if seq_len > self.cos_cached.size(1):
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :seq_len, :],
            self.sin_cached[:, :seq_len, :],
        )


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to query and key tensors."""
    # q, k: [Batch, NumHeads, SeqLen, HeadDim]
    # cos, sin: [1, SeqLen, HeadDim]
    cos = cos.unsqueeze(1)  # [1, 1, SeqLen, HeadDim]
    sin = sin.unsqueeze(1)  # [1, 1, SeqLen, HeadDim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into a high-dimensional vector using sine/cosine features.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class AdaLNBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        # Self-Attention projections (for RoPE-based attention)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)

        # Cross-Attention (no RoPE, using flash attention when available)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_q_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_k_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_v_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_out_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_dropout = nn.Dropout(dropout)

        # Feed Forward
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

        # AdaLN Modulation: Predicts shift (beta), scale (gamma), and gate for all 3 sublayers
        # Output dim = 9 * hidden_size: (shift, scale, gate) x 3 sublayers
        # Gates are crucial for training stability (typically zero-initialized)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
        # Zero-initialize the gate portions of the linear layer for stability
        nn.init.zeros_(self.adaLN_modulation[1].weight[6 * hidden_size :])
        nn.init.zeros_(self.adaLN_modulation[1].bias[6 * hidden_size :])

    def _rope_attention(self, x, cos, sin):
        """Self-attention with RoPE."""
        B, S, D = x.shape

        if FLASH_ATTN_AVAILABLE:
            # Flash Attention expects [batch, seq_len, num_heads, head_dim] format
            q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim)

            # Apply RoPE to Q and K
            # Need to reshape for RoPE application: [batch, num_heads, seq_len, head_dim]
            q_rope = q.transpose(1, 2)
            k_rope = k.transpose(1, 2)
            q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
            q = q_rope.transpose(1, 2)
            k = k_rope.transpose(1, 2)

            # Flash Attention requires fp16 or bf16
            original_dtype = q.dtype
            if original_dtype not in (torch.float16, torch.bfloat16):
                q = q.to(torch.bfloat16)
                k = k.to(torch.bfloat16)
                v = v.to(torch.bfloat16)

            attn_out = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                causal=False,
            )
            attn_out = attn_out.to(original_dtype).view(B, S, D)
        else:
            # Standard attention implementation
            q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

            # Apply RoPE to Q and K
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Scaled dot-product attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            # Compute attention output
            attn_out = torch.matmul(attn_weights, v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        return self.out_proj(attn_out)

    def _cross_attention(self, q_input, kv_input):
        """Cross-attention with flash attention support.

        Args:
            q_input: Query input (flow sequence) [Batch, SeqLen_Q, Dim]
            kv_input: Key/Value input (conditioning) [Batch, SeqLen_KV, Dim]
        """
        B, S_q, D = q_input.shape
        S_kv = kv_input.shape[1]

        if FLASH_ATTN_AVAILABLE:
            # Flash Attention expects [batch, seq_len, num_heads, head_dim] format
            q = self.cross_q_proj(q_input).view(B, S_q, self.num_heads, self.head_dim)
            k = self.cross_k_proj(kv_input).view(B, S_kv, self.num_heads, self.head_dim)
            v = self.cross_v_proj(kv_input).view(B, S_kv, self.num_heads, self.head_dim)

            # Flash Attention requires fp16 or bf16
            original_dtype = q.dtype
            if original_dtype not in (torch.float16, torch.bfloat16):
                q = q.to(torch.bfloat16)
                k = k.to(torch.bfloat16)
                v = v.to(torch.bfloat16)

            attn_out = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.cross_attn_dropout.p if self.training else 0.0,
                causal=False,
            )
            attn_out = attn_out.to(original_dtype).view(B, S_q, D)
        else:
            # Standard cross-attention implementation
            q = (
                self.cross_q_proj(q_input)
                .view(B, S_q, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = (
                self.cross_k_proj(kv_input)
                .view(B, S_kv, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.cross_v_proj(kv_input)
                .view(B, S_kv, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

            # Scaled dot-product attention
            # q: [B, num_heads, S_q, head_dim], k: [B, num_heads, S_kv, head_dim]
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.cross_attn_dropout(attn_weights)

            # Compute attention output
            attn_out = torch.matmul(attn_weights, v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S_q, D)

        return self.cross_out_proj(attn_out)

    def forward(self, x, c, t_emb, rope_cos, rope_sin):
        """
        x: Input sequence (Flow) [Batch, SeqLen, Dim]
        c: Conditioning sequence [Batch, CondSeqLen, Dim]
        t_emb: Timestep embedding [Batch, Dim]
        rope_cos: RoPE cosine [1, SeqLen, HeadDim]
        rope_sin: RoPE sine [1, SeqLen, HeadDim]
        """
        # 1. Regress modulation parameters from time embedding
        # For each sublayer: (shift, scale, gate) - total 9 parameters
        params = self.adaLN_modulation(t_emb).chunk(9, dim=1)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = params

        # Helper for modulation
        def modulate(x, shift, scale):
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # 2. Self-Attention Block with RoPE
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self._rope_attention(x_norm, rope_cos, rope_sin)

        # 3. Cross-Attention Block (no RoPE for cross-attention)
        x_norm = modulate(self.norm2(x), shift_cross, scale_cross)
        # Query = Flow (x), Key/Value = Conditioning (c)
        x = x + gate_cross.unsqueeze(1) * self._cross_attention(x_norm, c)

        # 4. Feed-Forward Block
        x_norm = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class FlowMatchingTransformer(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_size: int = 512,
        max_len: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: int = 0.1,
        cond_drop_prob: float = 0.1,  # Probability to drop conditioning
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cond_drop_prob = cond_drop_prob

        self.image_encoder = AutoModel.from_pretrained(
            "facebook/dinov3-vits16-pretrain-lvd1689m",
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()

        # 1. Embeddings
        self.x_embedder = nn.Linear(input_dim, hidden_size)
        self.c_embedder = nn.Linear(cond_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # RoPE operates on head_dim, not full hidden_size
        head_dim = hidden_size // num_heads
        self.rope = RotaryPositionEmbedding(head_dim, max_len=max_len)

        # 2. Null Conditioning (Learnable vector for classifier-free guidance)
        # We learn a single token and broadcast it to the sequence length
        self.null_cond_emb = nn.Parameter(torch.randn(1, 1, hidden_size))

        # 3. Transformer Backbone
        self.blocks = nn.ModuleList(
            [AdaLNBlock(hidden_size, num_heads, dropout) for _ in range(num_layers)]
        )

        # 4. Final Layer (Standard DiT final block)
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.final_proj = nn.Linear(hidden_size, input_dim)

        # Zero-init the final projection to start as an identity function
        nn.init.constant_(self.final_proj.weight, 0)
        nn.init.constant_(self.final_proj.bias, 0)

        # Flags to track if conditioning images have been logged (per prefix)
        self._val_cond_images_logged = False
        self._train_inference_cond_images_logged = False

    def forward(self, x, t, c, mask_cond=None):
        """
        x: Noisy input [Batch, SeqLen, Dim]
        t: Timesteps [Batch]
        c: Conditioning Vector Sequence [Batch, CondLen, CondDim]
        mask_cond: Boolean tensor [Batch]. If True, drop conditioning for that sample.
        """
        x = x.float()
        t = t.float()
        c = c.float()

        # Embedding inputs in explicit fp32
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.x_embedder(x)
            c = self.c_embedder(c)
            t_emb = self.t_embedder(t)

        # Get RoPE embeddings for the sequence
        seq_len = x.size(1)
        rope_cos, rope_sin = self.rope(seq_len)

        # Handle Null Conditioning (Dropout)
        if mask_cond is not None:
            # mask_cond is [Batch], True means drop
            # We construct a batch of null tokens
            null_emb_expanded = self.null_cond_emb.expand(c.shape[0], c.shape[1], -1)

            # Use where to swap: if mask_cond is True, use null, else use c
            # Need to reshape mask for broadcasting: [Batch, 1, 1]
            mask_reshaped = mask_cond.view(-1, 1, 1).float()
            c = (1.0 - mask_reshaped) * c + mask_reshaped * null_emb_expanded

        # Transformer Pass
        for block in self.blocks:
            x = block(x, c, t_emb, rope_cos, rope_sin)

        # Final AdaLN & Projection in explicit fp32
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            t_emb_fp32 = t_emb.float()
            shift, scale = self.final_adaLN(t_emb_fp32).chunk(2, dim=1)
            x_fp32 = self.final_norm(x_fp32) * (
                1 + scale.unsqueeze(1)
            ) + shift.unsqueeze(1)
            output = F.linear(
                x_fp32,
                self.final_proj.weight.float(),
                (
                    self.final_proj.bias.float()
                    if self.final_proj.bias is not None
                    else None
                ),
            )

        return output

    def training_step(self, batch, batch_idx):
        # x_1: Real Data
        # cond: Conditioning
        x_1, images = batch
        images = images.squeeze(1)  # [B, 1, 3, H, W] -> [B, 3, H, W]
        batch_size = x_1.size(0)
        device = x_1.device

        # Use no_grad (not inference_mode) so cond can still be used in autograd graph
        # for downstream trainable layers (e.g., c_embedder weight gradients).
        with torch.no_grad():
            cond = self.image_encoder(pixel_values=images).last_hidden_state

        # 1. Sample Time t ~ Logit-Normal
        # Sample from normal, then apply sigmoid to get t in [0, 1]
        # This biases t toward 0.5, improving training dynamics
        t = torch.sigmoid(torch.randn(batch_size, device=device))

        # 2. Sample Noise x_0 ~ N(0, 1)
        x_0 = torch.randn_like(x_1)

        # 3. Rectified Flow Interpolation
        # Formula: x_t = t * x_1 + (1 - t) * x_0
        t_reshaped = t.view(-1, 1, 1)
        x_t = t_reshaped * x_1 + (1 - t_reshaped) * x_0

        # 4. Target Velocity
        # The vector that points directly from noise to data
        target_v = x_1 - x_0

        # 5. Classifier-Free Guidance Masking
        mask_cond = torch.rand(batch_size, device=device) < self.cond_drop_prob

        # 6. Predict and Loss
        pred_v = self(x_t, t, cond, mask_cond=mask_cond)

        # 7. Loss computation (MSE on all tokens including padding)
        loss = F.mse_loss(pred_v, target_v)

        self.log("train_loss", loss)
        return loss

    def on_train_epoch_start(self):
        datamodule = self.trainer.datamodule
        if hasattr(datamodule, "set_synthetic_epoch"):
            datamodule.set_synthetic_epoch(self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, eps=1e-5
        )

    def _compute_validation_metrics(self, samples: torch.Tensor) -> dict:
        """
        Compute validation metrics for generated samples.

        Args:
            samples: Tensor of shape [batch, seq_len, input_dim]
                     where input_dim = 13 (x0,y0,x1,y1,x2,y2,r,g,b,opacity,path_start,subpath_start,real)

        Returns:
            Dictionary with computed metrics
        """
        metrics = {}

        # 1. Flag closeness to 1 or -1
        # Flags are at indices 10 (path_start), 11 (subpath_start), 12 (real)
        # path_start/subpath_start are only meaningful on real segments, while real is
        # evaluated on all positions (including padding slots).
        real_mask = samples[..., 12] > 0
        flag_indices = [10, 11, 12]
        flag_names = ["path_start", "subpath_start", "real"]
        flag_metric_values = []

        for idx, name in zip(flag_indices, flag_names):
            flag_values = samples[..., idx]  # [batch, seq_len]
            # Closeness = 1 - min(|value - 1|, |value + 1|)
            # Perfect flags have closeness = 1, random noise has closeness ~ 0
            dist_to_pos1 = torch.abs(flag_values - 1.0)
            dist_to_neg1 = torch.abs(flag_values + 1.0)
            min_dist = torch.minimum(dist_to_pos1, dist_to_neg1)
            closeness = torch.clamp(1.0 - min_dist, 0.0, 1.0)

            if name == "real":
                metric_val = closeness.mean().item()
            else:
                if real_mask.any():
                    metric_val = closeness[real_mask].mean().item()
                else:
                    metric_val = 0.0

            metrics[f"flag_{name}_closeness"] = metric_val
            flag_metric_values.append(metric_val)

        metrics["flag_closeness_avg"] = sum(flag_metric_values) / len(
            flag_metric_values
        )

        # 2. Color and opacity consistency per shape
        # Colors are at indices 6 (r), 7 (g), 8 (b), 9 (opacity)
        color_opacity_stds = []

        for batch_idx in range(samples.shape[0]):
            sample = samples[batch_idx]  # [seq_len, input_dim]

            # Find shape boundaries using path_start flag (threshold at 0)
            path_starts = sample[:, 10] > 0  # [seq_len]
            real_flags = sample[:, 12] > 0  # [seq_len]

            # Get indices where new shapes start
            shape_start_indices = torch.where(path_starts & real_flags)[0]

            if len(shape_start_indices) > 0:
                # Add end index for the last shape
                shape_ranges = []
                for i, start_idx in enumerate(shape_start_indices):
                    if i + 1 < len(shape_start_indices):
                        end_idx = shape_start_indices[i + 1]
                    else:
                        # Find the last real segment for the last shape
                        end_idx = sample.shape[0]
                    shape_ranges.append((start_idx.item(), end_idx))

                # Compute std for each shape
                for start_idx, end_idx in shape_ranges:
                    shape_segment = sample[start_idx:end_idx]
                    # Filter only real segments within this shape
                    real_mask = shape_segment[:, 12] > 0
                    real_segments = shape_segment[real_mask]

                    if len(real_segments) > 1:
                        # Compute std for each color channel and opacity
                        r_std = real_segments[:, 6].std().item()
                        g_std = real_segments[:, 7].std().item()
                        b_std = real_segments[:, 8].std().item()
                        opacity_std = real_segments[:, 9].std().item()
                        color_opacity_stds.append(
                            (r_std + g_std + b_std + opacity_std) / 4
                        )

        if color_opacity_stds:
            metrics["color_opacity_std_avg"] = sum(color_opacity_stds) / len(
                color_opacity_stds
            )
        else:
            metrics["color_opacity_std_avg"] = 0.0

        return metrics

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Generate and save validation samples with fixed seed.

        dataloader_idx 0: validation data, dataloader_idx 1: train data inference.
        """
        prefix = "val" if dataloader_idx == 0 else "train_inference"
        self._run_inference_step(batch, prefix)

    def _run_inference_step(self, batch, prefix):
        """Run inference sampling and log metrics/images with the given prefix."""
        VALIDATION_SEED = 42
        SAMPLE_SIZE = 256  # Sequence length
        IMG_SIZE = 512  # Output image size

        # batch is (curve_tensor, image_tensor)
        _, images_input = batch
        images_input = images_input.squeeze(1)  # [B, 1, 3, H, W] -> [B, 3, H, W]
        num_samples = images_input.shape[0]

        # Encode images to get conditioning
        with torch.inference_mode():
            cond = self.image_encoder(pixel_values=images_input).last_hidden_state

        # Sample from the model with fixed seed
        samples = self.sample(
            cond,
            steps=50,
            cfg_scale=1.0,
            shape=(num_samples, SAMPLE_SIZE, self.hparams.input_dim),
            seed=VALIDATION_SEED,
        )

        # Compute and log metrics
        metrics = self._compute_validation_metrics(samples)
        for metric_name, metric_value in metrics.items():
            self.log(
                f"{prefix}/{metric_name}",
                metric_value,
                add_dataloader_idx=False,
            )

        # Render and log each sample to wandb
        generated_images = []
        cond_images = []
        mse_values = []
        cond_logged_attr = f"_{prefix}_cond_images_logged"
        for i in range(num_samples):
            sample_tensor = samples[i]  # Shape: [SAMPLE_SIZE, input_dim]

            # Denormalize conditioning image for logging and MSE comparison
            cond_img = images_input[i].cpu()
            # ImageNet mean and std used by DINO processor
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            cond_img = cond_img * std + mean
            cond_img = (cond_img.clamp(0, 1) * 255).to(torch.uint8)
            cond_img_np = cond_img.permute(1, 2, 0).numpy()
            cond_pil = Image.fromarray(cond_img_np)

            # Log conditioning image only once (on first run for this prefix)
            if not getattr(self, cond_logged_attr, False):
                cond_images.append(
                    wandb.Image(cond_img_np, caption=f"Conditioning {i}")
                )

            try:
                # Convert tensor to shapes
                shapes = tensor_to_shapes(sample_tensor, IMG_SIZE, IMG_SIZE)
                # Convert shapes to SVG
                svg_content = save_bezier_shapes_to_svg(shapes, IMG_SIZE, IMG_SIZE)
                # Render SVG to image with white background (matching conditioning)
                image = render_svg_bg(svg_content)
                generated_images.append(wandb.Image(image, caption=f"Sample {i}"))

                # Resize generated image to conditioning image size for MSE
                gen_resized = image.resize(cond_pil.size, Image.LANCZOS)
                mse_val = calculate_mse(cond_pil, gen_resized)
                mse_values.append(mse_val)

            except Exception as e:
                print(
                    f"Warning: Failed to render sample {i} at epoch {self.current_epoch}: {e}"
                )

        # Log MSE between conditioning and generated images
        if mse_values:
            avg_mse = sum(mse_values) / len(mse_values)
            self.log(f"{prefix}/image_mse", avg_mse, add_dataloader_idx=False)

        # Log images to wandb
        log_dict = {"epoch": self.current_epoch}
        if generated_images:
            log_dict[f"{prefix}_samples"] = generated_images
        if cond_images:
            log_dict[f"{prefix}_conditioning"] = cond_images
            setattr(self, cond_logged_attr, True)
        if generated_images or cond_images:
            self.logger.experiment.log(log_dict, step=self.global_step)

    @torch.no_grad()
    def sample(self, cond, steps=50, cfg_scale=1.0, shape=None, seed=None):
        """
        RK4 ODE solver for sampling.

        cond: Conditioning sequence [Batch, CondLen, CondDim]
        steps: Number of integration steps
        cfg_scale: Classifier-free guidance scale.
                   1.0 = standard conditional, >1.0 = increased conditioning influence.
        shape: Output shape [Batch, SeqLen, Dim]. If None, inferred from cond.
        seed: Optional random seed for reproducible sampling.
        """
        self.eval()
        device = cond.device
        batch_size = cond.shape[0]

        # Determine output shape (assuming same seq len as cond for this example, or passed manually)
        if shape is None:
            # Default to output length = condition length (or set manually)
            shape = (batch_size, cond.shape[1], self.hparams.input_dim)

        # Set seed for reproducibility if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        # 1. Initialize x_0 from Normal distribution
        x = torch.randn(shape, device=device, generator=generator)

        # Time steps (0 to 1)
        ts = torch.linspace(0, 1, steps, device=device)

        # Prepare null conditioning mask for CFG
        null_mask = torch.ones(
            batch_size, device=device, dtype=torch.bool
        )  # All True = All Null

        def _velocity(x, t_tensor):
            """Compute velocity with optional CFG."""
            if cfg_scale == 1.0:
                # Standard conditional sampling
                return self(x, t_tensor, cond, mask_cond=None)
            else:
                # Classifier-Free Guidance
                # Conditional Pass (mask_cond=False)
                v_cond = self(x, t_tensor, cond, mask_cond=torch.zeros_like(null_mask))
                # Unconditional Pass (mask_cond=True)
                v_uncond = self(x, t_tensor, cond, mask_cond=null_mask)
                # Combine
                return v_uncond + cfg_scale * (v_cond - v_uncond)

        for i in range(len(ts) - 1):
            t_curr = ts[i]
            t_next = ts[i + 1]
            dt = t_next - t_curr
            t_curr_tensor = torch.full((batch_size,), t_curr, device=device)
            t_mid_tensor = torch.full((batch_size,), t_curr + 0.5 * dt, device=device)
            t_next_tensor = torch.full((batch_size,), t_next, device=device)

            # RK4 integration
            k1 = _velocity(x, t_curr_tensor)
            k2 = _velocity(x + 0.5 * dt * k1, t_mid_tensor)
            k3 = _velocity(x + 0.5 * dt * k2, t_mid_tensor)
            k4 = _velocity(x + dt * k3, t_next_tensor)

            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x
