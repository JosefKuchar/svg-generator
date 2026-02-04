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
from raster import render_svg

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

        # Cross-Attention (no RoPE, using standard attention)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=dropout
        )

        # Feed Forward
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

        # AdaLN Modulation: Predicts shift (beta) and scale (gamma) for all 3 norms
        # Output dim = 6 * hidden_size because we need (gamma, beta) for 3 norms
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

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

    def forward(self, x, c, t_emb, rope_cos, rope_sin):
        """
        x: Input sequence (Flow) [Batch, SeqLen, Dim]
        c: Conditioning sequence [Batch, CondSeqLen, Dim]
        t_emb: Timestep embedding [Batch, Dim]
        rope_cos: RoPE cosine [1, SeqLen, HeadDim]
        rope_sin: RoPE sine [1, SeqLen, HeadDim]
        """
        # 1. Regress modulation parameters from time embedding
        # shift_msa, scale_msa, shift_cross, scale_cross, shift_mlp, scale_mlp
        params = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        (shift_msa, scale_msa, shift_cross, scale_cross, shift_mlp, scale_mlp) = params

        # Helper for modulation
        def modulate(x, shift, scale):
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # 2. Self-Attention Block with RoPE
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + self._rope_attention(x_norm, rope_cos, rope_sin)

        # 3. Cross-Attention Block (no RoPE for cross-attention)
        x_norm = modulate(self.norm2(x), shift_cross, scale_cross)
        # Query = Flow (x), Key/Value = Conditioning (c)
        x = x + self.cross_attn(x_norm, c, c)[0]

        # 4. Feed-Forward Block
        x_norm = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + self.mlp(x_norm)

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

    def forward(self, x, t, c, mask_cond=None):
        """
        x: Noisy input [Batch, SeqLen, Dim]
        t: Timesteps [Batch]
        c: Conditioning Vector Sequence [Batch, CondLen, CondDim]
        mask_cond: Boolean tensor [Batch]. If True, drop conditioning for that sample.
        """
        # Embedding inputs
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

        # Final AdaLN & Projection
        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        output = self.final_proj(x)

        return output

    def training_step(self, batch, batch_idx):
        # x_1: Real Data
        # cond: Conditioning
        x_1, images = batch
        images = images.squeeze(1)  # [B, 1, 3, H, W] -> [B, 3, H, W]
        batch_size = x_1.size(0)
        device = x_1.device

        with torch.inference_mode():
            cond = self.image_encoder(pixel_values=images).last_hidden_state
        print(cond.shape)

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

        # 7. Padding-aware loss computation
        # x_1[..., -1] == -1 indicates padding tokens
        # For padding: only compute loss on the 'real' flag (last dim)
        # For real data: compute loss on all dimensions
        is_padding = x_1[..., -1] < 0  # [Batch, SeqLen]

        # Compute per-element squared error
        sq_error = (pred_v - target_v) ** 2  # [Batch, SeqLen, Dim]

        # Create mask: 1 for dimensions that should count, 0 for masked out
        # For real tokens: all dims count (mask = 1 everywhere)
        # For padding tokens: only last dim counts (mask = 0 for dims 0-13, 1 for dim 14)
        loss_mask = torch.ones_like(sq_error)
        # Zero out non-flag dimensions for padding tokens
        loss_mask[is_padding, :-1] = 0.0

        # Compute masked mean loss
        loss = (sq_error * loss_mask).sum() / loss_mask.sum()

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, eps=1e-5
        )

    def _compute_validation_metrics(self, samples: torch.Tensor) -> dict:
        """
        Compute validation metrics for generated samples.

        Args:
            samples: Tensor of shape [batch, seq_len, input_dim]
                     where input_dim = 15 (x0,y0,x1,y1,x2,y2,x3,y3,r,g,b,opacity,path_start,subpath_start,real)

        Returns:
            Dictionary with computed metrics
        """
        metrics = {}

        # 1. Flag closeness to 1 or -1
        # Flags are at indices 12 (path_start), 13 (subpath_start), 14 (real)
        flag_indices = [12, 13, 14]
        flag_names = ["path_start", "subpath_start", "real"]

        for idx, name in zip(flag_indices, flag_names):
            flag_values = samples[..., idx]  # [batch, seq_len]
            # Closeness = 1 - min(|value - 1|, |value + 1|)
            # Perfect flags have closeness = 1, random noise has closeness ~ 0
            dist_to_pos1 = torch.abs(flag_values - 1.0)
            dist_to_neg1 = torch.abs(flag_values + 1.0)
            min_dist = torch.minimum(dist_to_pos1, dist_to_neg1)
            # Clamp closeness to [0, 1] (values outside [-1, 1] would give negative closeness)
            closeness = torch.clamp(1.0 - min_dist, 0.0, 1.0)
            metrics[f"flag_{name}_closeness"] = closeness.mean().item()

        # Average flag closeness
        all_flag_closeness = []
        for idx in flag_indices:
            flag_values = samples[..., idx]
            dist_to_pos1 = torch.abs(flag_values - 1.0)
            dist_to_neg1 = torch.abs(flag_values + 1.0)
            min_dist = torch.minimum(dist_to_pos1, dist_to_neg1)
            closeness = torch.clamp(1.0 - min_dist, 0.0, 1.0)
            all_flag_closeness.append(closeness)
        metrics["flag_closeness_avg"] = torch.stack(all_flag_closeness).mean().item()

        # 2. Color and opacity consistency per shape
        # Colors are at indices 8 (r), 9 (g), 10 (b), 11 (opacity)
        color_opacity_stds = []

        for batch_idx in range(samples.shape[0]):
            sample = samples[batch_idx]  # [seq_len, input_dim]

            # Find shape boundaries using path_start flag (threshold at 0)
            path_starts = sample[:, 12] > 0  # [seq_len]
            real_flags = sample[:, 14] > 0  # [seq_len]

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
                    real_mask = shape_segment[:, 14] > 0
                    real_segments = shape_segment[real_mask]

                    if len(real_segments) > 1:
                        # Compute std for each color channel and opacity
                        r_std = real_segments[:, 8].std().item()
                        g_std = real_segments[:, 9].std().item()
                        b_std = real_segments[:, 10].std().item()
                        opacity_std = real_segments[:, 11].std().item()
                        color_opacity_stds.append(
                            (r_std + g_std + b_std + opacity_std) / 4
                        )

        if color_opacity_stds:
            metrics["color_opacity_std_avg"] = sum(color_opacity_stds) / len(
                color_opacity_stds
            )
        else:
            metrics["color_opacity_std_avg"] = 0.0

        # 3. Coordinate closure - distance between end of curve and start of next curve (cyclic per subpath)
        coord_closure_dists = []

        for batch_idx in range(samples.shape[0]):
            sample = samples[batch_idx]  # [seq_len, input_dim]
            real_flags = sample[:, 14] > 0
            subpath_starts = sample[:, 13] > 0

            # Get indices of real segments
            real_indices = torch.where(real_flags)[0]

            if len(real_indices) > 1:
                # Find subpath boundaries
                subpath_start_positions = torch.where(subpath_starts & real_flags)[0]

                if len(subpath_start_positions) == 0:
                    # Treat all as one subpath
                    subpath_ranges = [(0, len(real_indices))]
                else:
                    # Map subpath starts to positions in real_indices
                    subpath_ranges = []
                    real_indices_list = real_indices.tolist()

                    for i, start_pos in enumerate(subpath_start_positions):
                        start_in_real = (
                            real_indices_list.index(start_pos.item())
                            if start_pos.item() in real_indices_list
                            else -1
                        )
                        if start_in_real >= 0:
                            if i + 1 < len(subpath_start_positions):
                                next_start = subpath_start_positions[i + 1].item()
                                end_in_real = (
                                    real_indices_list.index(next_start)
                                    if next_start in real_indices_list
                                    else len(real_indices)
                                )
                            else:
                                end_in_real = len(real_indices)
                            subpath_ranges.append((start_in_real, end_in_real))

                # For each subpath, compute cyclic closure distances
                for start_in_real, end_in_real in subpath_ranges:
                    subpath_real_indices = real_indices[start_in_real:end_in_real]
                    if len(subpath_real_indices) > 1:
                        for i in range(len(subpath_real_indices)):
                            curr_idx = subpath_real_indices[i]
                            next_idx = subpath_real_indices[
                                (i + 1) % len(subpath_real_indices)
                            ]

                            # End point of current curve (x3, y3) = indices 6, 7
                            curr_end_x = sample[curr_idx, 6]
                            curr_end_y = sample[curr_idx, 7]

                            # Start point of next curve (x0, y0) = indices 0, 1
                            next_start_x = sample[next_idx, 0]
                            next_start_y = sample[next_idx, 1]

                            # Euclidean distance
                            dist = torch.sqrt(
                                (curr_end_x - next_start_x) ** 2
                                + (curr_end_y - next_start_y) ** 2
                            )
                            coord_closure_dists.append(dist.item())

        if coord_closure_dists:
            metrics["coord_closure_dist_avg"] = sum(coord_closure_dists) / len(
                coord_closure_dists
            )
        else:
            metrics["coord_closure_dist_avg"] = 0.0

        return metrics

    def validation_step(self, batch, batch_idx):
        """Generate and save validation samples with fixed seed."""
        VALIDATION_SEED = 42
        SAMPLE_SIZE = 256  # Sequence length
        IMG_SIZE = 512  # Output image size

        # batch is the conditioning tensor from ValidationSamplingDataset
        cond = batch  # Shape: [num_samples, 1, cond_dim]
        num_samples = cond.shape[0]

        # Sample from the model with fixed seed
        samples = self.sample(
            cond,
            steps=50,
            cfg_scale=1.0,
            shape=(num_samples, SAMPLE_SIZE, self.hparams.input_dim),
            seed=VALIDATION_SEED,
        )

        # Compute and log validation metrics
        val_metrics = self._compute_validation_metrics(samples)
        for metric_name, metric_value in val_metrics.items():
            self.log(f"val/{metric_name}", metric_value)

        # Render and log each sample to wandb
        images = []
        for i in range(num_samples):
            sample_tensor = samples[i]  # Shape: [SAMPLE_SIZE, input_dim]
            try:
                # Convert tensor to shapes
                shapes = tensor_to_shapes(sample_tensor, IMG_SIZE, IMG_SIZE)
                # Convert shapes to SVG
                svg_content = save_bezier_shapes_to_svg(shapes, IMG_SIZE, IMG_SIZE)
                # Render SVG to image
                image = render_svg(svg_content)
                images.append(wandb.Image(image, caption=f"Sample {i}"))

            except Exception as e:
                print(
                    f"Warning: Failed to render sample {i} at epoch {self.current_epoch}: {e}"
                )

        # Log images to wandb
        if images:
            self.logger.experiment.log(
                {"val_samples": images, "epoch": self.current_epoch}
            )

    @torch.no_grad()
    def sample(self, cond, steps=50, cfg_scale=1.0, shape=None, seed=None):
        """
        Euler ODE solver for sampling.

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
        dt = 1.0 / steps

        # Prepare null conditioning for CFG
        null_mask = torch.ones(
            batch_size, device=device, dtype=torch.bool
        )  # All True = All Null

        for i in range(len(ts) - 1):
            t_curr = ts[i]
            t_tensor = torch.full((batch_size,), t_curr, device=device)

            # Predict velocity
            if cfg_scale == 1.0:
                # Standard conditional sampling
                v_pred = self(x, t_tensor, cond, mask_cond=None)
            else:
                # Classifier-Free Guidance
                # We need two forward passes: Conditional and Unconditional

                # Conditional Pass (mask_cond=False)
                v_cond = self(x, t_tensor, cond, mask_cond=torch.zeros_like(null_mask))

                # Unconditional Pass (mask_cond=True)
                v_uncond = self(x, t_tensor, cond, mask_cond=null_mask)

                # Combine
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)

            # Euler Step
            x = x + v_pred * dt
        return x
