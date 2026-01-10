import typer
import torch
import torch.nn as nn
import math
import lightning as pl
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib import patches
from torch.utils.data import IterableDataset, DataLoader
import imageio
import os

app = typer.Typer()

# -----------------------------------------------------------------------------
# 1. Helper Components: Embeddings & Attention
# -----------------------------------------------------------------------------


class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal position embedding for vector sequences.
    """

    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, SeqLen, Dim]
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]


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


# -----------------------------------------------------------------------------
# 2. Transformer Block with AdaLN & Cross-Attention
# -----------------------------------------------------------------------------


class AdaLNBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()

        # Self-Attention (for the main flow sequence)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=dropout
        )

        # Cross-Attention (Conditioning on external vector sequence)
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

    def forward(self, x, c, t_emb):
        """
        x: Input sequence (Flow) [Batch, SeqLen, Dim]
        c: Conditioning sequence [Batch, CondSeqLen, Dim]
        t_emb: Timestep embedding [Batch, Dim]
        """
        # 1. Regress modulation parameters from time embedding
        # shift_msa, scale_msa, shift_cross, scale_cross, shift_mlp, scale_mlp
        params = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        (shift_msa, scale_msa, shift_cross, scale_cross, shift_mlp, scale_mlp) = params

        # Helper for modulation
        def modulate(x, shift, scale):
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # 2. Self-Attention Block
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]

        # 3. Cross-Attention Block
        x_norm = modulate(self.norm2(x), shift_cross, scale_cross)
        # Query = Flow (x), Key/Value = Conditioning (c)
        x = x + self.cross_attn(x_norm, c, c)[0]

        # 4. Feed-Forward Block
        x_norm = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + self.mlp(x_norm)

        return x


# -----------------------------------------------------------------------------
# 3. Main Lightning Module
# -----------------------------------------------------------------------------


class FlowMatchingTransformer(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: int = 0.1,
        cond_drop_prob: float = 0.1,  # Probability to drop conditioning
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cond_drop_prob = cond_drop_prob

        # 1. Embeddings
        self.x_embedder = nn.Linear(input_dim, hidden_size)
        self.c_embedder = nn.Linear(cond_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.pos_emb = SinusoidalPosEmb(hidden_size)

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

        # Add Positional Embeddings
        x = x + self.pos_emb(x)
        c = c + self.pos_emb(c)

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
            x = block(x, c, t_emb)

        # Final AdaLN & Projection
        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        output = self.final_proj(x)

        return output

    def training_step(self, batch, batch_idx):
        # x_1: Real Data
        # cond: Conditioning
        x_1, cond = batch
        batch_size = x_1.size(0)
        device = x_1.device

        # 1. Sample Time t ~ Uniform[0, 1]
        # RF often suggests sampling Logit-Normal for t,
        # but Uniform is standard for the base version.
        t = torch.rand(batch_size, device=device)

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
        loss = F.mse_loss(pred_v, target_v)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

    @torch.no_grad()
    def sample(
        self, cond, steps=50, cfg_scale=1.0, shape=None, return_intermediates=False
    ):
        """
        Euler ODE solver for sampling.

        cond: Conditioning sequence [Batch, CondLen, CondDim]
        steps: Number of integration steps
        cfg_scale: Classifier-free guidance scale.
                   1.0 = standard conditional, >1.0 = increased conditioning influence.
        shape: Output shape [Batch, SeqLen, Dim]. If None, inferred from cond.
        return_intermediates: If True, returns list of intermediate states during sampling.
        """
        self.eval()
        device = cond.device
        batch_size = cond.shape[0]

        # Determine output shape (assuming same seq len as cond for this example, or passed manually)
        if shape is None:
            # Default to output length = condition length (or set manually)
            shape = (batch_size, cond.shape[1], self.hparams.input_dim)

        # 1. Initialize x_0 from Normal distribution
        x = torch.randn(shape, device=device)

        # Time steps (0 to 1)
        ts = torch.linspace(0, 1, steps, device=device)
        dt = 1.0 / steps

        # Prepare null conditioning for CFG
        null_mask = torch.ones(
            batch_size, device=device, dtype=torch.bool
        )  # All True = All Null

        intermediates = []
        if return_intermediates:
            intermediates.append(x.cpu().clone())

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

            if return_intermediates:
                intermediates.append(x.cpu().clone())

        if return_intermediates:
            return x, intermediates
        return x


def generate_blob_bezier(
    num_points=None,
    min_points=4,
    max_points=16,
    radius=0.5,
    variance=0.2,
    smoothness=0.2,
):
    """
    Generates a random blob shape defined by a sequence of Cubic Bezier curves.

    Args:
        num_points (int, optional): Number of anchor points (segments). If None, randomly selected between min_points and max_points.
        min_points (int): Minimum number of anchor points when randomizing.
        max_points (int): Maximum number of anchor points when randomizing.
        radius (float): Base radius of the blob.
        variance (float): How much the radius can vary (0.0 to 1.0).
        smoothness (float): Strength of the control point handles (0.0 to 0.5 recommended).

    Returns:
        list of tuples: Each tuple is ((x0, y0), (cp1x, cp1y), (cp2x, cp2y), (x1, y1))
                        representing a Cubic Bezier segment.
    """

    # Randomize number of points if not specified
    if num_points is None:
        num_points = random.randint(min_points, max_points)

    # 1. Generate random anchor points in polar coordinates
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = []

    for angle in angles:
        # Vary the radius randomly
        r = radius * (1 + random.uniform(-variance, variance))
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        points.append(np.array([x, y]))

    points = np.array(points)
    n = len(points)

    # 2. Calculate Control Points
    # We use a heuristic where the tangent at point P_i is parallel
    # to the line connecting P_{i-1} and P_{i+1}.

    # Pre-calculate tangents for every point
    tangents = []
    for i in range(n):
        p_prev = points[(i - 1) % n]
        p_next = points[(i + 1) % n]

        # Vector from prev to next
        vec = p_next - p_prev

        # Tangent vector (normalized not strictly necessary if using dist scaling,
        # but useful for consistent logic)
        # Here we just scale the vector directly by the smoothness factor
        tangents.append(vec * smoothness)

    # 3. Construct Segments
    # A Cubic Bezier segment connects P_i to P_{i+1}
    # It uses the "outgoing" control point of P_i and "incoming" of P_{i+1}

    bezier_segments = []
    for i in range(n):
        p0 = points[i]
        p3 = points[(i + 1) % n]

        # Control Point 1: p0 + tangent[i]
        cp1 = p0 + tangents[i]

        # Control Point 2: p3 - tangent[i+1] (minus because it's incoming)
        cp2 = p3 - tangents[(i + 1) % n]

        bezier_segments.append((tuple(p0), tuple(cp1), tuple(cp2), tuple(p3)))

    return bezier_segments


# --- Visualization Helper ---


def plot_blob(segments, filename="blob.png", title=None, valid_segments=None):
    """
    Plot blob with optional validity information.

    Args:
        segments: List of Bezier curve segments
        filename: Output filename
        title: Plot title
        valid_segments: List of booleans indicating which segments are valid.
                       If None, all segments are considered valid.
    """
    if len(segments) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    # If no validity info provided, treat all as valid
    if valid_segments is None:
        valid_segments = [True] * len(segments)

    # Separate valid and invalid segments
    valid_paths = []
    invalid_paths = []

    for i, segment in enumerate(segments):
        if i < len(valid_segments) and valid_segments[i]:
            valid_paths.append(segment)
        else:
            invalid_paths.append(segment)

    # Plot valid segments in orange
    if valid_paths:
        codes = [Path.MOVETO]
        verts = [valid_paths[0][0]]  # Start at the first point

        for segment in valid_paths:
            verts.extend([segment[1], segment[2], segment[3]])
            codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])

        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, facecolor="orange", lw=2, alpha=0.6, edgecolor="orange"
        )
        ax.add_patch(patch)

    # Plot invalid segments in red
    if invalid_paths:
        for segment in invalid_paths:
            codes = [Path.MOVETO]
            verts = [segment[0]]
            verts.extend([segment[1], segment[2], segment[3]])
            codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])

            path = Path(verts, codes)
            patch = patches.PathPatch(
                path, facecolor="red", lw=2, alpha=0.6, edgecolor="red"
            )
            ax.add_patch(patch)

    # Plot formatting
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close(fig)


def create_sampling_gif(
    module,
    cond,
    output_path="sampling_process.gif",
    steps=50,
    cfg_scale=1.0,
    shape=None,
    fps=10,
):
    """
    Create a GIF showing the sampling process step by step.

    Args:
        module: FlowMatchingTransformer model
        cond: Conditioning tensor
        output_path: Path to save the GIF
        steps: Number of sampling steps
        cfg_scale: Classifier-free guidance scale
        shape: Output shape
        fps: Frames per second for the GIF
    """
    # Sample with intermediate states
    final_x, intermediates = module.sample(
        cond, steps=steps, cfg_scale=cfg_scale, shape=shape, return_intermediates=True
    )

    # Create temporary directory for frames
    temp_dir = "temp_gif_frames"
    os.makedirs(temp_dir, exist_ok=True)

    frame_paths = []
    try:
        # Generate frames
        for i, x_intermediate in enumerate(intermediates):
            curve, valid_segments = tensor_to_curve(
                x_intermediate, return_validity=True
            )
            if len(curve) > 0:
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                t_value = (
                    i / (len(intermediates) - 1) if len(intermediates) > 1 else 0.0
                )
                plot_blob(
                    curve,
                    filename=frame_path,
                    title=f"t = {t_value:.3f}",
                    valid_segments=valid_segments,
                )
                frame_paths.append(frame_path)

        # Create GIF from frames
        if frame_paths:
            images = [imageio.imread(path) for path in frame_paths]
            imageio.mimsave(output_path, images, fps=fps, loop=0)
            print(f"GIF saved to {output_path}")
        else:
            print("No frames generated!")

    finally:
        # Clean up temporary frames
        for path in frame_paths:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


def curve_to_tensor(curve, max_segments=16):
    t = torch.zeros(max_segments, 9)
    for i in range(max_segments):
        if i < len(curve):
            segment = curve[i]
            flattened = (
                tuple(segment[0])
                + tuple(segment[1])
                + tuple(segment[2])
                + tuple(segment[3])
            )
            flattened = flattened + (1.0,)
            t[i] = torch.tensor(flattened)
        else:
            t[i] = t[i - 1].clone()
            t[i][8] = -1.0
    return t


def tensor_to_curve(t, return_validity=False):
    # Handle batch dimension: if t has shape (batch, segments, 9), take first batch
    if len(t.shape) == 3:
        t = t[0]  # Take first batch element, now shape is (segments, 9)

    curve = []
    valid_segments = []
    for i in range(t.shape[0]):
        segment = t[i]  # shape: (9,)
        flag = segment[8].item()

        x0, y0 = segment[0].item(), segment[1].item()
        cp1x, cp1y = segment[2].item(), segment[3].item()
        cp2x, cp2y = segment[4].item(), segment[5].item()
        x1, y1 = segment[6].item(), segment[7].item()

        curve.append(((x0, y0), (cp1x, cp1y), (cp2x, cp2y), (x1, y1)))
        valid_segments.append(flag >= 0)

        if flag < 0:
            # Still include invalid segments for visualization
            pass

    if return_validity:
        return curve, valid_segments
    return curve


class SyntheticDataset(IterableDataset):
    def __init__(
        self,
        num_points=None,
        min_points=4,
        max_points=16,
        radius=0.5,
        variance=0.2,
        smoothness=0.2,
        cond_dim=6,
    ):
        self.num_points = num_points
        self.min_points = min_points
        self.max_points = max_points
        self.radius = radius
        self.variance = variance
        self.smoothness = smoothness
        self.cond_dim = cond_dim

    def __iter__(self):
        while True:
            curve = generate_blob_bezier(
                num_points=self.num_points,
                min_points=self.min_points,
                max_points=self.max_points,
                radius=self.radius,
                variance=self.variance,
                smoothness=self.smoothness,
            )
            curve_tensor = curve_to_tensor(curve)
            # Empty conditioning vector: shape (1, cond_dim) -> becomes (batch_size, 1, cond_dim) when batched
            cond_tensor = torch.zeros(1, self.cond_dim)
            yield curve_tensor, cond_tensor


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=2048, num_workers=10, cond_dim=6):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cond_dim = cond_dim

    def train_dataloader(self):
        dataset = SyntheticDataset(cond_dim=self.cond_dim)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


@app.command()
def app():
    torch.set_float32_matmul_precision("medium")

    module = FlowMatchingTransformer(
        input_dim=9, cond_dim=1, hidden_size=128, num_layers=4, num_heads=4
    )

    # trainer = pl.Trainer(max_epochs=5, limit_train_batches=10000, accelerator="auto")
    # trainer.fit(
    #     module,
    #     datamodule=DataModule(cond_dim=1),
    # )

    # Load lightning checkpoint
    module = FlowMatchingTransformer.load_from_checkpoint(
        "./lightning_logs/version_5/checkpoints/epoch=4-step=50000.ckpt"
    )

    # Test
    cond = torch.zeros(1, 1).to(module.device)
    x = module.sample(cond, shape=(1, 16, 9)).to("cpu")
    print(x)
    curve = tensor_to_curve(x)
    print(len(curve))
    plot_blob(curve)

    # Create GIF of sampling process
    print("Creating sampling GIF...")
    create_sampling_gif(
        module, cond, output_path="sampling_process.gif", steps=50, shape=(1, 16, 9)
    )


if __name__ == "__main__":
    app()
