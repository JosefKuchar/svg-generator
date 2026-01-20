import typer
import torch
from model import FlowMatchingTransformer
from dataset import DataModule
import pytorch_lightning as pl

app = typer.Typer()


# def generate_blob_bezier(
#     num_points=None,
#     min_points=4,
#     max_points=16,
#     radius=0.25,
#     variance=0.2,
#     smoothness=0.2,
#     center=(0.0, 0.0),
# ):
#     """
#     Generates a random blob shape defined by a sequence of Cubic Bezier curves.

#     Args:
#         num_points (int, optional): Number of anchor points (segments). If None, randomly selected between min_points and max_points.
#         min_points (int): Minimum number of anchor points when randomizing.
#         max_points (int): Maximum number of anchor points when randomizing.
#         radius (float): Base radius of the blob.
#         variance (float): How much the radius can vary (0.0 to 1.0).
#         smoothness (float): Strength of the control point handles (0.0 to 0.5 recommended).
#         center (tuple): Center point (x, y) of the blob. Defaults to (0.0, 0.0).

#     Returns:
#         list of tuples: Each tuple is ((x0, y0), (cp1x, cp1y), (cp2x, cp2y), (x1, y1))
#                         representing a Cubic Bezier segment.
#     """
#     cx, cy = center

#     # Randomize number of points if not specified
#     if num_points is None:
#         num_points = random.randint(min_points, max_points)

#     # 1. Generate random anchor points in polar coordinates
#     angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
#     points = []

#     for angle in angles:
#         # Vary the radius randomly
#         r = radius * (1 + random.uniform(-variance, variance))
#         x = cx + r * np.cos(angle)
#         y = cy + r * np.sin(angle)
#         points.append(np.array([x, y]))

#     points = np.array(points)
#     n = len(points)

#     # 2. Calculate Control Points
#     # We use a heuristic where the tangent at point P_i is parallel
#     # to the line connecting P_{i-1} and P_{i+1}.

#     # Pre-calculate tangents for every point
#     tangents = []
#     for i in range(n):
#         p_prev = points[(i - 1) % n]
#         p_next = points[(i + 1) % n]

#         # Vector from prev to next
#         vec = p_next - p_prev

#         # Tangent vector (normalized not strictly necessary if using dist scaling,
#         # but useful for consistent logic)
#         # Here we just scale the vector directly by the smoothness factor
#         tangents.append(vec * smoothness)

#     # 3. Construct Segments
#     # A Cubic Bezier segment connects P_i to P_{i+1}
#     # It uses the "outgoing" control point of P_i and "incoming" of P_{i+1}

#     bezier_segments = []
#     for i in range(n):
#         p0 = points[i]
#         p3 = points[(i + 1) % n]

#         # Control Point 1: p0 + tangent[i]
#         cp1 = p0 + tangents[i]

#         # Control Point 2: p3 - tangent[i+1] (minus because it's incoming)
#         cp2 = p3 - tangents[(i + 1) % n]

#         bezier_segments.append((tuple(p0), tuple(cp1), tuple(cp2), tuple(p3)))

#     return bezier_segments


@app.command()
def app():
    torch.set_float32_matmul_precision("medium")

    module = FlowMatchingTransformer(
        input_dim=11, cond_dim=2, hidden_size=512, num_layers=6, num_heads=8
    )

    trainer = pl.Trainer(max_epochs=500, accelerator="auto", gradient_clip_val=1.0)
    trainer.fit(
        module,
        datamodule=DataModule(),
    )


if __name__ == "__main__":
    app()
