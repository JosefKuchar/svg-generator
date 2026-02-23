"""
Synthetic dataset generator for random geometric shapes.

Generates random scenes of circles, ovals, rectangles, and squares
as BezierShape objects, compatible with the existing representation pipeline.
"""

import math
import random

import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

from parsing import save_bezier_shapes_to_svg
from raster import render_svg_bg
from representation import BezierPath, BezierShape, Curve, Point, shapes_to_tensor

# Kappa for approximating a circle with 4 cubic bezier curves
# This gives < 0.027% error
KAPPA = 0.5522847498


def _rotate_point(x: float, y: float, angle: float, cx: float, cy: float) -> Point:
    """Rotate point (x, y) by angle (radians) around center (cx, cy)."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = x - cx
    dy = y - cy
    return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)


def _rotate_curves(
    curves: list[Curve], angle: float, cx: float, cy: float
) -> list[Curve]:
    """Rotate all control points in a list of curves around (cx, cy)."""
    rotated = []
    for p0, p1, p2, p3 in curves:
        rotated.append(
            (
                _rotate_point(*p0, angle, cx, cy),
                _rotate_point(*p1, angle, cx, cy),
                _rotate_point(*p2, angle, cx, cy),
                _rotate_point(*p3, angle, cx, cy),
            )
        )
    return rotated


def _line_to_cubic(p0: Point, p3: Point) -> Curve:
    """Convert a line segment to a cubic bezier (control points at 1/3 and 2/3)."""
    p1 = (p0[0] + (p3[0] - p0[0]) / 3, p0[1] + (p3[1] - p0[1]) / 3)
    p2 = (p0[0] + 2 * (p3[0] - p0[0]) / 3, p0[1] + 2 * (p3[1] - p0[1]) / 3)
    return (p0, p1, p2, p3)


def make_ellipse(
    cx: float, cy: float, rx: float, ry: float, angle: float = 0.0
) -> list[Curve]:
    """
    Create an ellipse as 4 cubic bezier curves.

    Args:
        cx, cy: Center position.
        rx, ry: Radii along x and y axes (before rotation).
        angle: Rotation angle in radians.

    Returns:
        List of 4 Curve tuples forming a closed ellipse.
    """
    kx = KAPPA * rx
    ky = KAPPA * ry

    # 4 quadrants, starting from rightmost point, going clockwise
    curves = [
        # Right to bottom
        ((cx + rx, cy), (cx + rx, cy + ky), (cx + kx, cy + ry), (cx, cy + ry)),
        # Bottom to left
        ((cx, cy + ry), (cx - kx, cy + ry), (cx - rx, cy + ky), (cx - rx, cy)),
        # Left to top
        ((cx - rx, cy), (cx - rx, cy - ky), (cx - kx, cy - ry), (cx, cy - ry)),
        # Top to right
        ((cx, cy - ry), (cx + kx, cy - ry), (cx + rx, cy - ky), (cx + rx, cy)),
    ]

    if angle != 0.0:
        curves = _rotate_curves(curves, angle, cx, cy)

    return curves


def make_rectangle(
    cx: float, cy: float, w: float, h: float, angle: float = 0.0
) -> list[Curve]:
    """
    Create a rectangle as 4 cubic bezier line segments.

    Args:
        cx, cy: Center position.
        w, h: Width and height.
        angle: Rotation angle in radians.

    Returns:
        List of 4 Curve tuples forming a closed rectangle.
    """
    hw, hh = w / 2, h / 2
    corners = [
        (cx - hw, cy - hh),  # top-left
        (cx + hw, cy - hh),  # top-right
        (cx + hw, cy + hh),  # bottom-right
        (cx - hw, cy + hh),  # bottom-left
    ]

    curves = []
    for i in range(4):
        curves.append(_line_to_cubic(corners[i], corners[(i + 1) % 4]))

    if angle != 0.0:
        curves = _rotate_curves(curves, angle, cx, cy)

    return curves


def make_triangle(
    cx: float, cy: float, size: float, angle: float = 0.0
) -> list[Curve]:
    """
    Create an equilateral triangle as 3 cubic bezier line segments.

    Args:
        cx, cy: Center position.
        size: Distance from center to vertices.
        angle: Rotation angle in radians.

    Returns:
        List of 3 Curve tuples forming a closed triangle.
    """
    vertices = []
    for i in range(3):
        a = angle + i * 2 * math.pi / 3 - math.pi / 2  # start from top
        vertices.append((cx + size * math.cos(a), cy + size * math.sin(a)))

    curves = []
    for i in range(3):
        curves.append(_line_to_cubic(vertices[i], vertices[(i + 1) % 3]))

    return curves


def make_regular_polygon(
    cx: float, cy: float, size: float, n_sides: int, angle: float = 0.0
) -> list[Curve]:
    """
    Create a regular polygon as cubic bezier line segments.

    Args:
        cx, cy: Center position.
        size: Distance from center to vertices.
        n_sides: Number of sides (3=triangle, 5=pentagon, 6=hexagon, etc.).
        angle: Rotation angle in radians.

    Returns:
        List of n_sides Curve tuples forming a closed polygon.
    """
    vertices = []
    for i in range(n_sides):
        a = angle + i * 2 * math.pi / n_sides - math.pi / 2
        vertices.append((cx + size * math.cos(a), cy + size * math.sin(a)))

    curves = []
    for i in range(n_sides):
        curves.append(_line_to_cubic(vertices[i], vertices[(i + 1) % n_sides]))

    return curves


def make_star(
    cx: float,
    cy: float,
    outer_r: float,
    inner_r: float,
    n_points: int,
    angle: float = 0.0,
) -> list[Curve]:
    """
    Create a star shape as cubic bezier line segments.

    Args:
        cx, cy: Center position.
        outer_r: Outer radius (tips of the star).
        inner_r: Inner radius (indentations).
        n_points: Number of points on the star.
        angle: Rotation angle in radians.

    Returns:
        List of 2*n_points Curve tuples forming a closed star.
    """
    vertices = []
    for i in range(2 * n_points):
        a = angle + i * math.pi / n_points - math.pi / 2
        r = outer_r if i % 2 == 0 else inner_r
        vertices.append((cx + r * math.cos(a), cy + r * math.sin(a)))

    curves = []
    n = len(vertices)
    for i in range(n):
        curves.append(_line_to_cubic(vertices[i], vertices[(i + 1) % n]))

    return curves


def generate_random_shape(
    canvas_w: float, canvas_h: float, rng: random.Random | None = None
) -> BezierShape:
    """
    Generate a single random shape within the given canvas.

    Args:
        canvas_w, canvas_h: Canvas dimensions.
        rng: Optional Random instance for reproducibility.

    Returns:
        A BezierShape with random type, position, size, rotation, and color.
    """
    r = rng or random

    # Random rotation
    angle = r.uniform(0, 2 * math.pi)

    # Random color and opacity (80% chance fully opaque, 20% partial)
    color = (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255))
    opacity = 1.0 if r.random() < 0.8 else r.uniform(0.2, 0.9)

    # Pick a random shape type
    shape_type = r.choice(["circle", "oval", "square", "rectangle", "triangle",
                           "pentagon", "hexagon", "star"])

    # Determine the shape's maximum extent (half-diagonal) so we can constrain
    # the center position to keep it approximately within the canvas.
    canvas_min = min(canvas_w, canvas_h)
    min_size = canvas_min * 0.05
    max_size = canvas_min * 0.40

    if shape_type == "circle":
        radius = r.uniform(min_size, max_size)
        extent = radius
    elif shape_type == "oval":
        rx = r.uniform(min_size, max_size)
        ry = r.uniform(min_size, max_size)
        extent = max(rx, ry)
    elif shape_type == "square":
        side = r.uniform(min_size * 2, max_size * 1.5)
        extent = side * math.sqrt(2) / 2  # half-diagonal under rotation
    elif shape_type == "rectangle":
        w = r.uniform(min_size * 2, max_size * 1.5)
        h = r.uniform(min_size * 2, max_size * 1.5)
        extent = math.sqrt(w * w + h * h) / 2  # half-diagonal under rotation
    elif shape_type == "triangle":
        size = r.uniform(min_size, max_size)
        extent = size
    elif shape_type == "pentagon":
        size = r.uniform(min_size, max_size)
        extent = size
    elif shape_type == "hexagon":
        size = r.uniform(min_size, max_size)
        extent = size
    elif shape_type == "star":
        outer_r = r.uniform(min_size, max_size)
        inner_r = r.uniform(outer_r * 0.3, outer_r * 0.6)
        n_points = r.choice([4, 5, 6, 7, 8])
        extent = outer_r
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

    # Constrain center so the shape stays approximately within the canvas.
    # Allow a small fraction of the extent to go off-edge for natural variety.
    margin = max(extent * 0.7, canvas_min * 0.05)
    cx = r.uniform(margin, canvas_w - margin)
    cy = r.uniform(margin, canvas_h - margin)

    # Build curves
    if shape_type == "circle":
        curves = make_ellipse(cx, cy, radius, radius, angle=0.0)

    elif shape_type == "oval":
        curves = make_ellipse(cx, cy, rx, ry, angle=angle)

    elif shape_type == "square":
        curves = make_rectangle(cx, cy, side, side, angle=angle)

    elif shape_type == "rectangle":
        curves = make_rectangle(cx, cy, w, h, angle=angle)

    elif shape_type == "triangle":
        curves = make_triangle(cx, cy, size, angle=angle)

    elif shape_type == "pentagon":
        curves = make_regular_polygon(cx, cy, size, 5, angle=angle)

    elif shape_type == "hexagon":
        curves = make_regular_polygon(cx, cy, size, 6, angle=angle)

    elif shape_type == "star":
        curves = make_star(cx, cy, outer_r, inner_r, n_points, angle=angle)

    path = BezierPath(curves)
    return BezierShape(paths=[path], color=color, opacity=opacity)


def generate_random_scene(
    canvas_w: float = 256.0,
    canvas_h: float = 256.0,
    min_shapes: int = 1,
    max_shapes: int = 10,
    max_segments: int = 256,
    seed: int | None = None,
) -> list[BezierShape]:
    """
    Generate a random scene with multiple shapes.

    Args:
        canvas_w, canvas_h: Canvas dimensions.
        min_shapes, max_shapes: Range for number of shapes per scene.
        max_segments: Maximum total bezier segments allowed.
        seed: Optional seed for reproducibility.

    Returns:
        List of BezierShape objects forming the scene.
    """
    rng = random.Random(seed)
    n_shapes = rng.randint(min_shapes, max_shapes)

    shapes = []
    total_segments = 0

    for _ in range(n_shapes):
        shape = generate_random_shape(canvas_w, canvas_h, rng=rng)
        # Count segments in this shape
        n_segs = sum(len(p.curves) for p in shape.paths)
        if total_segments + n_segs > max_segments:
            break
        shapes.append(shape)
        total_segments += n_segs

    return shapes


class SyntheticBezierDataset(Dataset):
    """
    On-the-fly synthetic dataset that generates random geometric shapes.

    Returns the same (curve_tensor, image_tensor) format as BezierDataset,
    making it a drop-in replacement for training.
    """

    def __init__(
        self,
        length: int = 100_000,
        canvas_size: int = 256,
        max_segments: int = 256,
        min_shapes: int = 1,
        max_shapes: int = 10,
        base_seed: int = 42,
    ):
        """
        Args:
            length: Virtual dataset length (number of samples per epoch).
            canvas_size: Canvas width and height in pixels.
            max_segments: Maximum bezier segments per sample (must match model config).
            min_shapes: Minimum shapes per scene.
            max_shapes: Maximum shapes per scene.
            base_seed: Base seed; each index gets seed = base_seed + idx.
        """
        self.length = length
        self.canvas_size = canvas_size
        self.max_segments = max_segments
        self.min_shapes = min_shapes
        self.max_shapes = max_shapes
        self.base_seed = base_seed

        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov3-vits16-pretrain-lvd1689m"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Deterministic per-index seed for reproducibility within an epoch
        seed = self.base_seed + idx

        shapes = generate_random_scene(
            canvas_w=self.canvas_size,
            canvas_h=self.canvas_size,
            min_shapes=self.min_shapes,
            max_shapes=self.max_shapes,
            max_segments=self.max_segments,
            seed=seed,
        )

        # Convert shapes to tensor
        curve_tensor = shapes_to_tensor(
            shapes,
            self.canvas_size,
            self.canvas_size,
            max_segments=self.max_segments,
        )

        # Render to SVG then to raster image
        svg_content = save_bezier_shapes_to_svg(
            shapes, self.canvas_size, self.canvas_size
        )
        image = render_svg_bg(svg_content).convert("RGB")

        # Process through DINOv3 image processor
        image_tensor = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]

        return curve_tensor, image_tensor


class SyntheticSamplingDataset(SyntheticBezierDataset):
    """
    Small synthetic dataset for validation sampling visualization.
    """

    def __init__(self, num_samples: int = 8, **kwargs):
        super().__init__(length=num_samples, **kwargs)
