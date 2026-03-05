"""
Synthetic dataset generator for random geometric shapes.

Generates random scenes of geometric primitives, organic blobs, and compound
shapes as BezierShape objects, compatible with the existing representation pipeline.
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


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


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


def _polygon_to_curves(vertices: list[Point]) -> list[Curve]:
    """Convert a closed polygon (vertex list) to cubic bezier line segments."""
    n = len(vertices)
    return [_line_to_cubic(vertices[i], vertices[(i + 1) % n]) for i in range(n)]


def _reverse_curves(curves: list[Curve]) -> list[Curve]:
    """
    Reverse the winding direction of a closed curve loop.

    Each cubic bezier (p0, p1, p2, p3) becomes (p3, p2, p1, p0), and the
    order of curves in the list is reversed. This flips CW <-> CCW.
    """
    return [(p3, p2, p1, p0) for p0, p1, p2, p3 in reversed(curves)]


def _curves_signed_area(curves: list[Curve]) -> float:
    """
    Compute the signed area enclosed by a closed bezier curve loop using
    the shoelace formula on segment endpoints. Positive = CCW, negative = CW.
    """
    area = 0.0
    for p0, _, _, p3 in curves:
        area += p0[0] * p3[1] - p3[0] * p0[1]
    return area / 2.0


def _ensure_ccw(curves: list[Curve]) -> list[Curve]:
    """Ensure curves wind counter-clockwise (positive signed area)."""
    if _curves_signed_area(curves) < 0:
        return _reverse_curves(curves)
    return curves


def _ensure_cw(curves: list[Curve]) -> list[Curve]:
    """Ensure curves wind clockwise (negative signed area) -- used for holes."""
    if _curves_signed_area(curves) > 0:
        return _reverse_curves(curves)
    return curves


def _curves_bounding_box(curves: list[Curve]) -> tuple[float, float, float, float]:
    """
    Compute axis-aligned bounding box of curves from endpoint positions.
    Returns (min_x, min_y, max_x, max_y).
    """
    xs = []
    ys = []
    for p0, p1, p2, p3 in curves:
        for p in (p0, p1, p2, p3):
            xs.append(p[0])
            ys.append(p[1])
    return min(xs), min(ys), max(xs), max(ys)


def _arc_points(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    start_angle: float,
    end_angle: float,
    n_segments: int,
) -> list[Point]:
    """Sample points along an elliptical arc (for building arc-based shapes)."""
    points = []
    for i in range(n_segments + 1):
        t = i / n_segments
        a = start_angle + t * (end_angle - start_angle)
        points.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return points


# ---------------------------------------------------------------------------
# Bezier fitting for smooth closed curves
# ---------------------------------------------------------------------------


def _fit_closed_cubic_beziers(
    points: list[Point], smoothness: float = 0.35
) -> list[Curve]:
    """
    Fit smooth cubic bezier curves through a closed loop of points.

    Uses Catmull-Rom-style tangent estimation: the tangent at each point is
    parallel to the vector from the previous point to the next, and the handle
    length is a fraction (`smoothness`) of the distance to the neighbor.

    This guarantees C1 continuity and, because points are ordered by angle
    (monotonically increasing), the resulting curve is non-self-intersecting.

    Args:
        points: Ordered vertices of the closed contour.
        smoothness: Handle length as fraction of neighbor distance (0=polygon, ~0.35=smooth).

    Returns:
        List of Curve tuples forming a smooth closed loop.
    """
    n = len(points)
    if n < 3:
        return _polygon_to_curves(points)

    curves: list[Curve] = []
    for i in range(n):
        p_prev = points[(i - 1) % n]
        p_curr = points[i]
        p_next = points[(i + 1) % n]
        p_next2 = points[(i + 2) % n]

        # Tangent at p_curr: direction from p_prev -> p_next
        tx = p_next[0] - p_prev[0]
        ty = p_next[1] - p_prev[1]
        t_len = math.hypot(tx, ty)
        if t_len < 1e-9:
            tx, ty = 1.0, 0.0
        else:
            tx /= t_len
            ty /= t_len

        # Tangent at p_next: direction from p_curr -> p_next2
        tx2 = p_next2[0] - p_curr[0]
        ty2 = p_next2[1] - p_curr[1]
        t2_len = math.hypot(tx2, ty2)
        if t2_len < 1e-9:
            tx2, ty2 = 1.0, 0.0
        else:
            tx2 /= t2_len
            ty2 /= t2_len

        # Distance between current and next point determines handle length
        seg_dist = math.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        handle = seg_dist * smoothness

        # Control point 1: along tangent at p_curr (forward)
        cp1 = (p_curr[0] + tx * handle, p_curr[1] + ty * handle)
        # Control point 2: along tangent at p_next (backward)
        cp2 = (p_next[0] - tx2 * handle, p_next[1] - ty2 * handle)

        curves.append((p_curr, cp1, cp2, p_next))

    return curves


# ---------------------------------------------------------------------------
# Primitive shape generators
# ---------------------------------------------------------------------------


def make_ellipse(
    cx: float, cy: float, rx: float, ry: float, angle: float = 0.0
) -> list[Curve]:
    """Create an ellipse as 4 cubic bezier curves."""
    kx = KAPPA * rx
    ky = KAPPA * ry

    curves = [
        ((cx + rx, cy), (cx + rx, cy + ky), (cx + kx, cy + ry), (cx, cy + ry)),
        ((cx, cy + ry), (cx - kx, cy + ry), (cx - rx, cy + ky), (cx - rx, cy)),
        ((cx - rx, cy), (cx - rx, cy - ky), (cx - kx, cy - ry), (cx, cy - ry)),
        ((cx, cy - ry), (cx + kx, cy - ry), (cx + rx, cy - ky), (cx + rx, cy)),
    ]

    if angle != 0.0:
        curves = _rotate_curves(curves, angle, cx, cy)

    return curves


def make_rectangle(
    cx: float, cy: float, w: float, h: float, angle: float = 0.0
) -> list[Curve]:
    """Create a rectangle as 4 cubic bezier line segments."""
    hw, hh = w / 2, h / 2
    corners = [
        (cx - hw, cy - hh),
        (cx + hw, cy - hh),
        (cx + hw, cy + hh),
        (cx - hw, cy + hh),
    ]
    curves = _polygon_to_curves(corners)
    if angle != 0.0:
        curves = _rotate_curves(curves, angle, cx, cy)
    return curves


def make_triangle(
    cx: float, cy: float, size: float, angle: float = 0.0
) -> list[Curve]:
    """Create an equilateral triangle as 3 cubic bezier line segments."""
    vertices = []
    for i in range(3):
        a = angle + i * 2 * math.pi / 3 - math.pi / 2
        vertices.append((cx + size * math.cos(a), cy + size * math.sin(a)))
    return _polygon_to_curves(vertices)


def make_regular_polygon(
    cx: float, cy: float, size: float, n_sides: int, angle: float = 0.0
) -> list[Curve]:
    """Create a regular polygon as cubic bezier line segments."""
    vertices = []
    for i in range(n_sides):
        a = angle + i * 2 * math.pi / n_sides - math.pi / 2
        vertices.append((cx + size * math.cos(a), cy + size * math.sin(a)))
    return _polygon_to_curves(vertices)


def make_star(
    cx: float,
    cy: float,
    outer_r: float,
    inner_r: float,
    n_points: int,
    angle: float = 0.0,
) -> list[Curve]:
    """Create a star shape as cubic bezier line segments."""
    vertices = []
    for i in range(2 * n_points):
        a = angle + i * math.pi / n_points - math.pi / 2
        r = outer_r if i % 2 == 0 else inner_r
        vertices.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return _polygon_to_curves(vertices)


# ---------------------------------------------------------------------------
# Organic (blob) shape generator
# ---------------------------------------------------------------------------


def make_blob(
    cx: float,
    cy: float,
    base_radius: float,
    n_points: int,
    irregularity: float,
    spikiness: float,
    angle: float = 0.0,
    rng: random.Random | None = None,
) -> list[Curve]:
    """
    Generate a smooth organic blob shape.

    Algorithm:
      1. Distribute N angles around the circle with random angular perturbation
         (controlled by `irregularity`).
      2. For each angle, sample a radius = base_radius + gaussian noise scaled
         by `spikiness`.
      3. Convert polar -> cartesian to get contour points.
      4. Fit smooth cubic beziers through the points.

    Non-self-intersection is guaranteed because angles are monotonically
    increasing, so the contour never crosses itself.

    Args:
        cx, cy: Center position.
        base_radius: Average radius of the blob.
        n_points: Number of contour vertices (more = more detail). 5-20 typical.
        irregularity: Angular perturbation in [0, 1]. 0=even spacing, 1=very uneven.
        spikiness: Radial perturbation in [0, 1]. 0=circle, 1=very spiky.
        angle: Overall rotation offset.
        rng: Random instance for reproducibility.

    Returns:
        List of Curve tuples forming a smooth closed blob.
    """
    r = rng or random

    # Generate irregularly spaced angles
    angle_step = 2 * math.pi / n_points
    max_angle_dev = irregularity * angle_step * 0.5

    angles = []
    cumulative = angle
    for _ in range(n_points):
        cumulative += angle_step + r.uniform(-max_angle_dev, max_angle_dev)
        angles.append(cumulative)

    # Normalize angles to span exactly 2*pi
    total = angles[-1] - angle
    if total > 0:
        scale_factor = 2 * math.pi / total
        angles = [angle + (a - angle) * scale_factor for a in angles]

    # Generate radii with gaussian perturbation
    max_spike = spikiness * base_radius * 0.5
    points: list[Point] = []
    for a in angles:
        radius = base_radius + r.gauss(0, max_spike)
        radius = max(base_radius * 0.15, radius)  # clamp to avoid degenerate shapes
        points.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))

    # Smoothness inversely related to spikiness: spiky blobs get tighter handles
    smoothness = 0.20 + (1.0 - spikiness) * 0.25
    return _fit_closed_cubic_beziers(points, smoothness=smoothness)


# ---------------------------------------------------------------------------
# Compound geometric shape generators
# ---------------------------------------------------------------------------


def make_l_shape(
    cx: float, cy: float, w: float, h: float, thickness: float, angle: float = 0.0
) -> list[Curve]:
    """
    Create an L-shape (6 vertices) centered at (cx, cy).

    Shape (before rotation):
        +---+
        |   |
        |   +------+
        |          |
        +----------+
    """
    hw, hh = w / 2, h / 2
    t = thickness
    vertices = [
        (cx - hw, cy - hh),
        (cx - hw + t, cy - hh),
        (cx - hw + t, cy + hh - t),
        (cx + hw, cy + hh - t),
        (cx + hw, cy + hh),
        (cx - hw, cy + hh),
    ]
    curves = _polygon_to_curves(vertices)
    if angle != 0.0:
        curves = _rotate_curves(curves, angle, cx, cy)
    return curves


def make_cross(
    cx: float, cy: float, size: float, thickness: float, angle: float = 0.0
) -> list[Curve]:
    """
    Create a plus/cross shape (12 vertices) centered at (cx, cy).
    """
    hs = size / 2
    ht = thickness / 2
    vertices = [
        (cx - ht, cy - hs),
        (cx + ht, cy - hs),
        (cx + ht, cy - ht),
        (cx + hs, cy - ht),
        (cx + hs, cy + ht),
        (cx + ht, cy + ht),
        (cx + ht, cy + hs),
        (cx - ht, cy + hs),
        (cx - ht, cy + ht),
        (cx - hs, cy + ht),
        (cx - hs, cy - ht),
        (cx - ht, cy - ht),
    ]
    curves = _polygon_to_curves(vertices)
    if angle != 0.0:
        curves = _rotate_curves(curves, angle, cx, cy)
    return curves


def make_arrow(
    cx: float, cy: float, length: float, head_w: float, shaft_w: float,
    head_len: float, angle: float = 0.0,
) -> list[Curve]:
    """
    Create a right-pointing arrow shape (7 vertices), centered at (cx, cy).

    Shape (before rotation)::

             +
            /|
      +----+ |
      |      +
      +----+ |
            /|
             +
    """
    hl = length / 2
    hs = shaft_w / 2
    hh = head_w / 2
    tip_x = cx + hl
    notch_x = tip_x - head_len
    vertices = [
        (cx - hl, cy - hs),      # shaft top-left
        (notch_x, cy - hs),      # shaft top-right / head notch
        (notch_x, cy - hh),      # head top
        (tip_x, cy),             # tip
        (notch_x, cy + hh),      # head bottom
        (notch_x, cy + hs),      # shaft bottom-right / head notch
        (cx - hl, cy + hs),      # shaft bottom-left
    ]
    curves = _polygon_to_curves(vertices)
    if angle != 0.0:
        curves = _rotate_curves(curves, angle, cx, cy)
    return curves


def make_crescent(
    cx: float, cy: float, outer_r: float, inner_r: float, offset: float,
    n_segments: int = 8, angle: float = 0.0,
) -> list[Curve]:
    """
    Create a crescent (moon) shape by tracing the outer arc forward and inner
    arc backward, with the inner circle offset horizontally.

    Args:
        cx, cy: Center of outer circle.
        outer_r: Outer circle radius.
        inner_r: Inner circle radius (should be <= outer_r).
        offset: Horizontal offset of inner circle center from outer center.
        n_segments: Segments per arc (more = smoother).
        angle: Rotation angle.

    Returns:
        List of Curve tuples forming a crescent.
    """
    # Outer arc: full circle left half (pi/2 to 3pi/2 before rotation)
    # Inner arc: matching portion, reversed
    # For simplicity, use ~270 degrees of outer and connect to inner
    arc_span = math.pi * 1.4  # ~252 degrees, leaving a gap for the crescent opening

    outer_pts = _arc_points(cx, cy, outer_r, outer_r, -arc_span / 2, arc_span / 2, n_segments)
    inner_pts = _arc_points(
        cx + offset, cy, inner_r, inner_r, arc_span / 2, -arc_span / 2, n_segments
    )

    # Combine into single contour: outer forward, inner backward
    all_points = outer_pts + inner_pts

    curves = _fit_closed_cubic_beziers(all_points, smoothness=0.35)
    if angle != 0.0:
        curves = _rotate_curves(curves, angle, cx, cy)
    return curves


def make_ring_sector(
    cx: float, cy: float, outer_r: float, inner_r: float,
    sweep_angle: float, n_segments: int = 6, angle: float = 0.0,
) -> list[Curve]:
    """
    Create a ring sector (arc segment with thickness), like a pac-man or
    gauge segment.

    Args:
        cx, cy: Center.
        outer_r, inner_r: Outer and inner radii.
        sweep_angle: Angular sweep in radians.
        n_segments: Segments per arc.
        angle: Rotation.

    Returns:
        List of Curve tuples.
    """
    half = sweep_angle / 2
    outer_pts = _arc_points(cx, cy, outer_r, outer_r, -half, half, n_segments)
    inner_pts = _arc_points(cx, cy, inner_r, inner_r, half, -half, n_segments)

    all_points = outer_pts + inner_pts
    curves = _fit_closed_cubic_beziers(all_points, smoothness=0.30)
    if angle != 0.0:
        curves = _rotate_curves(curves, angle, cx, cy)
    return curves


def make_rounded_rectangle(
    cx: float, cy: float, w: float, h: float, corner_r: float, angle: float = 0.0,
) -> list[Curve]:
    """
    Create a rounded rectangle using lines for edges and quarter-circle arcs
    for corners.

    Args:
        cx, cy: Center.
        w, h: Width and height.
        corner_r: Corner radius (clamped to half of min(w, h)).
        angle: Rotation.
    """
    hw, hh = w / 2, h / 2
    cr = min(corner_r, hw - 0.01, hh - 0.01)
    k = KAPPA * cr

    # Build clockwise: top edge -> top-right corner -> right edge -> ...
    curves: list[Curve] = []

    # Top edge (left to right)
    curves.append(_line_to_cubic((cx - hw + cr, cy - hh), (cx + hw - cr, cy - hh)))
    # Top-right corner
    curves.append((
        (cx + hw - cr, cy - hh),
        (cx + hw - cr + k, cy - hh),
        (cx + hw, cy - hh + cr - k),
        (cx + hw, cy - hh + cr),
    ))
    # Right edge (top to bottom)
    curves.append(_line_to_cubic((cx + hw, cy - hh + cr), (cx + hw, cy + hh - cr)))
    # Bottom-right corner
    curves.append((
        (cx + hw, cy + hh - cr),
        (cx + hw, cy + hh - cr + k),
        (cx + hw - cr + k, cy + hh),
        (cx + hw - cr, cy + hh),
    ))
    # Bottom edge (right to left)
    curves.append(_line_to_cubic((cx + hw - cr, cy + hh), (cx - hw + cr, cy + hh)))
    # Bottom-left corner
    curves.append((
        (cx - hw + cr, cy + hh),
        (cx - hw + cr - k, cy + hh),
        (cx - hw, cy + hh - cr + k),
        (cx - hw, cy + hh - cr),
    ))
    # Left edge (bottom to top)
    curves.append(_line_to_cubic((cx - hw, cy + hh - cr), (cx - hw, cy - hh + cr)))
    # Top-left corner
    curves.append((
        (cx - hw, cy - hh + cr),
        (cx - hw, cy - hh + cr - k),
        (cx - hw + cr - k, cy - hh),
        (cx - hw + cr, cy - hh),
    ))

    if angle != 0.0:
        curves = _rotate_curves(curves, angle, cx, cy)
    return curves


def make_trapezoid(
    cx: float, cy: float, top_w: float, bot_w: float, h: float, angle: float = 0.0,
) -> list[Curve]:
    """Create a trapezoid with different top and bottom widths."""
    htw, hbw, hh = top_w / 2, bot_w / 2, h / 2
    vertices = [
        (cx - htw, cy - hh),
        (cx + htw, cy - hh),
        (cx + hbw, cy + hh),
        (cx - hbw, cy + hh),
    ]
    curves = _polygon_to_curves(vertices)
    if angle != 0.0:
        curves = _rotate_curves(curves, angle, cx, cy)
    return curves


def make_parallelogram(
    cx: float, cy: float, w: float, h: float, skew: float, angle: float = 0.0,
) -> list[Curve]:
    """Create a parallelogram with horizontal skew."""
    hw, hh = w / 2, h / 2
    vertices = [
        (cx - hw + skew, cy - hh),
        (cx + hw + skew, cy - hh),
        (cx + hw - skew, cy + hh),
        (cx - hw - skew, cy + hh),
    ]
    curves = _polygon_to_curves(vertices)
    if angle != 0.0:
        curves = _rotate_curves(curves, angle, cx, cy)
    return curves


# ---------------------------------------------------------------------------
# Shape selection and scene generation
# ---------------------------------------------------------------------------

# All available shape types grouped by category
PRIMITIVE_SHAPES = ["circle", "oval", "square", "rectangle", "triangle",
                    "pentagon", "hexagon", "star"]
ORGANIC_SHAPES = ["blob_smooth", "blob_rough", "blob_spiky"]
COMPOUND_SHAPES = ["l_shape", "cross", "arrow", "crescent", "ring_sector",
                   "rounded_rect", "trapezoid", "parallelogram"]
ALL_SHAPES = PRIMITIVE_SHAPES + ORGANIC_SHAPES + COMPOUND_SHAPES


def _estimate_extent(shape_type: str, params: dict) -> float:
    """Estimate maximum extent (distance from center to farthest point) for a shape."""
    if shape_type in ("circle",):
        return params["radius"]
    elif shape_type == "oval":
        return max(params["rx"], params["ry"])
    elif shape_type == "square":
        return params["side"] * math.sqrt(2) / 2
    elif shape_type in ("rectangle", "rounded_rect"):
        w, h = params["w"], params["h"]
        return math.sqrt(w * w + h * h) / 2
    elif shape_type in ("triangle", "pentagon", "hexagon"):
        return params["size"]
    elif shape_type == "star":
        return params["outer_r"]
    elif shape_type in ("blob_smooth", "blob_rough", "blob_spiky"):
        return params["base_radius"] * 1.5  # conservative estimate with perturbation
    elif shape_type == "l_shape":
        w, h = params["w"], params["h"]
        return math.sqrt(w * w + h * h) / 2
    elif shape_type == "cross":
        return params["size"] * math.sqrt(2) / 2
    elif shape_type == "arrow":
        l = params["length"]
        hw = params["head_w"]
        return math.sqrt(l * l + hw * hw) / 2
    elif shape_type in ("crescent", "ring_sector"):
        return params["outer_r"]
    elif shape_type == "trapezoid":
        w = max(params["top_w"], params["bot_w"])
        h = params["h"]
        return math.sqrt(w * w + h * h) / 2
    elif shape_type == "parallelogram":
        w, h, skew = params["w"], params["h"], params["skew"]
        return math.sqrt((w / 2 + abs(skew)) ** 2 + (h / 2) ** 2)
    return params.get("size", params.get("base_radius", 50))


# ---------------------------------------------------------------------------
# Hole generation
# ---------------------------------------------------------------------------

# Shape types suitable for use as holes (must be simple closed contours).
# Crescents and ring_sectors are excluded -- they have concavities that make
# containment harder to guarantee.
HOLE_SHAPES = [
    "circle", "oval", "square", "rectangle", "triangle",
    "pentagon", "hexagon", "star",
    "blob_smooth", "blob_rough",
    "rounded_rect", "trapezoid", "parallelogram",
]


def _generate_hole_curves(
    outer_curves: list[Curve],
    rng: random.Random,
    max_scale: float = 0.55,
    min_scale: float = 0.15,
) -> list[Curve] | None:
    """
    Generate a hole shape that fits inside the outer contour.

    Strategy:
      1. Compute bounding box of the outer contour.
      2. Pick a random hole shape type.
      3. Generate the hole at the center of the bounding box with a size
         that is a random fraction (min_scale..max_scale) of the outer bbox.
      4. Apply a random sub-offset so the hole isn't always dead-center,
         but stays well within the outer bounds.
      5. Ensure the hole winds CW (opposite of the CCW outer contour) so
         the nonzero fill rule cuts it out.

    Returns:
        CW-wound curve list for the hole, or None if the outer shape is
        too small.
    """
    min_x, min_y, max_x, max_y = _curves_bounding_box(outer_curves)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    if bbox_w < 4 or bbox_h < 4:
        return None

    # Center of the outer shape's bounding box
    bbox_cx = (min_x + max_x) / 2
    bbox_cy = (min_y + max_y) / 2

    # Hole size relative to outer bbox
    scale = rng.uniform(min_scale, max_scale)
    hole_extent = min(bbox_w, bbox_h) * scale / 2  # half-size

    # Random offset from center, constrained so hole stays inside outer bbox.
    # The hole's bbox half-size is roughly hole_extent, so keep the center
    # within (bbox_center +/- (bbox_half - hole_extent * margin)).
    margin = 1.3  # safety factor
    max_dx = max(0.0, bbox_w / 2 - hole_extent * margin)
    max_dy = max(0.0, bbox_h / 2 - hole_extent * margin)
    hole_cx = bbox_cx + rng.uniform(-max_dx, max_dx)
    hole_cy = bbox_cy + rng.uniform(-max_dy, max_dy)

    angle = rng.uniform(0, 2 * math.pi)
    hole_type = rng.choice(HOLE_SHAPES)

    # Generate hole curves centered at (hole_cx, hole_cy)
    if hole_type == "circle":
        curves = make_ellipse(hole_cx, hole_cy, hole_extent, hole_extent)

    elif hole_type == "oval":
        rx = hole_extent * rng.uniform(0.5, 1.0)
        ry = hole_extent * rng.uniform(0.5, 1.0)
        curves = make_ellipse(hole_cx, hole_cy, rx, ry, angle=angle)

    elif hole_type == "square":
        # Side length: hole_extent * sqrt(2) would touch bbox corners, so
        # use hole_extent directly for a safe inscribed square.
        curves = make_rectangle(hole_cx, hole_cy, hole_extent * 1.3, hole_extent * 1.3, angle=angle)

    elif hole_type == "rectangle":
        w = hole_extent * rng.uniform(1.0, 1.6)
        h = hole_extent * rng.uniform(1.0, 1.6)
        curves = make_rectangle(hole_cx, hole_cy, w, h, angle=angle)

    elif hole_type == "triangle":
        curves = make_triangle(hole_cx, hole_cy, hole_extent, angle=angle)

    elif hole_type == "pentagon":
        curves = make_regular_polygon(hole_cx, hole_cy, hole_extent, 5, angle=angle)

    elif hole_type == "hexagon":
        curves = make_regular_polygon(hole_cx, hole_cy, hole_extent, 6, angle=angle)

    elif hole_type == "star":
        inner_r = hole_extent * rng.uniform(0.3, 0.6)
        n_points = rng.choice([4, 5, 6])
        curves = make_star(hole_cx, hole_cy, hole_extent, inner_r, n_points, angle=angle)

    elif hole_type == "blob_smooth":
        n_pts = rng.randint(5, 10)
        curves = make_blob(hole_cx, hole_cy, hole_extent * 0.8, n_pts,
                           irregularity=rng.uniform(0.0, 0.3),
                           spikiness=rng.uniform(0.05, 0.2),
                           angle=angle, rng=rng)

    elif hole_type == "blob_rough":
        n_pts = rng.randint(6, 12)
        curves = make_blob(hole_cx, hole_cy, hole_extent * 0.7, n_pts,
                           irregularity=rng.uniform(0.2, 0.5),
                           spikiness=rng.uniform(0.15, 0.4),
                           angle=angle, rng=rng)

    elif hole_type == "rounded_rect":
        w = hole_extent * rng.uniform(1.0, 1.5)
        h = hole_extent * rng.uniform(1.0, 1.5)
        cr = min(w, h) * rng.uniform(0.1, 0.4)
        curves = make_rounded_rectangle(hole_cx, hole_cy, w, h, cr, angle=angle)

    elif hole_type == "trapezoid":
        tw = hole_extent * rng.uniform(0.6, 1.2)
        bw = hole_extent * rng.uniform(0.8, 1.4)
        th = hole_extent * rng.uniform(0.8, 1.4)
        curves = make_trapezoid(hole_cx, hole_cy, tw, bw, th, angle=angle)

    elif hole_type == "parallelogram":
        w = hole_extent * rng.uniform(1.0, 1.5)
        h = hole_extent * rng.uniform(0.6, 1.2)
        skew = w * rng.uniform(0.1, 0.3)
        curves = make_parallelogram(hole_cx, hole_cy, w, h, skew, angle=angle)

    else:
        return None

    # Ensure hole winds CW (opposite of CCW outer contour)
    curves = _ensure_cw(curves)
    return curves


def generate_random_shape(
    canvas_w: float, canvas_h: float, rng: random.Random | None = None
) -> BezierShape:
    """
    Generate a single random shape within the given canvas.

    Randomly selects from primitive geometric shapes, smooth organic blobs,
    and compound geometric figures.
    """
    r = rng or random

    angle = r.uniform(0, 2 * math.pi)
    color = (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255))
    opacity = 1.0 if r.random() < 0.8 else r.uniform(0.2, 0.9)

    shape_type = r.choice(ALL_SHAPES)

    canvas_min = min(canvas_w, canvas_h)
    min_size = canvas_min * 0.05
    max_size = canvas_min * 0.40

    # ---- Build shape-specific parameters and curves ----
    params: dict = {}

    if shape_type == "circle":
        params["radius"] = r.uniform(min_size, max_size)

    elif shape_type == "oval":
        params["rx"] = r.uniform(min_size, max_size)
        params["ry"] = r.uniform(min_size, max_size)

    elif shape_type == "square":
        params["side"] = r.uniform(min_size * 2, max_size * 1.5)

    elif shape_type == "rectangle":
        params["w"] = r.uniform(min_size * 2, max_size * 1.5)
        params["h"] = r.uniform(min_size * 2, max_size * 1.5)

    elif shape_type == "triangle":
        params["size"] = r.uniform(min_size, max_size)

    elif shape_type == "pentagon":
        params["size"] = r.uniform(min_size, max_size)

    elif shape_type == "hexagon":
        params["size"] = r.uniform(min_size, max_size)

    elif shape_type == "star":
        params["outer_r"] = r.uniform(min_size, max_size)
        params["inner_r"] = r.uniform(params["outer_r"] * 0.3, params["outer_r"] * 0.6)
        params["n_points"] = r.choice([4, 5, 6, 7, 8])

    elif shape_type == "blob_smooth":
        params["base_radius"] = r.uniform(min_size, max_size)
        params["n_points"] = r.randint(6, 12)
        params["irregularity"] = r.uniform(0.0, 0.3)
        params["spikiness"] = r.uniform(0.05, 0.25)

    elif shape_type == "blob_rough":
        params["base_radius"] = r.uniform(min_size, max_size)
        params["n_points"] = r.randint(8, 18)
        params["irregularity"] = r.uniform(0.3, 0.7)
        params["spikiness"] = r.uniform(0.2, 0.5)

    elif shape_type == "blob_spiky":
        params["base_radius"] = r.uniform(min_size, max_size)
        params["n_points"] = r.randint(5, 10)
        params["irregularity"] = r.uniform(0.1, 0.5)
        params["spikiness"] = r.uniform(0.5, 0.9)

    elif shape_type == "l_shape":
        params["w"] = r.uniform(min_size * 1.5, max_size * 1.5)
        params["h"] = r.uniform(min_size * 1.5, max_size * 1.5)
        params["thickness"] = r.uniform(
            min(params["w"], params["h"]) * 0.25,
            min(params["w"], params["h"]) * 0.5,
        )

    elif shape_type == "cross":
        params["size"] = r.uniform(min_size * 2, max_size * 1.5)
        params["thickness"] = r.uniform(params["size"] * 0.2, params["size"] * 0.45)

    elif shape_type == "arrow":
        params["length"] = r.uniform(min_size * 2, max_size * 2)
        params["head_w"] = r.uniform(min_size * 0.8, max_size * 0.8)
        params["shaft_w"] = r.uniform(params["head_w"] * 0.25, params["head_w"] * 0.5)
        params["head_len"] = r.uniform(params["length"] * 0.2, params["length"] * 0.4)

    elif shape_type == "crescent":
        params["outer_r"] = r.uniform(min_size, max_size)
        params["inner_r"] = r.uniform(params["outer_r"] * 0.6, params["outer_r"] * 0.9)
        params["offset"] = r.uniform(params["outer_r"] * 0.2, params["outer_r"] * 0.5)

    elif shape_type == "ring_sector":
        params["outer_r"] = r.uniform(min_size, max_size)
        params["inner_r"] = r.uniform(params["outer_r"] * 0.4, params["outer_r"] * 0.7)
        params["sweep_angle"] = r.uniform(math.pi * 0.5, math.pi * 1.5)

    elif shape_type == "rounded_rect":
        params["w"] = r.uniform(min_size * 2, max_size * 1.5)
        params["h"] = r.uniform(min_size * 2, max_size * 1.5)
        max_cr = min(params["w"], params["h"]) * 0.45
        params["corner_r"] = r.uniform(max_cr * 0.2, max_cr)

    elif shape_type == "trapezoid":
        params["top_w"] = r.uniform(min_size, max_size * 1.2)
        params["bot_w"] = r.uniform(min_size, max_size * 1.2)
        params["h"] = r.uniform(min_size, max_size * 1.2)

    elif shape_type == "parallelogram":
        params["w"] = r.uniform(min_size * 2, max_size * 1.5)
        params["h"] = r.uniform(min_size, max_size)
        params["skew"] = r.uniform(params["w"] * 0.1, params["w"] * 0.35)

    # ---- Compute extent and constrain center position ----
    extent = _estimate_extent(shape_type, params)
    margin = max(extent * 0.7, canvas_min * 0.05)
    margin = min(margin, canvas_w / 2 - 1, canvas_h / 2 - 1)  # never invert bounds
    cx = r.uniform(margin, canvas_w - margin)
    cy = r.uniform(margin, canvas_h - margin)

    # ---- Build curves ----
    if shape_type == "circle":
        curves = make_ellipse(cx, cy, params["radius"], params["radius"], angle=0.0)
    elif shape_type == "oval":
        curves = make_ellipse(cx, cy, params["rx"], params["ry"], angle=angle)
    elif shape_type == "square":
        curves = make_rectangle(cx, cy, params["side"], params["side"], angle=angle)
    elif shape_type == "rectangle":
        curves = make_rectangle(cx, cy, params["w"], params["h"], angle=angle)
    elif shape_type == "triangle":
        curves = make_triangle(cx, cy, params["size"], angle=angle)
    elif shape_type == "pentagon":
        curves = make_regular_polygon(cx, cy, params["size"], 5, angle=angle)
    elif shape_type == "hexagon":
        curves = make_regular_polygon(cx, cy, params["size"], 6, angle=angle)
    elif shape_type == "star":
        curves = make_star(cx, cy, params["outer_r"], params["inner_r"],
                           params["n_points"], angle=angle)
    elif shape_type in ("blob_smooth", "blob_rough", "blob_spiky"):
        curves = make_blob(cx, cy, params["base_radius"], params["n_points"],
                           params["irregularity"], params["spikiness"],
                           angle=angle, rng=r)
    elif shape_type == "l_shape":
        curves = make_l_shape(cx, cy, params["w"], params["h"],
                              params["thickness"], angle=angle)
    elif shape_type == "cross":
        curves = make_cross(cx, cy, params["size"], params["thickness"], angle=angle)
    elif shape_type == "arrow":
        curves = make_arrow(cx, cy, params["length"], params["head_w"],
                            params["shaft_w"], params["head_len"], angle=angle)
    elif shape_type == "crescent":
        curves = make_crescent(cx, cy, params["outer_r"], params["inner_r"],
                               params["offset"], angle=angle)
    elif shape_type == "ring_sector":
        curves = make_ring_sector(cx, cy, params["outer_r"], params["inner_r"],
                                  params["sweep_angle"], angle=angle)
    elif shape_type == "rounded_rect":
        curves = make_rounded_rectangle(cx, cy, params["w"], params["h"],
                                        params["corner_r"], angle=angle)
    elif shape_type == "trapezoid":
        curves = make_trapezoid(cx, cy, params["top_w"], params["bot_w"],
                                params["h"], angle=angle)
    elif shape_type == "parallelogram":
        curves = make_parallelogram(cx, cy, params["w"], params["h"],
                                    params["skew"], angle=angle)
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

    # Ensure outer contour is CCW
    curves = _ensure_ccw(curves)

    # Randomly add holes (~25% chance, skip for very small or concave shapes)
    paths = [BezierPath(curves)]
    # Shapes that are already concave or thin — holes would look odd or escape
    skip_holes = {"crescent", "ring_sector", "star", "cross", "arrow", "l_shape"}
    if shape_type not in skip_holes and r.random() < 0.25:
        n_holes = r.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        for _ in range(n_holes):
            hole_curves = _generate_hole_curves(curves, rng=r)
            if hole_curves is not None:
                paths.append(BezierPath(hole_curves))

    return BezierShape(paths=paths, color=color, opacity=opacity)


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
        epoch: int = 0,
    ):
        """
        Args:
            length: Virtual dataset length (number of samples per epoch).
            canvas_size: Canvas width and height in pixels.
            max_segments: Maximum bezier segments per sample (must match model config).
            min_shapes: Minimum shapes per scene.
            max_shapes: Maximum shapes per scene.
            base_seed: Base seed; each index gets seed = base_seed + idx.
            epoch: Epoch offset used for seeding synthetic scenes.
        """
        self.length = length
        self.canvas_size = canvas_size
        self.max_segments = max_segments
        self.min_shapes = min_shapes
        self.max_shapes = max_shapes
        self.base_seed = base_seed
        self.epoch = epoch

        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov3-vits16-pretrain-lvd1689m"
        )

    def __len__(self):
        return self.length

    def set_epoch(self, epoch: int):
        print(f"Setting SyntheticBezierDataset epoch to {epoch}")
        self.epoch = epoch

    def __getitem__(self, idx):
        # Deterministic per-index seed, shifted each epoch.
        seed = self.base_seed + idx + self.epoch * self.length

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
