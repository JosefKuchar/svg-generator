"""
Vector image representation using homogenous bezier curves

Representation for the flow-matching model:
- Each segment is represented as (x0, y0, x1, y1, x2, y2, r, g, b, opacity, path_start, subpath_start, real)
- x0, y0 are the start points of the curve
- x1, y1, x2, y2 are control points
- r, g, b are color of the curve
- opacity is opacity of the curve
- path_start translates to new <path> element in svg
- subpath_start translates to new M (move) command inside <path>
- real flag is used to differentiate between real data and padding
- Every dimension is normalized to [-1, 1]
"""

import torch
from loguru import logger

Point = tuple[float, float]
Curve = tuple[Point, Point, Point, Point]
PartialCurve = tuple[Point, Point, Point]


class BezierPath:
    """
    Bezier path with multiple curves
    """

    def __init__(
        self,
        curves: list[Curve],
    ):
        self.curves = curves

    def __repr__(self):
        buf = ""
        buf += f"BezierPath(curves={len(self.curves)})"
        return buf


class BezierShape:
    """
    Bezier shape with multiple paths
    """

    def __init__(
        self,
        paths: list[BezierPath],
        color: tuple[int, int, int],
        opacity: float,
    ):
        self.paths = paths
        self.color = color
        self.opacity = opacity

    def __repr__(self):
        buf = ""
        buf += f"BezierShape(paths={len(self.paths)}, "
        buf += f"color={self.color}, opacity={self.opacity})"
        return buf


def shapes_to_tensor(shapes: list[BezierShape], width, height, max_segments=100):
    """
    Convert list of BezierShape objects to a tensor.

    Returns:
        Tensor of shape (max_segments, 13) where each row is:
        (x0, y0, x1, y1, x2, y2, r, g, b, opacity, path_start, subpath_start, real)
        All values normalized to [-1, 1].
    """
    segments = []
    cx = width / 2.0
    cy = height / 2.0
    scale = 2.0 / max(width, height)

    def norm_point(x, y):
        return (x - cx) * scale, (y - cy) * scale

    for shape in shapes:
        r = (shape.color[0] / 255.0) * 2 - 1
        g = (shape.color[1] / 255.0) * 2 - 1
        b = (shape.color[2] / 255.0) * 2 - 1
        opacity = shape.opacity * 2 - 1

        for path_idx, path in enumerate(shape.paths):
            for curve_idx, curve in enumerate(path.curves):
                (x0, y0), (x1, y1), (x2, y2), _ = curve

                x0_norm, y0_norm = norm_point(x0, y0)
                x1_norm, y1_norm = norm_point(x1, y1)
                x2_norm, y2_norm = norm_point(x2, y2)
                path_start = 1.0 if path_idx == 0 and curve_idx == 0 else -1.0
                subpath_start = 1.0 if curve_idx == 0 else -1.0

                segment = [
                    x0_norm,
                    y0_norm,
                    x1_norm,
                    y1_norm,
                    x2_norm,
                    y2_norm,
                    r,
                    g,
                    b,
                    opacity,
                    path_start,
                    subpath_start,
                    1.0,
                ]
                segments.append(segment)

    if len(segments) > max_segments:
        logger.warning(f"Truncating {len(segments)} segments to {max_segments}")
        segments = segments[:max_segments]

    while len(segments) < max_segments:
        segments.append([0.0] * 12 + [-1.0])

    return torch.tensor(segments, dtype=torch.float32)


def tensor_to_shapes(
    tensor: torch.Tensor, width: int, height: int
) -> list[BezierShape]:
    """
    Convert a tensor back to a list of BezierShape objects.

    Args:
        tensor: Tensor of shape (max_segments, 13) where each row is:
            (x0, y0, x1, y1, x2, y2, r, g, b, opacity, path_start, subpath_start, real)
            All values normalized to [-1, 1].
        width: Original image width
        height: Original image height

    Returns:
        List of BezierShape objects with denormalized coordinates, averaged colors/opacity,
        and flags thresholded by 0.
    """
    cx = width / 2.0
    cy = height / 2.0
    scale = 2.0 / max(width, height)

    def denorm_point(x_norm, y_norm):
        return x_norm / scale + cx, y_norm / scale + cy

    def denorm_color(value):
        return int(((value + 1) / 2) * 255)

    def denorm_opacity(value):
        return (value + 1) / 2

    shapes = []
    current_shape_paths = []
    current_path_segments = []
    shape_colors = []

    def _close_loop(segments: list[PartialCurve]) -> list[Curve]:
        if not segments:
            return []

        curves = []
        for i, (p0, p1, p2) in enumerate(segments):
            next_idx = (i + 1) % len(segments)
            next_p0 = segments[next_idx][0]
            curves.append((p0, p1, p2, next_p0))
        return curves

    for i in range(tensor.shape[0]):
        segment = tensor[i]

        # Extract and threshold flags
        path_start = segment[10].item() > 0
        subpath_start = segment[11].item() > 0
        real = segment[12].item() > 0

        if not real:
            continue

        x0_norm, y0_norm = segment[0].item(), segment[1].item()
        x1_norm, y1_norm = segment[2].item(), segment[3].item()
        x2_norm, y2_norm = segment[4].item(), segment[5].item()

        x0, y0 = denorm_point(x0_norm, y0_norm)
        x1, y1 = denorm_point(x1_norm, y1_norm)
        x2, y2 = denorm_point(x2_norm, y2_norm)

        curve_segment = ((x0, y0), (x1, y1), (x2, y2))

        r_norm = segment[6].item()
        g_norm = segment[7].item()
        b_norm = segment[8].item()
        opacity_norm = segment[9].item()

        if path_start:
            if current_path_segments:
                closed_curves = _close_loop(current_path_segments)
                current_shape_paths.append(BezierPath(closed_curves))
            if current_shape_paths and shape_colors:
                avg_r = sum(c[0] for c in shape_colors) / len(shape_colors)
                avg_g = sum(c[1] for c in shape_colors) / len(shape_colors)
                avg_b = sum(c[2] for c in shape_colors) / len(shape_colors)
                avg_opacity = sum(c[3] for c in shape_colors) / len(shape_colors)

                color = (denorm_color(avg_r), denorm_color(avg_g), denorm_color(avg_b))
                opacity = denorm_opacity(avg_opacity)
                shapes.append(BezierShape(current_shape_paths, color, opacity))

            current_shape_paths = []
            current_path_segments = [curve_segment]
            shape_colors = [(r_norm, g_norm, b_norm, opacity_norm)]
        elif subpath_start:
            if current_path_segments:
                closed_curves = _close_loop(current_path_segments)
                current_shape_paths.append(BezierPath(closed_curves))
            current_path_segments = [curve_segment]
            shape_colors.append((r_norm, g_norm, b_norm, opacity_norm))
        else:
            current_path_segments.append(curve_segment)
            shape_colors.append((r_norm, g_norm, b_norm, opacity_norm))

    if current_path_segments:
        closed_curves = _close_loop(current_path_segments)
        current_shape_paths.append(BezierPath(closed_curves))
    if current_shape_paths and shape_colors:
        avg_r = sum(c[0] for c in shape_colors) / len(shape_colors)
        avg_g = sum(c[1] for c in shape_colors) / len(shape_colors)
        avg_b = sum(c[2] for c in shape_colors) / len(shape_colors)
        avg_opacity = sum(c[3] for c in shape_colors) / len(shape_colors)

        color = (denorm_color(avg_r), denorm_color(avg_g), denorm_color(avg_b))
        opacity = denorm_opacity(avg_opacity)
        shapes.append(BezierShape(current_shape_paths, color, opacity))

    return shapes
