"""
Vector image representation using homogenous bezier curves

Representation for the flow-matching model:
- Each segment is represented as (x0, y0, x1, y1, x2, y2, x3, y3, r, g, b, opacity, path_start, subpath_start, real)
- x0, y0, x3, y3 are start and end points of the curve
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


class BezierPath:
    """
    Bezier path with multiple curves
    """

    def __init__(
        self,
        curves: list[tuple[int, int, int, int, int, int, int, int]],
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
        Tensor of shape (max_segments, 15) where each row is:
        (x0, y0, x1, y1, x2, y2, x3, y3, r, g, b, opacity, path_start, subpath_start, real)
        All values normalized to [-1, 1].
    """
    segments = []
    cx = width / 2.0
    cy = height / 2.0
    scale = 2.0 / max(width, height)

    def norm_point(x, y):
        return (x - cx) * scale, (y - cy) * scale

    for shape_idx, shape in enumerate(shapes):
        r = (shape.color[0] / 255.0) * 2 - 1
        g = (shape.color[1] / 255.0) * 2 - 1
        b = (shape.color[2] / 255.0) * 2 - 1
        opacity = shape.opacity * 2 - 1

        for path_idx, path in enumerate(shape.paths):
            for curve_idx, curve in enumerate(path.curves):
                (x0, y0), (x1, y1), (x2, y2), (x3, y3) = curve

                x0_norm, y0_norm = norm_point(x0, y0)
                x1_norm, y1_norm = norm_point(x1, y1)
                x2_norm, y2_norm = norm_point(x2, y2)
                x3_norm, y3_norm = norm_point(x3, y3)

                path_start = 1.0 if path_idx == 0 and curve_idx == 0 else -1.0
                subpath_start = 1.0 if curve_idx == 0 else -1.0

                segment = [
                    x0_norm,
                    y0_norm,
                    x1_norm,
                    y1_norm,
                    x2_norm,
                    y2_norm,
                    x3_norm,
                    y3_norm,
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
        segments.append([0.0] * 14 + [-1.0])

    return torch.tensor(segments, dtype=torch.float32)


def _smooth_path_curves(curves: list) -> list:
    """
    Smooth consecutive curves cyclically by averaging end of previous with start of current.
    This reduces noise from model predictions.
    """
    if len(curves) <= 1:
        return curves

    smoothed = list(curves)

    for i in range(len(smoothed)):
        prev_idx = (i - 1) % len(smoothed)
        prev_curve = smoothed[prev_idx]
        curr_curve = smoothed[i]

        avg_x = (prev_curve[3][0] + curr_curve[0][0]) / 2
        avg_y = (prev_curve[3][1] + curr_curve[0][1]) / 2

        smoothed[prev_idx] = (
            prev_curve[0],
            prev_curve[1],
            prev_curve[2],
            (avg_x, avg_y),
        )

        smoothed[i] = ((avg_x, avg_y), curr_curve[1], curr_curve[2], curr_curve[3])

    return smoothed


def tensor_to_shapes(
    tensor: torch.Tensor, width: int, height: int
) -> list[BezierShape]:
    """
    Convert a tensor back to a list of BezierShape objects.

    Args:
        tensor: Tensor of shape (max_segments, 15) where each row is:
            (x0, y0, x1, y1, x2, y2, x3, y3, r, g, b, opacity, path_start, subpath_start, real)
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
    current_path_curves = []
    shape_colors = []

    for i in range(tensor.shape[0]):
        segment = tensor[i]

        # Extract and threshold flags
        path_start = segment[12].item() > 0
        subpath_start = segment[13].item() > 0
        real = segment[14].item() > 0

        if not real:
            continue

        x0_norm, y0_norm = segment[0].item(), segment[1].item()
        x1_norm, y1_norm = segment[2].item(), segment[3].item()
        x2_norm, y2_norm = segment[4].item(), segment[5].item()
        x3_norm, y3_norm = segment[6].item(), segment[7].item()

        x0, y0 = denorm_point(x0_norm, y0_norm)
        x1, y1 = denorm_point(x1_norm, y1_norm)
        x2, y2 = denorm_point(x2_norm, y2_norm)
        x3, y3 = denorm_point(x3_norm, y3_norm)

        curve = ((x0, y0), (x1, y1), (x2, y2), (x3, y3))

        r_norm = segment[8].item()
        g_norm = segment[9].item()
        b_norm = segment[10].item()
        opacity_norm = segment[11].item()

        if path_start:
            if current_path_curves:
                smoothed_curves = _smooth_path_curves(current_path_curves)
                current_shape_paths.append(BezierPath(smoothed_curves))
            if current_shape_paths and shape_colors:
                avg_r = sum(c[0] for c in shape_colors) / len(shape_colors)
                avg_g = sum(c[1] for c in shape_colors) / len(shape_colors)
                avg_b = sum(c[2] for c in shape_colors) / len(shape_colors)
                avg_opacity = sum(c[3] for c in shape_colors) / len(shape_colors)

                color = (denorm_color(avg_r), denorm_color(avg_g), denorm_color(avg_b))
                opacity = denorm_opacity(avg_opacity)
                shapes.append(BezierShape(current_shape_paths, color, opacity))

            current_shape_paths = []
            current_path_curves = [curve]
            shape_colors = [(r_norm, g_norm, b_norm, opacity_norm)]
        elif subpath_start:
            if current_path_curves:
                smoothed_curves = _smooth_path_curves(current_path_curves)
                current_shape_paths.append(BezierPath(smoothed_curves))
            current_path_curves = [curve]
            shape_colors.append((r_norm, g_norm, b_norm, opacity_norm))
        else:
            current_path_curves.append(curve)
            shape_colors.append((r_norm, g_norm, b_norm, opacity_norm))

    if current_path_curves:
        smoothed_curves = _smooth_path_curves(current_path_curves)
        current_shape_paths.append(BezierPath(smoothed_curves))
    if current_shape_paths and shape_colors:
        avg_r = sum(c[0] for c in shape_colors) / len(shape_colors)
        avg_g = sum(c[1] for c in shape_colors) / len(shape_colors)
        avg_b = sum(c[2] for c in shape_colors) / len(shape_colors)
        avg_opacity = sum(c[3] for c in shape_colors) / len(shape_colors)

        color = (denorm_color(avg_r), denorm_color(avg_g), denorm_color(avg_b))
        opacity = denorm_opacity(avg_opacity)
        shapes.append(BezierShape(current_shape_paths, color, opacity))

    return shapes
