import torch
from io import StringIO
from loguru import logger

from svgelements import (
    SVG,
    Path,
    Group,
    Shape,
    Line,
    Arc,
    CubicBezier,
    QuadraticBezier,
    Close,
    Color,
)
from raster import render_svg, calculate_mse


class BezierShape:
    def __init__(
        self,
        curves,
        color=None,
        stroke_color=None,
        stroke_width=None,
        opacity=1.0,
        fill_rule=None,
    ):
        self.curves = curves
        self.color = color
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.opacity = opacity
        self.fill_rule = fill_rule

    def __repr__(self):
        return (
            f"BezierShape(curves={len(self.curves)}, "
            f"color={self.color}, stroke_color={self.stroke_color}, "
            f"stroke_width={self.stroke_width}, opacity={self.opacity}, "
            f"fill_rule={self.fill_rule})"
        )

    def to_tensor(self, viewbox=None, max_seq_len=512):
        if len(self.curves) > max_seq_len:
            logger.warning(f"Shape has more curves than max_seq_len, truncating")
            self.curves = self.curves[:max_seq_len]

        t = torch.zeros([max_seq_len, 17])

        vx = viewbox.x
        vy = viewbox.y
        vw = viewbox.width
        vh = viewbox.height

        cx = vx + (vw / 2.0)
        cy = vy + (vh / 2.0)

        scale = 2.0 / max(vw, vh)

        def norm_point(x, y):
            return (x - cx) * scale, (y - cy) * scale

        color = self.color if self.color is not None else (0, 0, 0)
        stroke_color = self.stroke_color if self.stroke_color is not None else (0, 0, 0)
        stroke_width = self.stroke_width if self.stroke_width is not None else 1.0
        if self.stroke_color is None:
            stroke_width = 0.0
        if stroke_width >= 100.0:
            logger.warning(f"Stroke width is greater than 100, clamping to 100")
            stroke_width = 100.0
        if stroke_width < 0.0:
            logger.warning(f"Stroke width is less than 0, clamping to 0")
            stroke_width = 0.0
        opacity = self.opacity if self.opacity is not None else 1.0

        for i, curve in enumerate(self.curves):
            # curve structure: ((x0,y0), (x1,y1), (x2,y2), (x3,y3))
            p0, p1, p2, p3 = curve

            # Normalize Coordinates
            nx0, ny0 = norm_point(p0[0], p0[1])
            nx1, ny1 = norm_point(p1[0], p1[1])
            nx2, ny2 = norm_point(p2[0], p2[1])
            nx3, ny3 = norm_point(p3[0], p3[1])

            # Assign to Tensor
            t[i, 0], t[i, 1] = nx0, ny0
            t[i, 2], t[i, 3] = nx1, ny1
            t[i, 4], t[i, 5] = nx2, ny2
            t[i, 6], t[i, 7] = nx3, ny3

            # Attributes (Normalized -1 to 1)
            t[i, 8] = 2 * (color[0] / 255.0) - 1
            t[i, 9] = 2 * (color[1] / 255.0) - 1
            t[i, 10] = 2 * (color[2] / 255.0) - 1

            t[i, 11] = 2 * (stroke_color[0] / 255.0) - 1
            t[i, 12] = 2 * (stroke_color[1] / 255.0) - 1
            t[i, 13] = 2 * (stroke_color[2] / 255.0) - 1

            t[i, 14] = 2 * (stroke_width / 100.0) - 1
            t[i, 15] = 2 * opacity - 1
            t[i, 16] = 1  # Real data flag

        # Padding
        for i in range(len(self.curves), max_seq_len):
            t[i, :] = t[i - 1, :].clone()
            t[i, 16] = -1

        return t

    @classmethod
    def from_tensor(cls, viewbox, tensor):
        # 1. Setup Denormalization Parameters
        vx, vy = viewbox.x, viewbox.y
        vw, vh = viewbox.width, viewbox.height

        # Recalculate center and scale exactly as done in to_tensor
        cx = vx + (vw / 2.0)
        cy = vy + (vh / 2.0)
        scale = 2.0 / max(vw, vh)

        def denorm_point(nx, ny):
            return (nx / scale) + cx, (ny / scale) + cy

        def denorm_color(n_r, n_g, n_b):
            # Inverse of: 2 * (val / 255.0) - 1
            r = int(round(((n_r + 1) / 2.0) * 255.0))
            g = int(round(((n_g + 1) / 2.0) * 255.0))
            b = int(round(((n_b + 1) / 2.0) * 255.0))
            return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

        # 2. Identify Real Data
        # Column 16 is the mask: 1.0 = Real Data, -1.0 = Padding
        if tensor.is_cuda:
            tensor = tensor.cpu()

        valid_mask = tensor[:, 16] > 0
        valid_rows = tensor[valid_mask]

        if len(valid_rows) == 0:
            # Return an empty shape if no valid rows found
            return cls(curves=[])

        # 3. Extract Attributes (Assumed uniform, take from first valid row)
        first_row = valid_rows[0]

        color = denorm_color(
            float(first_row[8]), float(first_row[9]), float(first_row[10])
        )
        stroke_color = denorm_color(
            float(first_row[11]), float(first_row[12]), float(first_row[13])
        )

        # Extract Stroke Width (Inverse of: 2 * (w / 100.0) - 1)
        stroke_width_norm = float(first_row[14])
        stroke_width = ((stroke_width_norm + 1) / 2.0) * 100.0
        stroke_width = max(0.0, stroke_width)

        # Extract Opacity (Inverse of: 2 * opacity - 1)
        opacity_norm = float(first_row[15])
        opacity = (opacity_norm + 1) / 2.0
        opacity = max(0.0, min(1.0, opacity))

        # 4. Extract Curves
        # Columns 0-7 contain the control points (x0, y0, x1, y1, x2, y2, x3, y3)
        curves = []
        for row in valid_rows:
            p0 = denorm_point(float(row[0]), float(row[1]))
            p1 = denorm_point(float(row[2]), float(row[3]))
            p2 = denorm_point(float(row[4]), float(row[5]))
            p3 = denorm_point(float(row[6]), float(row[7]))
            curves.append((p0, p1, p2, p3))

        # 5. Return new instance
        return cls(
            curves=curves,
            color=color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            opacity=opacity,
            fill_rule=None,
        )


def get_cubic_bezier_segments(segment):
    """
    Helper: Converts a single path segment into a list of cubic bezier tuples.
    Returns: List of [(p0, p1, p2, p3), ...]
    """
    curves = []

    if isinstance(segment, (Line, Close)):
        if segment.start == segment.end:
            return []

        start = (segment.start.x, segment.start.y)
        end = (segment.end.x, segment.end.y)

        # Convert Line to Cubic Bezier
        p1 = (start[0] + (end[0] - start[0]) / 3, start[1] + (end[1] - start[1]) / 3)
        p2 = (
            start[0] + 2 * (end[0] - start[0]) / 3,
            start[1] + 2 * (end[1] - start[1]) / 3,
        )
        curves.append((start, p1, p2, end))

    elif isinstance(segment, QuadraticBezier):
        p0 = (segment.start.x, segment.start.y)
        pc = (segment.control.x, segment.control.y)
        p2 = (segment.end.x, segment.end.y)

        new_p1 = (p0[0] + (2 / 3) * (pc[0] - p0[0]), p0[1] + (2 / 3) * (pc[1] - p0[1]))
        new_p2 = (p2[0] + (2 / 3) * (pc[0] - p2[0]), p2[1] + (2 / 3) * (pc[1] - p2[1]))

        curves.append((p0, new_p1, new_p2, p2))

    elif isinstance(segment, CubicBezier):
        curves.append(
            (
                (segment.start.x, segment.start.y),
                (segment.control1.x, segment.control1.y),
                (segment.control2.x, segment.control2.y),
                (segment.end.x, segment.end.y),
            )
        )

    elif isinstance(segment, Arc):
        for curve in segment.as_cubic_curves():
            curves.append(
                (
                    (curve.start.x, curve.start.y),
                    (curve.control1.x, curve.control1.y),
                    (curve.control2.x, curve.control2.y),
                    (curve.end.x, curve.end.y),
                )
            )

    return curves


def parse_color(svg_color):
    if not svg_color or svg_color.value is None:
        return None

    return (int(svg_color.red), int(svg_color.green), int(svg_color.blue))


def convert_svg_to_bezier_curves(svg_input):
    output_data = []

    stack = (
        list(svg_input) if isinstance(svg_input, (list, SVG, Group)) else [svg_input]
    )

    while stack:
        element = stack.pop(0)

        if isinstance(element, (Group, SVG)):
            stack[0:0] = list(element)
            continue

        if isinstance(element, Shape):
            fill_color = parse_color(element.fill)
            stroke_color = parse_color(element.stroke)
            stroke_width = getattr(element, "stroke_width", None)
            opacity = getattr(element, "opacity", 1.0)
            fill_rule = getattr(element, "fill_rule", None)

            path = Path(element)
            path.reify()

            bezier_curves = []
            for segment in path:
                bezier_curves.extend(get_cubic_bezier_segments(segment))

            shape_data = BezierShape(
                curves=bezier_curves,
                color=fill_color,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                opacity=opacity,
                fill_rule=fill_rule,
            )

            output_data.append(shape_data)

    return output_data


def save_bezier_shapes_to_svg(shapes, viewbox):
    # 1. Setup SVG Header
    vx, vy = viewbox.x, viewbox.y
    vw, vh = viewbox.width, viewbox.height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{vx} {vy} {vw} {vh}" width="{vw}" height="{vh}">'
    ]

    # 2. Helper to format colors
    def format_color(rgb_tuple):
        if rgb_tuple is None:
            return "none"
        # Clamp values to 0-255 just in case
        r = max(0, min(255, int(rgb_tuple[0])))
        g = max(0, min(255, int(rgb_tuple[1])))
        b = max(0, min(255, int(rgb_tuple[2])))
        return f"#{r:02x}{g:02x}{b:02x}"

    # 3. Process each shape
    for shape in shapes:
        if not shape.curves:
            continue

        # Build path data string (d attribute)
        path_commands = []
        last_pos = None

        for p0, p1, p2, p3 in shape.curves:
            # If the start of this curve (p0) is not where the last one ended,
            # we must explicitly Move (M) there.
            # We compare with a small epsilon for float stability if needed,
            # but direct equality check usually works for generated chains.
            if last_pos != p0:
                path_commands.append(f"M {p0[0]},{p0[1]}")

            # Cubic Bezier command (C x1 y1, x2 y2, x y)
            path_commands.append(f"C {p1[0]},{p1[1]} {p2[0]},{p2[1]} {p3[0]},{p3[1]}")

            last_pos = p3

        d_str = " ".join(path_commands)

        # 4. format Styles
        fill = format_color(shape.color)
        stroke = format_color(shape.stroke_color)
        stroke_width = shape.stroke_width if shape.stroke_width is not None else 0.0
        opacity = shape.opacity if shape.opacity is not None else 1.0

        # Build the <path> element
        # Note: If stroke color is None/transparent, stroke-width essentially doesn't matter,
        # but we preserve it.
        path_tag = (
            f'<path d="{d_str}" '
            f'fill="{fill}" '
            f'stroke="{stroke}" '
            f'stroke-width="{stroke_width}" '
            f'opacity="{opacity}" '
        )

        if shape.fill_rule:
            path_tag += f'fill-rule="{shape.fill_rule}" '

        path_tag += "/>"
        lines.append(f"  {path_tag}")

    lines.append("</svg>")

    # 5. Write file
    content = "\n".join(lines)
    return content


# Example usage
if __name__ == "__main__":

    file = "svgs/test3.svg"

    # Load original SVG string
    with open(file, "r") as f:
        original = f.read()

    # Parse SVG
    elements = SVG.parse(StringIO(original))

    # Extract all paths and convert to bezier curves
    bezier_curves = convert_svg_to_bezier_curves(elements)

    # Print results
    print(f"Found {len(bezier_curves)} bezier curve(s)")
    for curve in bezier_curves:
        print(curve)

    reconstructed = []
    for curve in bezier_curves:
        t = curve.to_tensor(viewbox=elements.viewbox)
        b = BezierShape.from_tensor(elements.viewbox, t)
        reconstructed.append(b)

    # Save reconstructed curves to SVG
    output = save_bezier_shapes_to_svg(reconstructed, elements.viewbox)

    reconstructed_image = render_svg(output)
    original_image = render_svg(original)

    # Calculate MSE
    mse = calculate_mse(original_image, reconstructed_image)
    print(f"MSE: {mse}")

    # Save reconstructed image
    reconstructed_image.save("reconstructed.png")
