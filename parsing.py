import torch

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

    def to_tensor(self, viewbox=None, max_seq_len=64):
        t = torch.zeros([max_seq_len, 17])

        scale = 1.0
        cx, cy = 0.0, 0.0

        vx = getattr(
            viewbox, "x", viewbox[0] if isinstance(viewbox, (tuple, list)) else 0
        )
        vy = getattr(
            viewbox, "y", viewbox[1] if isinstance(viewbox, (tuple, list)) else 0
        )
        vw = getattr(
            viewbox,
            "width",
            viewbox[2] if isinstance(viewbox, (tuple, list)) else 1,
        )
        vh = getattr(
            viewbox,
            "height",
            viewbox[3] if isinstance(viewbox, (tuple, list)) else 1,
        )

        # Calculate Center
        cx = vx + (vw / 2.0)
        cy = vy + (vh / 2.0)

        # Calculate Scale (based on largest dimension to preserve aspect ratio)
        max_dim = max(vw, vh)
        scale = 2.0 / max_dim

        def norm_point(x, y):
            if viewbox:
                return (x - cx) * scale, (y - cy) * scale
            return x, y

        color = self.color if self.color is not None else (0, 0, 0)
        stroke_color = self.stroke_color if self.stroke_color is not None else (0, 0, 0)
        stroke_width = self.stroke_width if self.stroke_width is not None else 1.0
        if stroke_width >= 100.0:
            stroke_width = 100.0
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


# Example usage
if __name__ == "__main__":
    elements = SVG.parse("svgs/test.svg")
    print(elements)

    # Extract all paths and convert to bezier curves
    bezier_curves = convert_svg_to_bezier_curves(elements)

    # Print results
    print(f"Found {len(bezier_curves)} bezier curve(s)")
    print(bezier_curves)

    # Convert to tensor
    t = bezier_curves[0].to_tensor(viewbox=elements.viewbox)
    print(t.shape)
    print(t)
