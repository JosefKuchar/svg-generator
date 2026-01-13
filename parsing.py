# TODO: Handle gradients

import torch
from io import StringIO
from loguru import logger
from datasets import load_dataset, Dataset
import tempfile
import subprocess
import pathlib
from tqdm import tqdm
import json

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
)
from raster import render_svg, calculate_mse


class BezierShape:
    def __init__(
        self,
        curves,
        color=None,
        opacity=1.0,
    ):
        self.curves = curves
        self.color = color
        self.opacity = opacity

    def __repr__(self):
        buf = ""
        buf += f"BezierShape(curves={len(self.curves)}, "
        buf += f"color={self.color}, opacity={self.opacity})"
        return buf

    def to_tensor(self, width, height, max_seq_len=512):
        if len(self.curves) > max_seq_len:
            logger.warning(f"Shape has more curves than max_seq_len, truncating")
            self.curves = self.curves[:max_seq_len]

        t = torch.zeros([max_seq_len, 13])

        vx = 0
        vy = 0
        vw = width
        vh = height

        cx = vx + (vw / 2.0)
        cy = vy + (vh / 2.0)

        scale = 2.0 / max(vw, vh)

        def norm_point(x, y):
            return (x - cx) * scale, (y - cy) * scale

        color = self.color if self.color is not None else (0, 0, 0)
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

            t[i, 11] = 2 * opacity - 1
            t[i, 12] = 1  # Real data flag

        # Padding
        for i in range(len(self.curves), max_seq_len):
            t[i, :] = t[i - 1, :].clone()
            t[i, 12] = -1

        return t

    @classmethod
    def from_tensor(cls, width, height, tensor):
        # 1. Setup Denormalization Parameters
        vx, vy = 0, 0
        vw, vh = width, height

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

        valid_mask = tensor[:, 12] > 0
        valid_rows = tensor[valid_mask]

        if len(valid_rows) == 0:
            # Return an empty shape if no valid rows found
            return cls(curves=[])

        # 3. Extract Attributes (Assumed uniform, take from first valid row)
        first_row = valid_rows[0]

        color = denorm_color(
            float(first_row[8]), float(first_row[9]), float(first_row[10])
        )

        # Extract Opacity (Inverse of: 2 * opacity - 1)
        opacity_norm = float(first_row[11])
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
            opacity=opacity,
        )


def calculate_signed_area_approx(curves):
    """
    Calculates approximate signed area of a list of bezier curves
    using the Shoelace formula on control points.
    Result < 0 usually implies Clockwise in SVG (Y-down) coordinates.
    """
    area = 0.0

    poly = []
    for curve in curves:
        p0, p1, p2, p3 = curve
        poly.extend([p0, p1, p2, p3])

    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += (x2 - x1) * (y2 + y1)

    return area / 2.0


def point_in_bezier_path(px, py, curves, samples_per_curve=10):
    """
    Check if a point (px, py) is inside a closed bezier path using ray casting.
    Samples the bezier curves to approximate them as a polyline.
    """
    if not curves:
        return False

    polyline = []
    for curve in curves:
        p0, p1, p2, p3 = curve
        for i in range(samples_per_curve):
            t = i / samples_per_curve
            mt = 1 - t
            x = (
                mt**3 * p0[0]
                + 3 * mt**2 * t * p1[0]
                + 3 * mt * t**2 * p2[0]
                + t**3 * p3[0]
            )
            y = (
                mt**3 * p0[1]
                + 3 * mt**2 * t * p1[1]
                + 3 * mt * t**2 * p2[1]
                + t**3 * p3[1]
            )
            polyline.append((x, y))

    last_curve = curves[-1]
    polyline.append(last_curve[3])

    n = len(polyline)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polyline[i]
        xj, yj = polyline[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def normalize_contour_winding(curves):
    """
    Re-orients subpaths to be compatible with non-zero fill rule.
    Bodies (Depth 0, 2...) -> Clockwise
    Holes (Depth 1, 3...) -> Counter-Clockwise
    """
    if not curves:
        return []

    subpaths = []
    current_subpath = []

    for curve in curves:
        p0, p1, p2, p3 = curve
        if current_subpath:
            prev_p3 = current_subpath[-1][3]
            if p0 != prev_p3:
                subpaths.append(current_subpath)
                current_subpath = []
        current_subpath.append(curve)

    if current_subpath:
        subpaths.append(current_subpath)

    subpaths = [sp for sp in subpaths if sp]
    n = len(subpaths)
    depths = [0] * n

    for i in range(n):
        c = subpaths[i][0]
        mx = 0.125 * c[0][0] + 0.375 * c[1][0] + 0.375 * c[2][0] + 0.125 * c[3][0]
        my = 0.125 * c[0][1] + 0.375 * c[1][1] + 0.375 * c[2][1] + 0.125 * c[3][1]

        for j in range(n):
            if i == j:
                continue

            if point_in_bezier_path(mx, my, subpaths[j]):
                depths[i] += 1

    # 4. Re-orient based on depth
    normalized_curves = []

    for i, sp in enumerate(subpaths):
        area = calculate_signed_area_approx(sp)

        is_hole = depths[i] % 2 == 1
        is_clockwise = area < 0

        should_reverse = False
        if not is_hole and not is_clockwise:
            should_reverse = True
        elif is_hole and is_clockwise:
            should_reverse = True

        if should_reverse:
            reversed_sp = []
            for curve in reversed(sp):
                p0, p1, p2, p3 = curve
                reversed_sp.append((p3, p2, p1, p0))
            normalized_curves.extend(reversed_sp)
        else:
            normalized_curves.extend(sp)

    return normalized_curves


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


def parse_numeric_value(value, default=None, is_percentage_normalized=False):
    """
    Parse a numeric value that might be a string with percentage (e.g., '50%').

    Args:
        value: The value to parse (can be int, float, or string like '50%')
        default: Default value if parsing fails
        is_percentage_normalized: If True, '50%' becomes 0.5; if False, '50%' becomes 50

    Returns:
        Float value or default
    """
    if value is None:
        return default

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value = value.strip()
        if value.endswith("%"):
            try:
                num = float(value[:-1])
                return num / 100.0 if is_percentage_normalized else num
            except ValueError:
                return default
        try:
            return float(value)
        except ValueError:
            return default

    return default


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
            opacity = element.values.get("opacity", 1.0)
            if "fill-opacity" in element.values:
                opacity = element.values["fill-opacity"]
            opacity = parse_numeric_value(
                opacity, default=1.0, is_percentage_normalized=True
            )
            fill_rule = (
                element.values.get("fill-rule", "nonzero")
                if isinstance(element.values, dict)
                else "nonzero"
            )

            path = Path(element)
            path.reify()

            bezier_curves = []
            for segment in path:
                bezier_curves.extend(get_cubic_bezier_segments(segment))

            if fill_rule == "evenodd":
                bezier_curves = normalize_contour_winding(bezier_curves)

            shape_data = BezierShape(
                curves=bezier_curves,
                color=fill_color,
                opacity=opacity,
            )

            output_data.append(shape_data)

    return output_data


def save_bezier_shapes_to_svg(shapes, width, height):
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
    ]

    def format_color(rgb_tuple):
        if rgb_tuple is None:
            return "none"
        r = max(0, min(255, int(rgb_tuple[0])))
        g = max(0, min(255, int(rgb_tuple[1])))
        b = max(0, min(255, int(rgb_tuple[2])))
        return f"#{r:02x}{g:02x}{b:02x}"

    for shape in shapes:
        if not shape.curves:
            continue

        path_commands = []
        last_pos = None

        for p0, p1, p2, p3 in shape.curves:
            if last_pos != p0:
                path_commands.append(f"M {p0[0]},{p0[1]}")
            path_commands.append(f"C {p1[0]},{p1[1]} {p2[0]},{p2[1]} {p3[0]},{p3[1]}")
            last_pos = p3
        d_str = " ".join(path_commands)

        fill = format_color(shape.color)
        opacity = shape.opacity if shape.opacity is not None else 1.0

        path_tag = f'<path d="{d_str}" ' f'fill="{fill}" ' f'opacity="{opacity}" '

        path_tag += "/>"
        lines.append(f"  {path_tag}")

    lines.append("</svg>")

    # 5. Write file
    content = "\n".join(lines)
    return content


def convert_svg_strings(svg_strings):
    processed_outputs = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        commands = []

        for i, svg_str in enumerate(svg_strings):
            input_file = tmp_path / f"input_{i}.svg"
            output_file = tmp_path / f"output_{i}.svg"
            input_file.write_text(svg_str, encoding="utf-8")
            cmd = (
                f"file-open:{input_file.absolute()}; "
                f"select-all:all; "
                f"selection-ungroup; "
                f"object-to-path; "
                f"object-stroke-to-path; "
                f"export-type:svg; "
                f"export-filename:{output_file.absolute()}; "
                f"export-do; "
                f"file-close"
            )
            commands.append(cmd)

        full_script = "\n".join(commands) + "\nquit\n"
        process = subprocess.Popen(
            ["/home/xkuchar/opt/inkscape", "--shell"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout, stderr = process.communicate(input=full_script)
        for i in range(len(svg_strings)):
            output_file = tmp_path / f"output_{i}.svg"
            if output_file.exists():
                processed_outputs.append(output_file.read_text(encoding="utf-8"))
            else:
                print(f"Error processing file {i}: {stderr}")
                processed_outputs.append(None)

    return processed_outputs


def process_batch(batch, converted_batch, processed_items):
    for original_svg, data2 in zip(batch, converted_batch):
        if data2 is None:
            continue

        try:
            elements = SVG.parse(StringIO(data2))
            bezier_curves = convert_svg_to_bezier_curves(elements)

            # Collect all shapes from this SVG
            shapes = []
            for shape in bezier_curves:
                shapes.append(
                    {
                        "curves": shape.curves,
                        "color": shape.color,
                        "opacity": shape.opacity,
                    }
                )

            processed_items.append(
                {
                    "original_svg": original_svg,
                    "shapes": json.dumps(shapes),
                    "width": elements.width,
                    "height": elements.height,
                }
            )
        except Exception as e:
            logger.warning(f"Error processing SVG: {e}")
            continue


def process_split(dataset_split, split_name):
    """Process a single dataset split and return the processed items."""
    batch_size = 100
    batch = []
    processed_items = []

    for item in tqdm(dataset_split, desc=f"Processing {split_name}"):
        data = item["item_svg"]

        # Skip if there is gradient fill
        if "radialGradient" in data or "linearGradient" in data:
            continue

        # Skip if there is mask
        if "<mask" in data:
            continue

        # Skip if there is style
        if "<style" in data:
            continue

        batch.append(data)

        if len(batch) >= batch_size:
            converted_batch = convert_svg_strings(batch)
            process_batch(batch, converted_batch, processed_items)
            batch = []

    # Process remaining items
    if batch:
        converted_batch = convert_svg_strings(batch)
        process_batch(batch, converted_batch, processed_items)

    return processed_items


if __name__ == "__main__":
    from datasets import DatasetDict

    # Load the full dataset with all splits
    full_dataset = load_dataset("mikronai/svg-svgrepo")

    processed_splits = {}

    # Process each available split
    for split_name in full_dataset.keys():
        print(f"\n=== Processing {split_name} split ===")
        processed_items = process_split(full_dataset[split_name], split_name)
        processed_splits[split_name] = Dataset.from_list(processed_items)
        print(f"Processed {len(processed_items)} items for {split_name} split")

    # Create and save DatasetDict with all splits
    final_dataset = DatasetDict(processed_splits)
    final_dataset.save_to_disk("bezier_dataset")

    print(f"\n=== Summary ===")
    for split_name, split_data in final_dataset.items():
        print(f"{split_name}: {len(split_data)} items")
    print(f"Saved all splits to bezier_dataset")
