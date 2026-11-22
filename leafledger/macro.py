"""Plan and register local, partial-field macro captures.

The survey photograph covers the whole mounted specimen. A macro camera only
covers a small physical rectangle (for example 38 x 28 mm). Two utilities live
here:

* :func:`plan_macro_frame` chooses an actually reachable rectangle in the
  registered survey plane that holds the priority region as completely as
  possible, then maximizes the secondary region, and reports what stays outside.
* :func:`detail_registration` / :func:`preview_detail` align a locally captured
  photograph with three or more hand-matched point pairs using an affine map,
  so the warped detail can feed the existing measurement pipeline.
"""
import numpy as np
from PIL import Image, ImageDraw

from .art import ACCENT, INK, SHADOW, base_rgba
from .geometry import (
    affine_fit,
    point_pairs_parameter,
    polygon_mask,
    warp_detail,
)
from .model import (
    InputError,
    dictionary,
    identity,
    scalar,
)
from .photos import open_working


PRIMARY_OUTLINE = (246, 139, 96)
SECONDARY_OUTLINE = (120, 190, 231)
FRAME_OUTLINE = (244, 226, 156)
FRAME_FILL = (244, 226, 156, 38)
OUTSIDE_FILL = (236, 121, 125, 70)
COARSE_GRID = 64
REFINE_GRID = 40
REFINE_SPAN = 0.06
MIN_FRAME_SIDE_FRACTION = 0.01


def _find_region(tile, region_id, label):
    if not region_id:
        raise InputError(f"Choose a {label} region before planning the macro frame.")
    region = next((item for item in tile["regions"] if item["id"] == region_id), None)
    if region is None:
        raise InputError(f"The {label} region has been removed from this specimen.")
    if region["purpose"] == "ignore":
        raise InputError(f"The {label} region is an exclusion and cannot be photographed as a target.")
    return region


def planner_payload(payload):
    data = dictionary(payload, "Macro plan")
    width_mm = scalar(data.get("width_mm"), "Macro field width", lower=1.0, upper=100000.0)
    height_mm = scalar(data.get("height_mm"), "Macro field height", lower=1.0, upper=100000.0)
    primary_id = identity(data.get("primary_region_id"), "Priority region ID")
    secondary_id = identity(data.get("secondary_region_id"), "Secondary region ID") if data.get(
        "secondary_region_id"
    ) else None
    return width_mm, height_mm, primary_id, secondary_id


def _physical_fractions(tile, width_mm, height_mm):
    if tile.get("width_mm") is None or tile.get("height_mm") is None:
        raise InputError("Enter the specimen width and height in millimetres before planning a macro frame.")
    plane = tile["plane"]
    plane_w_mm = float(tile["width_mm"])
    plane_h_mm = float(tile["height_mm"])
    # The registered plane keeps the specimen's physical aspect ratio, so the
    # physical field maps to constant normalized fractions on each axis.
    frac_w = width_mm / plane_w_mm
    frac_h = height_mm / plane_h_mm
    if frac_w > 1.0 or frac_h > 1.0:
        raise InputError("The macro field is larger than the registered specimen plane.")
    if min(frac_w, frac_h) < MIN_FRAME_SIDE_FRACTION:
        raise InputError("The macro field is too small relative to the specimen to locate reliably.")
    size = plane["size"]
    return frac_w, frac_h, plane_w_mm, plane_h_mm, tuple(size)


def _region_mask(extent, region):
    return polygon_mask(extent, region["vertices"])


def _coverage(mask, left, top, right, bottom):
    height, width = mask.shape
    x0 = max(0, int(np.floor(left * width)))
    y0 = max(0, int(np.floor(top * height)))
    x1 = min(width, int(np.ceil(right * width)))
    y1 = min(height, int(np.ceil(bottom * height)))
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(np.count_nonzero(mask[y0:y1, x0:x1]))


def _candidate_grid(frac_w, frac_h):
    xs = np.linspace(0.0, 1.0 - frac_w, COARSE_GRID + 1)
    ys = np.linspace(0.0, 1.0 - frac_h, COARSE_GRID + 1)
    return [(float(x), float(y)) for y in ys for x in xs]


def _best_placement(primary, secondary, frac_w, frac_h):
    primary_total = int(np.count_nonzero(primary))
    secondary_total = int(np.count_nonzero(secondary))
    if primary_total == 0:
        raise InputError("The priority region has no area inside the registered plane.")
    best = None
    for left, top in _candidate_grid(frac_w, frac_h):
        right, bottom = left + frac_w, top + frac_h
        primary_in = _coverage(primary, left, top, right, bottom)
        secondary_in = _coverage(secondary, left, top, right, bottom)
        score = (primary_in, secondary_in, -left, -top)
        if best is None or score > best[0]:
            best = (score, left, top, primary_in, secondary_in)
    # Local refinement around the coarse winner on a finer, normalized grid.
    _, coarse_left, coarse_top, _, _ = best
    span_x = min(REFINE_SPAN, frac_w)
    span_y = min(REFINE_SPAN, frac_h)
    local_x = np.linspace(max(0.0, coarse_left - span_x / 2), min(1.0 - frac_w, coarse_left + span_x / 2), REFINE_GRID)
    local_y = np.linspace(max(0.0, coarse_top - span_y / 2), min(1.0 - frac_h, coarse_top + span_y / 2), REFINE_GRID)
    for top in local_y:
        for left in local_x:
            right, bottom = float(left) + frac_w, float(top) + frac_h
            primary_in = _coverage(primary, left, top, right, bottom)
            secondary_in = _coverage(secondary, left, top, right, bottom)
            score = (primary_in, secondary_in, -float(left), -float(top))
            if score > best[0]:
                best = (score, float(left), float(top), primary_in, secondary_in)
    _, left, top, primary_in, secondary_in = best
    return {
        "left": round(left, 6),
        "top": round(top, 6),
        "right": round(left + frac_w, 6),
        "bottom": round(top + frac_h, 6),
        "primary_in": primary_in,
        "primary_total": primary_total,
        "secondary_in": secondary_in,
        "secondary_total": secondary_total,
    }


def plan_macro_frame(tile, payload):
    if tile.get("plane") is None:
        raise InputError("Set a base photo and mark the specimen plane before planning a macro frame.")
    width_mm, height_mm, primary_id, secondary_id = planner_payload(payload)
    primary = _find_region(tile, primary_id, "priority")
    secondary = _find_region(tile, secondary_id, "secondary") if secondary_id else None
    frac_w, frac_h, plane_w_mm, plane_h_mm, extent = _physical_fractions(tile, width_mm, height_mm)
    primary_mask = _region_mask(extent, primary)
    if secondary is not None and secondary["id"] == primary["id"]:
        raise InputError("Choose a different secondary region, or leave it unset.")
    secondary_mask = _region_mask(extent, secondary) if secondary else np.zeros(
        (extent[1], extent[0]), dtype=bool
    )
    placement = _best_placement(primary_mask, secondary_mask, frac_w, frac_h)
    frame = {key: placement[key] for key in ("left", "top", "right", "bottom")}
    primary_miss = placement["primary_total"] - placement["primary_in"]
    secondary_miss = placement["secondary_total"] - placement["secondary_in"]
    scale_x, scale_y = plane_w_mm / extent[0], plane_h_mm / extent[1]
    plan = {
        "field_mm": [width_mm, height_mm],
        "frame": frame,
        "frame_size_px": [
            round((frame["right"] - frame["left"]) * extent[0], 2),
            round((frame["bottom"] - frame["top"]) * extent[1], 2),
        ],
        "primary_region": {"id": primary["id"], "name": primary["name"]},
        "secondary_region": {"id": secondary["id"], "name": secondary["name"]} if secondary else None,
        "primary": {
            "covered_px": placement["primary_in"],
            "total_px": placement["primary_total"],
            "outside_px": primary_miss,
            "covered_percent": round(100 * placement["primary_in"] / placement["primary_total"], 2),
            "covered_mm2": round(placement["primary_in"] * scale_x * scale_y, 3),
            "outside_mm2": round(primary_miss * scale_x * scale_y, 3),
            "fully_inside": primary_miss == 0,
        },
        "secondary": {
            "covered_px": placement["secondary_in"],
            "total_px": placement["secondary_total"],
            "outside_px": secondary_miss,
            "covered_percent": round(100 * placement["secondary_in"] / placement["secondary_total"], 2)
            if placement["secondary_total"]
            else None,
            "covered_mm2": round(placement["secondary_in"] * scale_x * scale_y, 3),
            "outside_mm2": round(secondary_miss * scale_x * scale_y, 3),
        }
        if secondary
        else None,
        "extent": list(extent),
        "plane_mm": [plane_w_mm, plane_h_mm],
    }
    return plan


def _draw_rectangle(draw, extent, frame, outline, fill=None, width=3):
    width_px, height_px = extent
    box = (
        frame["left"] * width_px,
        frame["top"] * height_px,
        frame["right"] * width_px,
        frame["bottom"] * height_px,
    )
    if fill is not None:
        draw.rectangle(box, fill=fill)
    draw.rectangle(box, outline=outline, width=width)


def _uncovered_mask(extent, region, frame):
    """Boolean mask of the part of a region that the macro frame does not reach."""
    region_mask = polygon_mask(extent, region["vertices"])
    width, height = extent
    inside = np.zeros((height, width), dtype=bool)
    x0 = max(0, int(np.floor(frame["left"] * width)))
    y0 = max(0, int(np.floor(frame["top"] * height)))
    x1 = min(width, int(np.ceil(frame["right"] * width)))
    y1 = min(height, int(np.ceil(frame["bottom"] * height)))
    inside[y0:y1, x0:x1] = True
    return region_mask & ~inside


def _hatch_layer(mask, color, *, spacing=12):
    """Diagonal hatching over a boolean mask as an RGBA array."""
    height, width = mask.shape
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    yy, xx = np.ogrid[:height, :width]
    hatch = ((xx + yy) % spacing < 2) & mask
    layer[hatch] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(layer, "RGBA")


def plan_overlay(frame_image, plan, tile):
    """Annotate the base survey image with the reachable macro frame."""
    image = base_rgba(frame_image).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    extent = image.size
    primary_record = next(
        (item for item in tile["regions"] if item["id"] == plan["primary_region"]["id"]), None
    )
    # Hatch the priority area that remains outside the reachable frame.
    if primary_record is not None and not plan["primary"]["fully_inside"]:
        uncovered = _uncovered_mask(extent, primary_record, plan["frame"])
        overlay = Image.alpha_composite(overlay, _hatch_layer(uncovered, OUTSIDE_FILL))
        draw = ImageDraw.Draw(overlay)
    for region, outline in (
        (plan["primary_region"], PRIMARY_OUTLINE),
        (plan["secondary_region"], SECONDARY_OUTLINE),
    ):
        if region is None:
            continue
        record = next((item for item in tile["regions"] if item["id"] == region["id"]), None)
        if record is None:
            continue
        points = [(x * extent[0], y * extent[1]) for x, y in record["vertices"]]
        draw.line([*points, points[0]], fill=outline + (235,), width=3, joint="curve")
    _draw_rectangle(draw, extent, plan["frame"], FRAME_OUTLINE, FRAME_FILL, width=4)
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)
    primary = plan["primary"]
    label = f"Macro {plan['field_mm'][0]:g} x {plan['field_mm'][1]:g} mm"
    draw.text((plan["frame"]["left"] * extent[0] + 6, plan["frame"]["top"] * extent[1] + 5), label,
              fill=INK, stroke_width=2, stroke_fill=SHADOW)
    if not primary["fully_inside"]:
        draw.text((8, 8), f"Hatched: {primary['outside_px']} px of the priority region stay outside the frame",
                  fill=(255, 205, 196), stroke_width=2, stroke_fill=SHADOW)
    return image.convert("RGB")


def detail_points(payload):
    data = dictionary(payload, "Detail registration")
    pairs = point_pairs_parameter(data.get("pairs"))
    return [pair["plane"] for pair in pairs], [pair["capture"] for pair in pairs], pairs


def detail_transform(tile, photo_id, payload):
    if tile.get("plane") is None:
        raise InputError("Set a base photo and mark the specimen plane before aligning a detail.")
    photo_id = identity(photo_id, "Detail photo ID")
    photo = next((item for item in tile["photos"] if item["id"] == photo_id), None)
    if photo is None:
        raise InputError("The detail capture was removed from this specimen.")
    if photo_id == tile["plane"]["photo_id"]:
        raise InputError("The base survey photo is already the full plane; register a separate detail capture.")
    plane_points, capture_points, pairs = detail_points(payload)
    matrix, residuals = affine_fit(plane_points, capture_points)
    return photo, matrix, residuals, pairs


def preview_detail(blobs, tile, photo_id, payload):
    photo, matrix, residuals, pairs = detail_transform(tile, photo_id, payload)
    destination = tuple(tile["plane"]["size"])
    rgb, alpha = open_working(blobs, photo)
    linear, valid, clipped = warp_detail(rgb, alpha, destination, matrix)
    from .geometry import SampleFrame
    from .model import physical_scale
    from .photos import linear_to_srgb
    frame = SampleFrame(linear, valid, photo, tuple(matrix.ravel()),
                        physical_scale(tile, *destination), clipped)
    image = base_rgba(frame).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width_px, height_px = image.size
    for index, pair in enumerate(pairs, start=1):
        x, y = pair["plane"][0] * width_px, pair["plane"][1] * height_px
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=ACCENT, outline=SHADOW, width=2)
        draw.text((x + 7, y - 7), str(index), fill=INK, stroke_width=2, stroke_fill=SHADOW)
    diagnostics = {
        "support_percent": round(100 * float(valid.mean()), 3),
        "supported_px": int(np.count_nonzero(valid)),
        "point_residuals": residuals,
        "max_residual": round(float(max(residuals)), 6),
        "frame_size": list(destination),
    }
    return image.convert("RGB"), diagnostics, matrix
