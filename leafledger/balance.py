"""Whole-specimen neutral calibration across multiple same-batch color cards.

A mounted herbarium sheet can sit under two lamps (for example a cool lamp on
the left and a warm lamp on the right), with one neutral card at each end.
The calibration always produces ONE set of per-channel gains applied to the
entire registered plane: local or side-specific gains would erase genuine
leaf color differences.  Each card is then checked against the laboratory's
neutrality line (red and blue each within 5% of green after correction).

When no single global fit brings every card across the 5% line, the best
effort fit is still retained.  The plane is partitioned into trusted color
zones (backed by a passing card, or inside the protected near-tip fading
reference) and calibration-limited zones.  Measurements keep the two classes
separate so a failed card is never mistaken for a real leaf color difference.
"""

import numpy as np
from PIL import Image, ImageDraw

from .art import font
from .color import float_list, swatch
from .geometry import polygon_mask
from .model import InputError
from .photos import linear_to_srgb, luminance, uint8_image


MIN_CARD_SAMPLES = 40
MIN_CARD_LEVEL = 0.012
MAX_CARD_CHROMA_SPREAD = 0.22
MIN_GAIN = 0.22
MAX_GAIN = 4.0
NEUTRAL_LINE = 0.05
TRUST_GRID = 8
MIN_TRUSTED_PIXELS = 60

TRUSTED = 1
LIMITED = 0
ZONE_COLORS = {"trusted": (96, 178, 138), "limited": (226, 138, 92), "card": (236, 232, 205)}


def card_sample(frame, neutral_region, candidate_area, target):
    """Robust median of one neutral card, reusing the single-card safeguards."""
    measured_area = polygon_mask(frame.extent, neutral_region["vertices"]) & candidate_area
    native_count = int(np.count_nonzero(measured_area & frame.valid))
    support = measured_area & frame.valid & ~frame.clipped
    count = int(np.count_nonzero(support))
    if count < MIN_CARD_SAMPLES:
        raise InputError(f"Neutral card '{neutral_region['name']}' needs at least 40 supported, unclipped pixels.")
    if count / max(native_count, 1) < 0.7:
        raise InputError(f"Neutral card '{neutral_region['name']}' is heavily clipped. Choose another photo or card region.")
    colors = frame.linear[support]
    card = np.median(colors, axis=0)
    if np.min(card) < MIN_CARD_LEVEL:
        raise InputError(f"Neutral card '{neutral_region['name']}' is too dark for stable channel correction.")
    levels = luminance(colors)
    floor, ceiling = np.quantile(levels, (0.10, 0.90))
    mid = float(luminance(card))
    if (ceiling - floor) / max(mid, 1e-8) > 0.38:
        raise InputError(f"Card brightness varies too much across '{neutral_region['name']}'. Select one evenly lit patch.")
    chroma = colors / np.maximum(levels[:, None], 0.001)
    spread = np.quantile(chroma, 0.9, axis=0) - np.quantile(chroma, 0.10, axis=0)
    if np.max(spread) > MAX_CARD_CHROMA_SPREAD:
        raise InputError(f"Card chroma varies too much across '{neutral_region['name']}'. Keep the selection inside the neutral patch.")
    gains = np.full(3, target, dtype=np.float32) / card
    if np.max(gains) > MAX_GAIN or np.min(gains) < MIN_GAIN:
        raise InputError(f"Card '{neutral_region['name']}' would require channel gains outside the supported range.")
    center = centroid(support)
    return {
        "region_id": neutral_region["id"],
        "name": neutral_region["name"],
        "vertices": [list(point) for point in neutral_region["vertices"]],
        "card_samples": count,
        "card_median_linear_rgb": float_list(card, 7),
        "before_swatch": swatch(card),
        "card_lightness_range": float_list((floor, mid, ceiling), 6),
        "centroid": [round(float(center[0]), 6), round(float(center[1]), 6)],
        "mask": support,
    }


def centroid(support):
    ys, xs = np.nonzero(support)
    return np.array([xs.mean(), ys.mean()], dtype=np.float32)


def solve_global_gains(cards, target, weights=None):
    """One global per-channel gain set for every card.

    Weighted least squares in log space on the within-card channel ratios,
    followed by a single luminance normalization onto the neutral target.
    With one card this is exactly the legacy neutral balance.
    """
    medians = np.asarray([card["card_median_linear_rgb"] for card in cards], dtype=np.float64)
    count = len(cards)
    if weights is None:
        weights = np.ones(count, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    log_cards = np.log(medians)
    center = np.eye(3) - np.ones((3, 3)) / 3.0
    log_bar = (weights[:, None] * log_cards).sum(axis=0)
    log_gains = -center @ log_bar
    gains = np.exp(log_gains).astype(np.float32)
    lumas = medians * gains[None, :]
    average_luma = float((weights @ lumas) @ np.array([0.2126, 0.7152, 0.0722]))
    gains *= np.float32(target / average_luma)
    if float(np.max(gains)) > MAX_GAIN or float(np.min(gains)) < MIN_GAIN:
        raise InputError("A single whole-specimen correction would need channel gains outside the supported range.")
    return gains


def assess_card(card, gains):
    """Red/green and blue/green departure from neutral after the global fit."""
    median = np.asarray(card["card_median_linear_rgb"], dtype=np.float64)
    corrected = median * np.asarray(gains, dtype=np.float64)
    red_green = float(corrected[0] / corrected[1] - 1.0)
    blue_green = float(corrected[2] / corrected[1] - 1.0)
    worst = max(abs(red_green), abs(blue_green))
    return {
        "region_id": card["region_id"],
        "name": card["name"],
        "centroid": card["centroid"],
        "card_samples": card["card_samples"],
        "before_swatch": card["before_swatch"],
        "after_swatch": swatch(corrected),
        "corrected_median_linear_rgb": float_list(corrected, 7),
        "red_vs_green_percent": round(100 * red_green, 3),
        "blue_vs_green_percent": round(100 * blue_green, 3),
        "worst_neutrality_percent": round(100 * worst, 3),
        "neutral_line_percent": round(100 * NEUTRAL_LINE, 3),
        "passes_line": bool(worst <= NEUTRAL_LINE),
    }


def nearest_card_zones(extent, cards):
    """Voronoi assignment of every supported pixel to its nearest card."""
    width, height = extent
    y, x = np.indices((height, width), dtype=np.float32)
    distances = np.zeros((height, width, len(cards)), dtype=np.float32)
    for index, card in enumerate(cards):
        cx, cy = card["centroid"]
        distances[:, :, index] = (x - cx) ** 2 + (y - cy) ** 2
    return np.argmin(distances, axis=2).astype(np.int8)

def region_mask(extent, region):
    if region is None:
        width, height = extent
        return np.zeros((height, width), dtype=bool)
    return polygon_mask(extent, region["vertices"])


def build_trust(extent, cards, assessments, zones, tip_region, grid_count=TRUST_GRID):
    """Trusted vs calibration-limited pixels, at grid-cell resolution.

    A cell is trusted when its dominant card passed the 5% line.  Cells tied
    to a failing card are calibration-limited, except the protected near-tip
    fading reference which is always carved out as trusted.
    """
    width, height = extent
    passed = np.array([item["passes_line"] for item in assessments], dtype=bool)
    cell_trusted = np.zeros((grid_count, grid_count), dtype=bool)
    cell_zone = np.zeros((grid_count, grid_count), dtype=np.int8)
    rows = np.rint(np.linspace(0, height, grid_count + 1)).astype(int)
    cols = np.rint(np.linspace(0, width, grid_count + 1)).astype(int)
    grid_trust = np.zeros((height, width), dtype=bool)
    grid_zone = np.zeros((height, width), dtype=np.int8)
    for y_index, (top, bottom) in enumerate(zip(rows, rows[1:])):
        for x_index, (left, right) in enumerate(zip(cols, cols[1:])):
            block = zones[top:bottom, left:right]
            local_zone = int(np.bincount(block.ravel(), minlength=len(cards)).argmax())
            cell_zone[y_index, x_index] = local_zone
            ok = bool(passed[local_zone])
            cell_trusted[y_index, x_index] = ok
            grid_zone[top:bottom, left:right] = local_zone
            if ok:
                grid_trust[top:bottom, left:right] = True
    tip = region_mask(extent, tip_region)
    tip_pixels = int(np.count_nonzero(tip))
    trusted = grid_trust.copy()
    if tip_pixels:
        trusted |= tip
    limited = ~trusted
    return {
        "trusted": trusted,
        "limited": limited,
        "tip_mask": tip,
        "grid_trust": grid_trust,
        "grid_zone": grid_zone,
        "cell_trusted": cell_trusted,
        "cell_zone": cell_zone,
        "grid_count": grid_count,
        "tip_region_id": tip_region["id"] if tip_region else None,
        "tip_pixels": tip_pixels,
    }


def calibration_record(frame, candidate_area, neutral_regions, tip_region, target, weights=None):
    """Full calibration record for one registered base capture."""
    if not 1 <= len(neutral_regions) <= 4:
        raise InputError("Select between one and four neutral color card regions.")
    cards = [card_sample(frame, region, candidate_area, target) for region in neutral_regions]
    gains = solve_global_gains(cards, target, weights)
    assessments = [assess_card(card, gains) for card in cards]
    zones = nearest_card_zones(frame.extent, cards)
    trust = build_trust(frame.extent, cards, assessments, zones, tip_region)
    all_pass = all(item["passes_line"] for item in assessments)
    record = {
        "schema": "leafledger.balance.v2",
        "mode": "single" if len(cards) == 1 else "multi-card",
        "target": target,
        "gains": float_list(gains, 7),
        "photo_id": frame.photo["id"],
        "neutral_line": NEUTRAL_LINE,
        "grid_count": trust["grid_count"],
        "all_cards_pass": bool(all_pass),
        "card_region_ids": [card["region_id"] for card in cards],
        "tip_region_id": trust["tip_region_id"],
        "cards": [
            {key: value for key, value in assessment.items()}
            for assessment in assessments
        ],
        "trust_summary": {
            "trusted_pixels": int(np.count_nonzero(trust["trusted"])),
            "calibration_limited_pixels": int(np.count_nonzero(trust["limited"])),
            "tip_protected_pixels": trust["tip_pixels"],
            "passing_card_regions": [a["region_id"] for a in assessments if a["passes_line"]],
            "failing_card_regions": [a["region_id"] for a in assessments if not a["passes_line"]],
        },
    }
    return record, trust, zones, cards


def reconstruct_trust(tile, record, extent):
    """Rebuild the trusted mask from a stored calibration record and live regions."""
    if not record or record.get("schema") != "leafledger.balance.v2":
        return None
    regions = {region["id"]: region for region in tile["regions"]}
    card_regions = [regions[region_id] for region_id in record["card_region_ids"] if region_id in regions]
    if len(card_regions) != len(record["card_region_ids"]):
        return None
    tip_region = regions.get(record["tip_region_id"]) if record.get("tip_region_id") else None
    cards = [
        {"region_id": region["id"], "centroid": mask_centroid(extent, region)}
        for region in card_regions
    ]
    zones = nearest_card_zones(extent, cards)
    assessments = [
        {"passes_line": region_id in record["trust_summary"]["passing_card_regions"]}
        for region_id in record["card_region_ids"]
    ]
    return build_trust(extent, cards, assessments, zones, tip_region, record.get("grid_count", TRUST_GRID))


def mask_centroid(extent, region):
    points = polygon_mask(extent, region["vertices"])
    ys, xs = np.nonzero(points)
    if not len(xs):
        vertices = np.asarray(region["vertices"], dtype=np.float32)
        return [float(vertices[:, 0].mean() * extent[0]), float(vertices[:, 1].mean() * extent[1])]
    return [float(xs.mean()), float(ys.mean())]


def apply_trust(support, trust):
    """Split a measurement support into trusted and calibration-limited parts."""
    if trust is None:
        return support, np.zeros_like(support)
    trusted = support & trust["trusted"]
    limited = support & trust["limited"]
    return trusted, limited


def require_trusted(trusted, context="This measurement"):
    if int(np.count_nonzero(trusted)) < MIN_TRUSTED_PIXELS:
        raise InputError(
            f"{context} has fewer than {MIN_TRUSTED_PIXELS} pixels backed by a card across the 5% neutrality line. "
            "Widen the sampling region or mark a near-tip fading reference."
        )


def trust_metrics(support, trusted, limited, record):
    total = max(int(np.count_nonzero(support)), 1)
    trusted_count = int(np.count_nonzero(trusted))
    limited_count = int(np.count_nonzero(limited))
    metrics = {
        "color_trust": "not-calibrated" if record is None else (
            "all-trusted" if record.get("all_cards_pass", True) else "zones"
        ),
        "trusted_pixels": trusted_count,
        "calibration_limited_pixels": limited_count,
        "trusted_percent": round(100 * trusted_count / total, 2),
        "calibration_limited_percent": round(100 * limited_count / total, 2),
        "neutral_line_percent": round(100 * record.get("neutral_line", NEUTRAL_LINE), 2) if record else None,
    }
    if record and record.get("schema") == "leafledger.balance.v2":
        metrics["cards"] = [
            {
                "name": card["name"],
                "passes_line": card["passes_line"],
                "red_vs_green_percent": card["red_vs_green_percent"],
                "blue_vs_green_percent": card["blue_vs_green_percent"],
                "worst_neutrality_percent": card["worst_neutrality_percent"],
            }
            for card in record["cards"]
        ]
    return metrics


def trust_note(record):
    if record is None:
        return "No neutral-card correction was applied."
    if record.get("schema") != "leafledger.balance.v2":
        return "Legacy single-card neutral balance; no 5% zone partition is attached."
    if record["all_cards_pass"]:
        return (
            f"All {len(record['cards'])} neutral card(s) pass the "
            f"{round(100 * record['neutral_line']):g}% red/blue-to-green line under one whole-specimen correction."
        )
    failed = ", ".join(card["name"] for card in record["cards"] if not card["passes_line"])
    return (
        "One global correction was retained, but these card(s) stay outside the 5% neutrality line: "
        f"{failed}. Calibration-limited zones are reported separately and must not be read as leaf color difference."
    )


def trust_overlay(frame, linear, record, trust, zones, cards):
    """Corrected plane with trusted zones outlined and limited zones hatched."""
    base = linear_to_srgb(linear)
    shown = base * 0.62
    trusted_color = np.asarray(ZONE_COLORS["trusted"], dtype=np.float32) / 255
    limited_color = np.asarray(ZONE_COLORS["limited"], dtype=np.float32) / 255
    trusted = trust["trusted"] & frame.valid
    limited = trust["limited"] & frame.valid
    shown[trusted] = 0.72 * shown[trusted] + 0.28 * trusted_color
    shown[limited] = 0.66 * shown[limited] + 0.34 * limited_color
    image = uint8_image(shown, frame.valid.astype(np.float32)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for card in cards:
        points = [(x * width - 0.5, y * height - 0.5) for x, y in card["vertices"]]
        draw.line([*points, points[0]], fill=ZONE_COLORS["card"], width=2)
    # Diagonal hatch over calibration-limited cells.
    cell = trust["cell_trusted"]
    grid_count = trust["grid_count"]
    cell_w, cell_h = width / grid_count, height / grid_count
    for row_index in range(grid_count):
        for col_index in range(grid_count):
            if cell[row_index, col_index]:
                continue
            left, top = col_index * cell_w, row_index * cell_h
            draw.line((left, top + cell_h, left + cell_w, top), fill=(255, 239, 214), width=1)
    label_font = font(max(12, min(18, int(width / 48))))
    for assessment, card in zip(record["cards"], cards):
        cx, cy = card["centroid"]
        caption = ("PASS " if assessment["passes_line"] else "LIMITED ") + f"{assessment['worst_neutrality_percent']:g}%"
        draw.text((cx - 28, cy - 8), caption, fill=(28, 38, 34), font=label_font, stroke_width=1, stroke_fill=(245, 242, 226))
    footer_h = 46
    canvas = Image.new("RGB", (width, height + footer_h), (27, 38, 34))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, height + 8), trust_note(record), fill=(238, 236, 220), font=font(13))
    draw.text((12, height + 27), "Green: trusted color zone · Orange hatch: calibration-limited · Outlined: neutral cards",
              fill=(180, 196, 184), font=font(11))
    return canvas.convert("RGBA")
