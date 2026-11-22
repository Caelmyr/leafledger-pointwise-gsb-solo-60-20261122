import numpy as np

from .geometry import polygon_mask
from .model import InputError
from .photos import linear_to_srgb, luminance


RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)
XYZ_TO_RGB = np.linalg.inv(RGB_TO_XYZ).astype(np.float32)
WHITE_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
LAB_BREAK = 6.0 / 29.0
MIN_CARD_SAMPLES = 40
MIN_CARD_LEVEL = 0.012
MAX_CARD_CHROMA_SPREAD = 0.22


def linear_lab(rgb):
    xyz = np.asarray(rgb, dtype=np.float32) @ RGB_TO_XYZ.T
    values = xyz / WHITE_D65
    response = np.where(
        values > LAB_BREAK**3,
        np.cbrt(values),
        values / (3 * LAB_BREAK**2) + 4.0 / 29.0,
    )
    light = 116 * response[..., 1] - 16
    green_red = 500 * (response[..., 0] - response[..., 1])
    blue_yellow = 200 * (response[..., 1] - response[..., 2])
    return np.stack((light, green_red, blue_yellow), axis=-1)


def lab_linear(lab):
    color = np.asarray(lab, dtype=np.float32)
    midpoint = (color[..., 0] + 16) / 116
    axes = np.stack(
        (
            midpoint + color[..., 1] / 500,
            midpoint,
            midpoint - color[..., 2] / 200,
        ),
        axis=-1,
    )
    xyz = np.where(axes > LAB_BREAK, axes**3, 3 * LAB_BREAK**2 * (axes - 4 / 29)) * WHITE_D65
    return xyz @ XYZ_TO_RGB.T


def delta_e(first, second):
    return np.linalg.norm(first - second, axis=-1)


def swatch(rgb):
    clipped = np.rint(np.clip(linear_to_srgb(np.asarray(rgb)), 0, 1) * 255).astype(np.uint8)
    return "#" + "".join(f"{int(item):02x}" for item in clipped)


def quantile_rows(values, levels=(0.05, 0.25, 0.50, 0.75, 0.95), precision=3):
    numbers = np.quantile(values, levels)
    return {str(round(level * 100)): round(float(number), precision) for level, number in zip(levels, numbers)}


def float_list(values, places=4):
    return np.round(np.asarray(values, dtype=np.float64), places).tolist()


def robust_neutral(frame, neutral_region, candidate_area, target):
    measured_area = polygon_mask(frame.extent, neutral_region["vertices"]) & candidate_area
    native_count = int(np.count_nonzero(measured_area & frame.valid))
    support = measured_area & frame.valid & ~frame.clipped
    count = int(np.count_nonzero(support))
    if count < MIN_CARD_SAMPLES:
        raise InputError("The neutral card needs at least 40 supported, unclipped pixels.")
    if count / max(native_count, 1) < 0.7:
        raise InputError("The neutral card is heavily clipped. Choose a different photo or card region.")
    colors = frame.linear[support]
    card = np.median(colors, axis=0)
    if np.min(card) < MIN_CARD_LEVEL:
        raise InputError("The calibration card is too dark for stable channel correction.")
    levels = luminance(colors)
    floor, ceiling = np.quantile(levels, (0.10, 0.90))
    mid = float(luminance(card))
    if (ceiling - floor) / max(mid, 1e-8) > 0.38:
        raise InputError("Card brightness varies too much. Select a single consistently lit patch.")
    chroma = colors / np.maximum(levels[:, None], 0.001)
    spread = np.quantile(chroma, 0.9, axis=0) - np.quantile(chroma, 0.10, axis=0)
    if np.max(spread) > MAX_CARD_CHROMA_SPREAD:
        raise InputError("Card chroma varies too much. Keep the entire selection inside the neutral patch.")
    gains = np.full(3, target, dtype=np.float32) / card
    if np.max(gains) > 4 or np.min(gains) < 0.22:
        raise InputError("Card exposure would require channel gains outside the supported range.")
    return {
        "target": target,
        "gains": float_list(gains, 7),
        "photo_id": frame.photo["id"],
        "region_id": neutral_region["id"],
        "card_samples": count,
        "card_median_linear_rgb": float_list(card, 7),
        "before_swatch": swatch(card),
        "after_swatch": swatch(card * gains),
        "card_lightness_range": float_list((floor, mid, ceiling), 6),
    }


def calibration_for(tile, frame):
    record = tile.get("calibration")
    if not record or frame.photo["id"] != record["photo_id"]:
        return None
    if record.get("schema") == "leafledger.balance.v2":
        if record["plane_size"] != list(frame.extent) or record["plane_corners"] != tile["plane"]["corners"]:
            return None
        region_lookup = {region["id"]: region["vertices"] for region in tile["regions"]}
        if any(region_id not in region_lookup for region_id in record["card_region_ids"]):
            return None
        return record
    if record["plane_size"] != list(frame.extent) or record["plane_corners"] != tile["plane"]["corners"]:
        return None
    selected = next(
        (region for region in tile["regions"] if region["id"] == record["region_id"]),
        None,
    )
    if not selected or selected["vertices"] != record["region_vertices"]:
        return None
    return record


def corrected_linear(tile, frame, enabled=True):
    record = calibration_for(tile, frame) if enabled else None
    if record is None:
        return frame.linear, None
    gains = np.array(record["gains"], dtype=np.float32)
    return frame.linear * gains, record


def cell_summaries(lab, support, count, center):
    height, width = support.shape
    rows = np.rint(np.linspace(0, height, count + 1)).astype(int)
    cols = np.rint(np.linspace(0, width, count + 1)).astype(int)
    cells = []
    for y, (top, bottom) in enumerate(zip(rows, rows[1:])):
        for x, (left, right) in enumerate(zip(cols, cols[1:])):
            active = support[top:bottom, left:right]
            size = int(np.count_nonzero(active))
            if size < 12 or size < active.size * 0.15:
                continue
            points = lab[top:bottom, left:right][active]
            local_center = np.median(points, axis=0)
            local_delta = delta_e(points, local_center)
            cells.append({
                "cell": [x, y],
                "bounds": [left / width, top / height, right / width, bottom / height],
                "pixels": size,
                "lab": float_list(local_center),
                "delta_from_center": round(float(delta_e(local_center, center)), 3),
                "within_p90": round(float(np.quantile(local_delta, 0.90)), 3),
                "swatch": swatch(lab_linear(local_center)),
            })
    return cells


def dominant_swatches(lab, support, *, count=5):
    samples = lab[support]
    stride = max(1, len(samples) // 9000)
    values = samples[::stride]
    ranks = np.argsort(values[:, 0])
    sample_positions = np.rint(np.linspace(0.1, 0.9, count) * (len(values) - 1)).astype(int)
    centers = values[ranks[sample_positions]].copy()
    assignment = np.zeros(len(values), dtype=np.int32)
    for _ in range(14):
        distances = np.sum((values[:, None] - centers[None, :]) ** 2, axis=-1)
        updated = np.argmin(distances, axis=1)
        for index in range(count):
            selected = values[updated == index]
            if len(selected):
                centers[index] = np.mean(selected, axis=0)
        if np.array_equal(assignment, updated):
            break
        assignment = updated
    memberships = np.bincount(assignment, minlength=count)
    result = []
    for index in np.argsort(-memberships, kind="stable"):
        if memberships[index] < len(values) * 0.02:
            continue
        result.append({
            "coverage_percent": round(100 * int(memberships[index]) / len(values), 2),
            "lab": float_list(centers[index]),
            "swatch": swatch(lab_linear(centers[index])),
        })
    return result


def summarize_colors(linear, support, grid_count, trust=None, calibration=None):
    from .balance import apply_trust, trust_metrics
    lab_image = linear_lab(linear)
    trusted_support, limited_support = apply_trust(support, trust)
    color_support = trusted_support if trust is not None else support
    if int(np.count_nonzero(color_support)) < 40:
        raise InputError("Too few trusted color pixels remain after the calibration-zone partition.")
    colors = lab_image[color_support]
    center = np.median(colors, axis=0)
    distances = delta_e(lab_image, center)
    representative = swatch(lab_linear(center))
    quantiles = quantile_rows(distances[color_support], (0.25, 0.50, 0.90, 0.95, 0.99))
    summary = {
        "swatch": representative,
        "median_lab": float_list(center),
        "lightness": quantile_rows(colors[:, 0]),
        "chroma": quantile_rows(np.linalg.norm(colors[:, 1:], axis=1)),
        "delta_e_from_tile_median": quantiles,
        "cells": cell_summaries(lab_image, color_support, grid_count, center),
        "palette": dominant_swatches(lab_image, color_support),
        "pixels": int(np.count_nonzero(color_support)),
        "coverage_percent": round(100 * np.mean(support), 2),
    }
    if trust is not None:
        summary["color_zone_support"] = trust_metrics(support, trusted_support, limited_support, calibration)
    return summary, distances


def reference_fingerprint(linear, support):
    lab = linear_lab(linear[support])
    lightness = lab[:, 0]
    chroma = np.linalg.norm(lab[:, 1:], axis=1)
    center = np.median(lab, axis=0)
    distance = delta_e(lab, center)
    return {
        "median_lab": float_list(center, 5),
        "swatch": swatch(lab_linear(center)),
        "lightness_10_50_90": float_list(np.quantile(lightness, (0.10, 0.5, 0.90))),
        "chroma_10_50_90": float_list(np.quantile(chroma, (0.10, 0.5, 0.90))),
        "dispersion": round(float(np.quantile(distance, 0.9)), 5),
        "supported_pixels": len(lab),
    }


def trusted_fingerprint(linear, support, trust):
    """Fingerprint computed only on pixels whose color calibration is trusted."""
    from .balance import apply_trust, require_trusted
    trusted, limited = apply_trust(support, trust)
    require_trusted(trusted, "The reference region")
    fingerprint = reference_fingerprint(linear, trusted)
    fingerprint["trusted_pixels"] = int(np.count_nonzero(trusted))
    fingerprint["calibration_limited_pixels"] = int(np.count_nonzero(limited))
    return fingerprint


def compare_reference(fingerprint, reference, this_basis, reference_basis, this_trust=None, reference_trust=None):
    observed = np.asarray(fingerprint["median_lab"])
    approved = np.asarray(reference["median_lab"])
    delta = observed - approved
    light_delta = np.asarray(fingerprint["lightness_10_50_90"]) - np.asarray(reference["lightness_10_50_90"])
    chroma_delta = np.asarray(fingerprint["chroma_10_50_90"]) - np.asarray(reference["chroma_10_50_90"])
    limitation = matching_basis(this_basis, reference_basis)
    zone = zone_comparison(this_trust, reference_trust)
    return {
        "specimen": fingerprint,
        "reference": reference,
        "delta_e76": round(float(delta_e(observed, approved)), 3),
        "delta_lab": float_list(delta),
        "lightness_quantile_delta": float_list(light_delta),
        "chroma_quantile_delta": float_list(chroma_delta),
        "dispersion_delta": round(fingerprint["dispersion"] - reference["dispersion"], 3),
        "capture_note": limitation,
        "zone_note": zone["note"],
        "same_trusted_zones": zone["same_zones"],
        "trusted_zone": zone["current"],
        "reference_zone": zone["approved"],
    }


def zone_comparison(this_trust, reference_trust):
    current = this_trust or None
    approved = reference_trust or None
    if current is None and approved is None:
        return {
            "same_zones": True,
            "note": "Neither side records a calibration-zone partition; the full requested region was compared.",
            "current": None,
            "approved": None,
        }
    if current is None or approved is None:
        return {
            "same_zones": False,
            "note": "Only one side of this match carries calibration trust zones. Re-run the side without zones on the same calibrated sample.",
            "current": current,
            "approved": approved,
        }
    same = (
        current["card_region_ids"] == approved["card_region_ids"]
        and current.get("tip_region_id") == approved.get("tip_region_id")
        and current.get("grid_count") == approved.get("grid_count")
        and current["photo_id"] == approved["photo_id"]
    )
    note = (
        "This match reuses the same card zones and protected near-tip reference as the approved sample."
        if same
        else "The calibration trust zones differ from the approved sample. Re-check on the same card and tip regions before interpreting color distance."
    )
    return {"same_zones": same, "note": note, "current": current, "approved": approved}


def matching_basis(first, second):
    if first["synthetic"] != second["synthetic"]:
        return "This comparison mixes synthetic and photographed material."
    if first["photo_role"] != second["photo_role"]:
        return "The illumination labels differ. Interpret these color distances accordingly."
    a, b = first["neutral_target"], second["neutral_target"]
    if a != b:
        return "The pair uses different neutral-card targets or one result uses uncorrected color."
    if a is None:
        return "Both values use the photo color assumption without neutral-card adjustment."
    return "Both were measured with the same neutral target and recorded illumination label."


def reference_map(linear, reference):
    values = linear_lab(linear)
    median = np.array(reference["median_lab"], dtype=np.float32)
    return delta_e(values, median)


def capture_basis(tile, frame, correction):
    return {
        "photo_role": frame.photo["role"],
        "synthetic": bool(frame.photo.get("synthetic")),
        "neutral_target": correction["target"] if correction else None,
        "source_space": frame.photo["working_color_space"],
        "tile_id": tile["id"],
    }
