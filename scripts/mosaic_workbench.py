"""Two-capture folio mosaic for one mounted herbarium leaf.

Builds two registered (same folio plane, pixel-for-pixel aligned) synthetic
captures of one leaf:

* capture A ("left fill", cross-polarized style light) keeps fine veins crisp
  but the small marginal tear is almost invisible;
* capture B ("right fill", raking light) reveals the marginal tear through
  relief shading while its fine veins are faint.

The two coverage domains overlap only in a narrow central band, and a small
bottom wedge is covered by neither capture.

A full-frame reference mosaic is assembled with a hard cut only: every output
pixel is copied wholesale from one source, no blending or averaging anywhere,
and the whole shared band is assigned to a single capture (A).

Run from the repository root:

    python3 -I -m scripts.mosaic_workbench --out /workspace/output/folio_mosaic_2026-11-22
"""

from __future__ import annotations

import argparse
import csv
import json
import site
import sys
from pathlib import Path

# Under `python3 -I` neither PYTHONPATH nor user site-packages are consulted.
# The host keeps its trusted wheels in this explicit, non-user-writable tree.
for _trusted in ("/opt/pointwise-env/site-packages", str(Path(__file__).resolve().parents[1])):
    site.addsitedir(_trusted)

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from leafledger import demo
from leafledger.photos import linear_to_srgb, png_bytes, srgb_to_linear

# Registered folio plane: 2x the demo plane so veins/tear stay crisp.
SCALE = 2
PLANE_W, PLANE_H = demo.PLANE_SIZE[0] * SCALE, demo.PLANE_SIZE[1] * SCALE

# Coverage geometry in normalized folio coordinates (x: 0 left -> 1 right).
A_EDGE_X0 = 0.585          # left capture reaches this far on its straight side
B_EDGE_X0 = 0.385          # right capture reaches this far on its straight side
BAND_X0, BAND_X1 = B_EDGE_X0, A_EDGE_X0     # shared narrow band (0.20 wide)
SEAM_X0 = A_EDGE_X0                          # whole band assigned to A; cut here
VOID_TIP_Y = 0.905        # bottom wedge covered by neither capture
VOID_CORNER_X = 0.600

# Marginal tear location (normalized coords). It sits just beyond A's
# coverage edge, in B-only territory, so the mosaic keeps the raking view of it.
TEAR_X = 0.630
TEAR_Y_TOP, TEAR_Y_BOTTOM = 0.345, 0.415

SPECIMEN = demo.SERIES[0]  # Willow-leaved aster


# --------------------------------------------------------------------------- #
# Field construction
# --------------------------------------------------------------------------- #
def plane_grid():
    yy, xx = np.indices((PLANE_H, PLANE_W), dtype=np.float32)
    return xx / PLANE_W, yy / PLANE_H


def build_field():
    """Bake the demo leaf at 2x resolution and add a marginal tear."""
    field = demo.bake_field(SPECIMEN)
    diffuse = ndimage.zoom(field.diffuse, (SCALE, SCALE, 1), order=1)
    height = ndimage.zoom(field.height, SCALE, order=1)
    blade = ndimage.zoom(field.leaf_fraction, SCALE, order=1)
    vascular = ndimage.zoom(field.matte_veins, SCALE, order=1)
    rough = ndimage.zoom(field.roughness, SCALE, order=1)
    x, y = plane_grid()

    # ---- marginal tear: a curved slit running inward from the upper edge ---
    slit_x = TEAR_X + 0.030 * (y - TEAR_Y_TOP) / (TEAR_Y_BOTTOM - TEAR_Y_TOP)
    slit_x += 0.004 * np.sin((y - TEAR_Y_TOP) * 90.0)
    along_tear = (y >= TEAR_Y_TOP) & (y <= TEAR_Y_BOTTOM)
    gap_width = 0.0034 + 0.0016 * np.sin(
        (y - TEAR_Y_TOP) / (TEAR_Y_BOTTOM - TEAR_Y_TOP) * np.pi
    )
    gap = along_tear & (np.abs(x - slit_x) < gap_width)
    # raised lips immediately beside the slit -> raking-light signature
    lip_l = np.exp(-((x - (slit_x - gap_width - 0.0045)) / 0.0022) ** 2) * along_tear
    lip_r = np.exp(-((x - (slit_x + gap_width + 0.0045)) / 0.0022) ** 2) * along_tear
    lip = np.maximum(lip_l, lip_r)
    inside_blade = blade > 0.35
    tear_gap = gap & inside_blade

    # gap reads as dark mounting sheet showing through, not leaf tissue
    sheet_tone = np.broadcast_to(
        np.array([0.205, 0.185, 0.140], dtype=np.float32), diffuse.shape
    ).copy()
    diffuse = np.where(tear_gap[..., None], sheet_tone, diffuse)
    blade = np.where(tear_gap, 0.0, blade)
    # lips carry only a subtle color change; their main signal is relief
    diffuse *= (1 - 0.08 * lip[..., None])
    height = height + 0.085 * lip * blade - 0.030 * tear_gap
    vascular = np.where(tear_gap, 0.0, vascular)

    return demo.LeafField(
        np.clip(diffuse, 0, 1),
        height.astype(np.float32),
        blade.astype(np.float32),
        vascular.astype(np.float32),
        np.clip(rough, 0, 1),
        x,
        y,
    ), tear_gap, lip


def vein_adjusted(field, role, x):
    """Return a field copy whose vein strength fits the capture's lighting."""
    main = np.exp(-(field.v / (0.0043 - 0.0025 * 0.5)) ** 2)  # midrib component
    fine_only = np.maximum(field.matte_veins - 0.91 * main, 0.0)
    if role == "cross":
        # cross-polarized, diffuse: lift fine venation a little
        gain = np.clip(1.0 + 0.55 * fine_only, 1, 1.5)
        diffuse = field.diffuse * gain[..., None]
        vascular = field.matte_veins
    else:
        # raking capture: fine veins washed out, only midrib survives
        main = np.exp(-(field.v / (0.0043 - 0.0025 * 0.5)) ** 2)  # midrib component
        fine_only = np.maximum(field.matte_veins - 0.91 * main, 0.0)
        keep = main + 0.18 * fine_only
        factor = np.ones_like(field.matte_veins)
        factor = np.where(
            field.leaf_fraction > 0.35,
            1.0 - 0.10 * np.maximum(field.matte_veins - keep, 0.0),
            factor,
        )
        diffuse = field.diffuse * factor[..., None]
        vascular = keep * np.clip(field.leaf_fraction, 0, 1)
    return demo.LeafField(
        np.clip(diffuse, 0, 1), field.height, field.leaf_fraction,
        np.clip(vascular, 0, 1), field.roughness, field.u, field.v,
    )


def render_capture(field, role, preserve_relief=None):
    """Light the folio in the registered plane (identity homography)."""
    adjusted = vein_adjusted(field, role, field.u)
    if role.startswith("rake"):
        # raking light would turn fine vein ridges into bright relief lines;
        # flatten the vein-scale relief so the raking capture reads veins faintly
        vein_soft = ndimage.gaussian_filter(adjusted.height, (2.0, 2.0))
        damping = np.clip(adjusted.matte_veins, 0, 1)
        if preserve_relief is not None:
            damping *= np.clip(1.0 - preserve_relief, 0, 1)
        height = adjusted.height - 0.62 * damping * (adjusted.height - vein_soft)
        # also smooth vein-scale contrast in the lit diffuse of this capture,
        # without touching the tear lips (they carry no vascular component)
        diffuse = adjusted.diffuse
        diffuse_blur = ndimage.gaussian_filter(diffuse, (1.35, 1.35, 0))
        vein_weight = np.clip(adjusted.matte_veins * np.clip(adjusted.leaf_fraction, 0, 1), 0, 0.8)
        diffuse = diffuse * (1 - 0.55 * vein_weight[..., None]) + diffuse_blur * (
            0.55 * vein_weight[..., None]
        )
        adjusted = demo.LeafField(
            np.clip(diffuse, 0, 1), height.astype(np.float32), adjusted.leaf_fraction,
            adjusted.matte_veins, adjusted.roughness, adjusted.u, adjusted.v,
        )
    rgb = demo.light_field(adjusted, role)
    rgb = np.clip(rgb, 0, 1)
    if role == "cross":
        # even, soft, low-gloss exposure; blur so the tear slit stays faint
        rgb_lin = ndimage.gaussian_filter(rgb, sigma=(1.15, 1.15, 0))
        rgb = 0.96 * rgb_lin + 0.04 * rgb
    else:
        # gentle local relief contrast only (avoid re-introducing vein texture)
        blur = ndimage.gaussian_filter(rgb, (7, 7, 0))
        rgb = np.clip(blur + (rgb - blur) * 1.06, 0, 1)
        # raking setup is slightly soft-focused: fine venation fades while the
        # larger-scale tear lips keep their relief contrast
        rgb = ndimage.gaussian_filter(rgb, (0.7, 0.7, 0))
    # independent camera grain for each capture
    rng = np.random.default_rng(901 if role == "cross" else 902)
    grain_sigma = 0.0045 if role == "cross" else 0.0032
    rgb = np.clip(rgb + rng.normal(0, grain_sigma, rgb.shape).astype(np.float32), 0, 1)
    return rgb  # linear RGB in folio-plane coordinates


# --------------------------------------------------------------------------- #
# Coverage masks
# --------------------------------------------------------------------------- #
def coverage_masks():
    x, y = plane_grid()
    wiggle_a = 0.006 * np.sin(y * 22.0) + 0.0035 * np.sin(y * 57.0 + 1.3)
    wiggle_b = 0.007 * np.sin(y * 19.0 + 0.7) + 0.003 * np.sin(y * 61.0)
    a_edge = A_EDGE_X0 + wiggle_a
    b_edge = B_EDGE_X0 + wiggle_b
    mask_a = x <= a_edge
    mask_b = x >= b_edge
    # bottom wedge: neither frame reached this corner of the mount
    void_line = VOID_CORNER_X + (VOID_TIP_Y - y) * 1.35
    void = (y >= VOID_TIP_Y) & (x >= void_line)
    mask_a = mask_a & ~void
    mask_b = mask_b & ~void
    # actual shared band = true intersection of the two coverage domains
    overlap = mask_a & mask_b
    return mask_a, mask_b, void, overlap


def stripe_fill(shape):
    """Neutral diagonal-stripe plate for uncovered (void) pixels."""
    yy, xx = np.indices(shape[:2], dtype=np.float32)
    stripe = ((xx + yy) % 16 <= 5)
    plate = np.zeros(shape, dtype=np.float32)
    plate[stripe] = (0.82, 0.74, 0.30)
    plate[~stripe] = (0.20, 0.21, 0.22)
    return plate


# --------------------------------------------------------------------------- #
# Mosaic assembly (hard cut, no blending)
# --------------------------------------------------------------------------- #
def save_plane_png(path, rgb_linear, alpha=None):
    srgb = linear_to_srgb(rgb_linear)
    arr = np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8)
    if alpha is None:
        image = Image.fromarray(arr, "RGB")
    else:
        a = np.rint(np.clip(alpha, 0, 1) * 255).astype(np.uint8)
        image = Image.fromarray(np.dstack([arr, a]), "RGBA")
    path.write_bytes(png_bytes(image))
    return image


def assemble_mosaic(rgb_a, rgb_b, mask_a, mask_b, void, overlap):
    seam_col = int(round(SEAM_X0 * PLANE_W))
    choose_a = mask_a.copy()
    # anything the left frame does not cover, and the right frame does, comes
    # from B; the shared band itself stays entirely on A (hard cut at seam)
    choose_b = mask_b & ~choose_a
    assert not np.any(choose_a & choose_b), "overlap assignment leaked"

    plate = stripe_fill(rgb_a.shape)
    mosaic = plate.copy()
    mosaic[choose_a] = rgb_a[choose_a]
    mosaic[choose_b] = rgb_b[choose_b]
    source = np.full(rgb_a.shape[:2], "V", dtype="<U1")
    source[choose_a] = "A"
    source[choose_b] = "B"
    # void stays plate; check coverage consistency
    covered = mask_a | mask_b
    assert np.array_equal(void, ~covered), "void must equal the uncovered set"
    # hard museum rule: every pixel covered by BOTH originals comes from A only
    assert np.all(source[overlap] == "A"), "shared band contains a non-A pixel"
    return mosaic, source, seam_col, choose_a, choose_b


# --------------------------------------------------------------------------- #
# Line scans with provenance
# --------------------------------------------------------------------------- #
def sample_line(source, rgb_a, rgb_b, start, end, stations, band_px):
    """Station table along one line; void stations carry empty (null) values.

    Pixel values are read from the ORIGINAL capture (not the mosaic) so each
    number can be traced back to its source file and source pixel location.
    """
    sx, sy = start
    ex, ey = end
    frac = np.linspace(0, 1, stations)
    px = (sx + (ex - sx) * frac) * PLANE_W - 0.5
    py = (sy + (ey - sy) * frac) * PLANE_H - 0.5
    length = float(np.hypot((ex - sx) * PLANE_W, (ey - sy) * PLANE_H))
    dxn, dyn = (ex - sx), (ey - sy)
    norm = max(np.hypot(dxn, dyn), 1e-12)
    perp_x, perp_y = -dyn / norm, dxn / norm
    offsets = np.linspace(-band_px / 2, band_px / 2, max(int(round(band_px)), 1))
    rows, runs = [], []
    run_label, run_start = None, 0
    for i in range(stations):
        ox = px[i] + perp_x * offsets
        oy = py[i] + perp_y * offsets
        cxi = np.rint(np.clip(ox, 0, PLANE_W - 1)).astype(int)
        cyi = np.rint(np.clip(oy, 0, PLANE_H - 1)).astype(int)
        inside = (ox >= -0.5) & (ox <= PLANE_W - 0.5) & (oy >= -0.5) & (oy <= PLANE_H - 0.5)
        labels = source[cyi, cxi]
        labels = labels[inside]
        use_cxi, use_cyi = cxi[inside], cyi[inside]
        n = len(labels)
        n_a = int(np.count_nonzero(labels == "A"))
        n_b = int(np.count_nonzero(labels == "B"))
        n_v = int(np.count_nonzero(labels == "V"))
        if n == 0 or n_v == n:
            label, origin, values = "VOID", None, None
        elif n_a and not n_b:
            label, origin = "A", "A_vein_left.png"
            pick = labels == "A"
            values = rgb_a[use_cyi[pick], use_cxi[pick]]
        elif n_b and not n_a:
            label, origin = "B", "B_tear_right.png"
            pick = labels == "B"
            values = rgb_b[use_cyi[pick], use_cxi[pick]]
        else:
            # station straddles the hard cut: report each side separately
            label = "MIXED"
            origin = "A_vein_left.png + B_tear_right.png"
            values = np.vstack([
                rgb_a[use_cyi[labels == "A"], use_cxi[labels == "A"]],
                rgb_b[use_cyi[labels == "B"], use_cxi[labels == "B"]],
            ])
        if values is None:
            r = g = b = luma = None
        else:
            mean_lin = values.mean(axis=0)
            r8, g8, b8 = np.rint(np.clip(linear_to_srgb(mean_lin), 0, 1) * 255).astype(int)
            r, g, b = int(r8), int(g8), int(b8)
            luma = round(float(np.sum(
                mean_lin * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            )), 5)
        rows.append({
            "station": i + 1,
            "distance_px": round(i * length / max(stations - 1, 1), 3),
            "folio_x": round(float(sx + (ex - sx) * frac[i]), 6),
            "folio_y": round(float(sy + (ey - sy) * frac[i]), 6),
            "center_col_px": round(float(px[i] + 0.5), 2),
            "center_row_px": round(float(py[i] + 0.5), 2),
            "band_samples": int(n),
            "samples_from_A": n_a,
            "samples_from_B": n_b,
            "samples_uncovered": n_v,
            "source": label,
            "source_file": origin,
            "source_pixel_col_px": (round(float(px[i] + 0.5), 2) if origin else None),
            "source_pixel_row_px": (round(float(py[i] + 0.5), 2) if origin else None),
            "R_0_255": r,
            "G_0_255": g,
            "B_0_255": b,
            "linear_luminance": luma,
        })
        if label != run_label:
            if run_label is not None:
                runs.append((run_label, run_start, i - 1))
            run_label, run_start = label, i
    runs.append((run_label, run_start, stations - 1))
    segment_runs = [
        {
            "source": label,
            "source_file": (
                None if label == "VOID"
                else "A_vein_left.png" if label in ("A", "MIXED")
                else "B_tear_right.png"
            ),
            "station_start": s + 1, "station_end": e + 1,
            "folio_from": [round(float(sx + (ex - sx) * s / max(stations - 1, 1)), 6),
                           round(float(sy + (ey - sy) * s / max(stations - 1, 1)), 6)],
            "folio_to": [round(float(sx + (ex - sx) * e / max(stations - 1, 1)), 6),
                         round(float(sy + (ey - sy) * e / max(stations - 1, 1)), 6)],
            "note": "uncovered by both originals - left empty" if label == "VOID"
                    else ("straddles the hard cut" if label == "MIXED" else ""),
        }
        for label, s, e in runs
    ]
    return rows, segment_runs


# --------------------------------------------------------------------------- #
# Annotation artwork (English labels; no CJK font on this host)
# --------------------------------------------------------------------------- #
def load_font(size):
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_reference(mosaic_srgb, source, seam_col, overlap, tear_gap, out_path):
    base = Image.fromarray(np.rint(np.clip(mosaic_srgb, 0, 1) * 255).astype(np.uint8), "RGB")
    image = base.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    f_big, f_mid, f_small = load_font(18), load_font(14), load_font(11)
    w, h = image.size

    # actual shared band (true intersection of the two coverage domains)
    band_edge = ndimage.binary_dilation(overlap, iterations=1) & ~overlap
    band_rgb = np.array(image)
    band_rgb[band_edge] = (90, 200, 245)
    image = Image.fromarray(band_rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    # hard cut
    draw.line((seam_col, 0, seam_col, h), fill=(255, 70, 60, 255), width=2)

    # tear location: box around the slit
    ty, tx = np.nonzero(tear_gap)
    bx0, bx1 = tx.min() - 10, tx.max() + 10
    by0, by1 = ty.min() - 10, ty.max() + 10
    draw.rectangle((bx0, by0, bx1, by1), outline=(255, 170, 40, 255), width=2)
    # clean zoom crop taken before any annotation is drawn
    zoom = base.crop((bx0 - 34, by0 - 34, bx1 + 34, by1 + 34)).resize(
        (250, 250), Image.Resampling.NEAREST
    )

    # sacrificed fine-vein zone: blade area sourced from B beyond the seam
    mask_b_only = source == "B"
    rows, cols = np.nonzero(mask_b_only)
    draw.rectangle((cols.min(), rows.min(), cols.max(), rows.max()),
                   outline=(200, 120, 230, 255), width=1)

    def tag(xy, text, fill, font):
        tw = int(draw.textlength(text, font=font)) + 8
        x, y = xy
        draw.rectangle((x, y, x + tw, y + font.size + 6), fill=(20, 26, 30, 215))
        draw.text((x + 4, y + 3), text, font=font, fill=fill)

    tag((8, 8), "A: left fill, cross-polarized (fine veins kept)", (150, 225, 170), f_mid)
    tag((8, 32), "B: right fill, raking light (tear kept; fine veins faint)", (255, 200, 120), f_mid)
    tag((int(BAND_X0 * w) + 4, h - 96),
        "actual shared band: A only, no mixing", (120, 205, 245), f_small)
    tag((seam_col + 6, h - 34), "hard cut (film change)", (255, 100, 95), f_small)
    tag((bx0 - 8, by1 + 10), "tear: kept from B raking view; faint in A", (255, 190, 90), f_small)
    tag((cols.min() + 6, rows.max() - 52),
        "SACRIFICED: fine veins here survive only as faint B texture", (225, 160, 245), f_small)

    # void wedge
    void = source == "V"
    vy, vx = np.nonzero(void)
    if len(vx):
        tag((vx.min() + 220, vy.min() + 30), "VOID: neither original covers this",
            (240, 225, 120), f_small)

    # zoomed inset of the tear (clean crop)
    image.paste(zoom, (w - 262, 12))
    draw.rectangle((w - 262, 12, w - 12, 262), outline=(255, 170, 40, 255), width=2)
    draw.text((w - 258, 268), "tear zoom (B side past cut)", font=f_small, fill=(255, 190, 90))
    image.save(out_path)


def draw_scan_overlay(mosaic_srgb, lines, out_path, title):
    image = Image.fromarray(np.rint(np.clip(mosaic_srgb, 0, 1) * 255).astype(np.uint8), "RGB").convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(13)
    colors = {"A": (90, 200, 130, 235), "B": (255, 170, 70, 235),
              "VOID": (235, 80, 80, 235), "MIXED": (255, 255, 255, 235)}
    w, h = image.size
    for line_index, (start, end, rows, runs) in enumerate(lines):
        p0 = (start[0] * w, start[1] * h)
        p1 = (end[0] * w, end[1] * h)
        draw.line((p0, p1), fill=(20, 20, 20, 200), width=7)
        n_stations = len(rows)
        for seg in runs:
            label, s, e = seg["source"], seg["station_start"] - 1, seg["station_end"] - 1
            f0 = np.array(p0) + (np.array(p1) - np.array(p0)) * s / max(n_stations - 1, 1)
            f1 = np.array(p0) + (np.array(p1) - np.array(p0)) * e / max(n_stations - 1, 1)
            draw.line((tuple(f0), tuple(f1)), fill=colors.get(label, colors["MIXED"]), width=4)
        for tag_index, pt in enumerate((p0, p1)):
            draw.ellipse((pt[0] - 5, pt[1] - 5, pt[0] + 5, pt[1] + 5),
                         fill=(255, 255, 255, 255), outline=(20, 20, 20, 255), width=2)
            draw.text((pt[0] + 7, pt[1] - 8), f"L{line_index+1}{'AB'[tag_index]}",
                      font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    combined = Image.alpha_composite(image, overlay).convert("RGB")
    head = Image.new("RGB", (combined.width, 34), (24, 31, 34))
    hd = ImageDraw.Draw(head)
    hd.text((8, 7), title, font=load_font(16), fill=(240, 240, 230))
    final = Image.new("RGB", (combined.width, combined.height + 34), (24, 31, 34))
    final.paste(head, (0, 0))
    final.paste(combined, (0, 34))
    final.save(out_path)


# --------------------------------------------------------------------------- #
# A/B side-by-side evidence plate
# --------------------------------------------------------------------------- #
def comparison_plate(rgb_a, rgb_b, tear_gap, out_path):
    srgb_a = linear_to_srgb(rgb_a)
    srgb_b = linear_to_srgb(rgb_b)
    arr_a = np.rint(np.clip(srgb_a, 0, 1) * 255).astype(np.uint8)
    arr_b = np.rint(np.clip(srgb_b, 0, 1) * 255).astype(np.uint8)

    ty, tx = np.nonzero(tear_gap)
    tear_box = (tx.min() - 60, ty.min() - 60, tx.max() + 60, ty.max() + 60)
    # a blade patch well inside both frames, on the left half where A dominates
    vein_box = (int(0.20 * PLANE_W), int(0.36 * PLANE_H),
                int(0.36 * PLANE_W), int(0.58 * PLANE_H))
    cell_w, cell_h = 460, 300
    plate = Image.new("RGB", (cell_w * 2 + 36, cell_h * 2 + 128), (24, 31, 34))
    draw = ImageDraw.Draw(plate)
    font, small = load_font(16), load_font(13)
    draw.text((12, 10), "Same registered folio coordinates in both originals",
              font=font, fill=(240, 240, 230))

    def paste_crop(arr, box, origin, heading):
        crop = Image.fromarray(arr).crop(box).resize((cell_w, cell_h), Image.Resampling.NEAREST)
        plate.paste(crop, origin)
        draw.rectangle((origin[0], origin[1], origin[0] + cell_w, origin[1] + cell_h),
                       outline=(200, 210, 205), width=1)
        draw.text((origin[0] + 6, origin[1] + cell_h + 6), heading,
                  font=small, fill=(235, 230, 210))

    x_l, x_r = 12, cell_w + 24
    row1_top, row2_top = 44, 44 + cell_h + 34
    paste_crop(arr_a, tear_box, (x_l, row1_top), "A  tear region: slit faint / blurred")
    paste_crop(arr_b, tear_box, (x_r, row1_top), "B  tear region: slit dark, lips shaded")
    paste_crop(arr_a, vein_box, (x_l, row2_top), "A  vein patch: fine network crisp")
    paste_crop(arr_b, vein_box, (x_r, row2_top), "B  vein patch: network soft, direction lost")
    plate.save(out_path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    field, tear_gap, lip = build_field()
    rgb_a = render_capture(field, "cross")
    rgb_b = render_capture(field, "rake-right", preserve_relief=lip)
    mask_a, mask_b, void, overlap = coverage_masks()

    # save the two registered originals (RGB + coverage alpha)
    save_plane_png(out / "A_vein_left.png", rgb_a, mask_a.astype(np.float32))
    save_plane_png(out / "B_tear_right.png", rgb_b, mask_b.astype(np.float32))

    mosaic, source, seam_col, choose_a, choose_b = assemble_mosaic(
        rgb_a, rgb_b, mask_a, mask_b, void, overlap
    )
    mosaic_srgb = linear_to_srgb(mosaic)
    save_plane_png(out / "reference_full_frame.png", mosaic)

    # masks / provenance rasters
    Image.fromarray((mask_a * 255).astype(np.uint8)).save(out / "mask_A_coverage.png")
    Image.fromarray((mask_b * 255).astype(np.uint8)).save(out / "mask_B_coverage.png")
    Image.fromarray((void * 255).astype(np.uint8)).save(out / "mask_void_neither.png")
    code = np.zeros(source.shape, np.uint8)
    code[source == "A"] = 1
    code[source == "B"] = 2
    Image.fromarray(code).save(out / "source_label_map.png")  # 0=void 1=A 2=B

    annotate_reference(mosaic_srgb, source, seam_col, overlap,
                       tear_gap, out / "reference_annotated.png")
    comparison_plate(rgb_a, rgb_b, tear_gap, out / "AB_same_region_comparison.png")

    # ---- line scans ------------------------------------------------------- #
    line_specs = [
        # main scan: crosses the shared band, the hard cut and the tear slit
        ("scan1_across_seam_and_tear", (0.10, 0.382), (0.92, 0.382), 120, 5),
        # scan through the bottom wedge: demonstrates an empty (null) segment
        ("scan2_through_void_wedge", (0.42, 0.955), (0.82, 0.955), 90, 5),
        # scan fully inside the shared band: proves the band is single-source
        ("scan3_inside_shared_band", (BAND_X0 + 0.01, 0.62), (BAND_X1 - 0.01, 0.62), 60, 3),
    ]
    all_lines, manifest_rows = [], []
    for name, start, end, stations, band in line_specs:
        rows, runs = sample_line(source, rgb_a, rgb_b, start, end, stations, band)
        all_lines.append((start, end, rows, runs))
        with (out / f"{name}.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        (out / f"{name}_segments.json").write_text(json.dumps(runs, indent=2))
        for r in rows:
            manifest_rows.append({"line": name, **r})

    with (out / "line_scan_provenance.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    draw_scan_overlay(
        mosaic_srgb, all_lines, out / "line_scan_overlay.png",
        "Line scans: green=A (vein capture)  orange=B (tear capture)  red=void (left empty)",
    )

    # ---- machine-readable manifest --------------------------------------- #
    manifest = {
        "folio_plane_px": [PLANE_W, PLANE_H],
        "specimen": SPECIMEN.name,
        "captures": {
            "A_vein_left.png": {
                "role": "left fill, cross-polarized diffuse",
                "strength": "fine veins crisp; marginal tear faint",
            },
            "B_tear_right.png": {
                "role": "right fill, raking light",
                "strength": "tear relief clear; fine veins faint",
            },
        },
        "shared_band_nominal_normalized": {"x0": BAND_X0, "x1": BAND_X1},
        "shared_band_actual": "intersection of mask_A_coverage and mask_B_coverage",
        "hard_cut_seam_x_normalized": SEAM_X0,
        "hard_cut_seam_col_px": seam_col,
        "band_assignment": (
            "every pixel covered by both originals is copied from A; verified by "
            "assertion that the label map has no B pixel inside the overlap mask"
        ),
        "blending": "none; every mosaic pixel copied from exactly one source",
        "void_wedge_normalized": {"tip_y": VOID_TIP_Y, "corner_x": VOID_CORNER_X},
        "tear_normalized": {"x": TEAR_X, "y_top": TEAR_Y_TOP, "y_bottom": TEAR_Y_BOTTOM},
        "sacrifice": (
            "The tear was placed past A's coverage edge on purpose: the mosaic keeps "
            "the B raking view of it. What is sacrificed is the fine venation of the "
            "B-sourced blade beyond the hard cut - its vein-scale relief was flattened "
            "in that capture, so only faint vein texture remains there; no A pixels "
            "exist in that region to recover the crisp network."
        ),
        "line_scans": [
            {"name": n, "start": s, "end": e, "stations": st, "band_px": b}
            for n, s, e, st, b in line_specs
        ],
        "source_label_map_codes": {"0": "void", "1": "A", "2": "B"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "out": str(out),
        "plane": [PLANE_W, PLANE_H],
        "A_px": int(np.count_nonzero(mask_a)),
        "B_px": int(np.count_nonzero(mask_b)),
        "void_px": int(np.count_nonzero(void)),
        "seam_col": seam_col,
        "mosaic_A": int(np.count_nonzero(source == "A")),
        "mosaic_B": int(np.count_nonzero(source == "B")),
        "mosaic_void": int(np.count_nonzero(source == "V")),
        "tear_gap_px": int(np.count_nonzero(tear_gap)),
    }, indent=2))


if __name__ == "__main__":
    main()
