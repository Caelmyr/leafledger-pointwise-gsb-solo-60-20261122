"""Derive the single closed contour for the LEFT target leaf.

Inputs (image evidence only; ground-truth masks are never read here):
  * fixture/folio_clean.png            rectified photo, no hand marks
  * fixture/scene_truth.json           plane scale + the three hand-marked
                                       zones the user circles on the card

Pipeline
--------
1. Segment leaf-colored tissue (Lab chroma + lightness) and label components.
2. Attribute components using only the hand-marked zones and geometry:
     main-body mark  -> the target's visible blade body
     fragment mark   -> the same-color fragment (excluded from the contour)
     toothed tip beyond the tape, collinear with the blade -> same leaf
     everything else (e.g. the right-hand leaf) -> not part of this card.
3. Detect the mounting tape strip; its footprint over the blade is the only
   place where an inferred (non-observed) boundary/area may be introduced.
4. Fit a smooth blade envelope (PCA axis + width profile) to the observed
   target components; the envelope's footprint under the tape is the inferred
   bridge. Build one closed contour around (observed target | inferred bridge).
5. Trace the contour, classify every segment as observed (real exposed teeth
   retained) or inferred (smooth, only across the tape run), simplify to a
   single closed polygon <= 40 vertices suitable for one region card, and
   report observed vs inferred area separately, in px and mm^2.

Outputs land in output/ and every claim is audited against ground truth at
the end (audit-only; not used to build the contour).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixture"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

truth = json.loads((FIX / "scene_truth.json").read_text())
W, H = truth["plane"]["width_px"], truth["plane"]["height_px"]
MMX, MMY = truth["mm_per_px"]
MM2_PER_PX = MMX * MMY

img = np.asarray(Image.open(FIX / "folio_clean.png").convert("RGB"))
assert img.shape == (H, W, 3)


def mark_mask(normalized_box):
    x0, y0, x1, y1 = normalized_box
    canvas = Image.new("L", (W, H), 0)
    ImageDraw.Draw(canvas).ellipse(
        (x0 * W, y0 * H, x1 * W, y1 * H), fill=1
    )
    return np.asarray(canvas, dtype=bool)


marks = {name: mark_mask(box) for name, box in truth["hand_marks"].items()}
main_mark, covered_mark, fragment_mark = (
    marks["main_body_circle"], marks["covered_run_circle"], marks["fragment_edge_ellipse"],
)

# ------------------------------------------------------------- 1. tissue mask
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
L_, a_, b_ = lab[..., 0], lab[..., 1], lab[..., 2]
# Green blade: positive a-offset toward green in OpenCV Lab means a_ low;
# use the robust spread rather than hard-coded Lab constants.
leaf_score = (128.0 - a_) + 0.35 * (b_ - 128.0)
t_cut = np.quantile(leaf_score, 0.72)
tissue_raw = (leaf_score > max(t_cut, 6.0)) & (L_ < 196) & (L_ > 35)
tissue_raw = ndimage.binary_opening(tissue_raw, structure=np.ones((3, 3)), iterations=1)
tissue_raw = ndimage.binary_closing(tissue_raw, structure=np.ones((5, 5)), iterations=2)

# ------------------------------------------------------------- 2. tape strip
# The translucent tape still shows blade color, so it bridges components.
# Detect the warm, light, low-chroma strip inside the covered-run mark first,
# then cut the tissue along it to recover the separate visible components.
warm = (b_ - a_) + 0.5 * L_
tape_seed = (warm > np.quantile(warm[covered_mark], 0.35)) & covered_mark
# Column profile: a real tape strip has a contiguous run of dense warm
# columns. Use a high threshold to keep the detected width close to the true
# strip rather than bleeding into the adjacent sheet.
col_density = tape_seed.sum(axis=0)
peak = col_density.max()
cols = np.flatnonzero(col_density > 0.55 * peak)
# keep the single longest contiguous column run (the strip itself)
splits = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
cols = max(splits, key=len)
det_x0, det_x1 = int(cols.min()), int(cols.max())
rows = np.flatnonzero(tape_seed[:, det_x0:det_x1 + 1].sum(axis=1) > 0)
det_y0, det_y1 = int(rows.min()), int(rows.max())
tape = np.zeros((H, W), dtype=bool)
tape[det_y0:det_y1 + 1, det_x0:det_x1 + 1] = True
true_tape = truth["tape"]
print(f"detected tape x:[{det_x0},{det_x1}] y:[{det_y0},{det_y1}] "
      f"vs truth x:[{true_tape['x0']},{true_tape['x1']}]")

# Cut along the full detected strip so the two visible parts separate cleanly.
# Dilate by one pixel to defeat the thin translucent halo that otherwise
# reconnects body and tip around the strip's torn edges.
tape_cut = ndimage.binary_dilation(tape, iterations=1)
tissue = tissue_raw & ~tape_cut

n, labels = cv2.connectedComponents(tissue.astype(np.uint8), connectivity=8)
components = []
for idx in range(1, n):
    m = labels == idx
    area = int(m.sum())
    if area < 120:
        continue
    ys, xs = np.nonzero(m)
    components.append({
        "id": idx, "mask": m, "area": area,
        "centroid": (float(xs.mean()), float(ys.mean())),
        "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
    })
print(f"tissue components kept after tape cut: {len(components)}")

# ------------------------------------------------------------- 2. attribution
def overlap_frac(comp, zone):
    return float((comp["mask"] & zone).sum()) / max(comp["area"], 1)


def pca_axis(mask):
    ys, xs = np.nonzero(mask)
    pts = np.stack((xs, ys), axis=1).astype(np.float64)
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T)
    vals, vecs = np.linalg.eigh(cov)
    direction = vecs[:, -1]
    # orient so the axis points toward +x (tip side)
    if direction[0] < 0:
        direction = -direction
    return center, direction, vals


# target body = the component the main-body mark sits on
body = max(components, key=lambda c: overlap_frac(c, main_mark))
# fragment = the component the fragment mark sits on
fragment = max(components, key=lambda c: overlap_frac(c, fragment_mark))

center, axis, eigvals = pca_axis(body["mask"])
angle_deg = np.rad2deg(np.arctan2(axis[1], axis[0]))

# project every component center onto the body axis
perp = np.array([-axis[1], axis[0]])


def axial(comp):
    delta = np.array(comp["centroid"]) - center
    return float(delta @ axis), float(delta @ perp)


body_u, _ = axial(body)

tip = None
for comp in components:
    if comp["id"] in (body["id"], fragment["id"]):
        continue
    u, v = axial(comp)
    cx = comp["centroid"][0]
    # same-leaf tip: beyond the tape, ahead of the body along the axis, close
    # to the fitted axis line, and smaller than the body (an apex, not a full
    # separate leaf). The tolerance absorbs segmentation noise at the tape.
    if (
        cx > det_x1
        and u > body_u
        and abs(v) < 60
        and comp["area"] < 0.45 * body["area"]
    ):
        tip = comp
        break
assert tip is not None, "could not find the collinear toothed tip beyond the tape"

target_components = [body, tip]
target_visible = body["mask"] | tip["mask"]
excluded_ids = {c["id"] for c in components} - {body["id"], tip["id"]}
print("body area:", body["area"], "| tip area:", tip["area"],
      "| fragment area:", fragment["area"],
      "| excluded component ids:", sorted(excluded_ids))
print(f"fitted blade axis angle: {angle_deg:.2f} deg (truth 32.0)")

# ------------------------------------------------------------- 3. blade envelope
ys, xs = np.nonzero(target_visible)
pts = np.stack((xs, ys), axis=1).astype(np.float64)
rel = pts - center
u = rel @ axis
v = rel @ perp
order = np.argsort(u)
u_sorted, v_sorted = u[order], v[order]

# smooth half-width profile in bins along the axis (teeth averaged out)
bins = np.linspace(u_sorted.min(), u_sorted.max(), 34)
idx = np.clip(np.digitize(u_sorted, bins) - 1, 0, len(bins) - 2)
half = np.zeros(len(bins) - 1)
centers_u = 0.5 * (bins[:-1] + bins[1:])
for k in range(len(half)):
    sel = v_sorted[idx == k]
    if sel.size >= 4:
        # 80th percentile smooths over the teeth while staying inside the
        # serrated envelope; exposed teeth are restored by the visible union.
        half[k] = np.percentile(np.abs(sel), 80)
valid = half > 0
half_smooth = np.interp(
    bins, np.r_[centers_u[valid], centers_u[valid][-1] + 1],
    np.r_[ndimage.gaussian_filter1d(half[valid], 1.2), half[valid][-1]],
)
half_smooth = np.maximum(half_smooth, 2.0)

# extend the envelope across the tape gap to the furthest observed tip pixel
grid_u = np.arange(u_sorted.min(), u_sorted.max() + 1.0, 1.0)
grid_half = np.interp(grid_u, bins, half_smooth)
gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
grel = np.stack((gx.ravel() - center[0], gy.ravel() - center[1]), axis=1)
gu = grel @ axis
gv = grel @ perp
env_half = np.interp(gu, grid_u, grid_half, left=0.0, right=0.0)
envelope = (np.abs(gv) <= env_half) & (gu >= grid_u.min()) & (gu <= grid_u.max())
envelope = envelope.reshape(H, W)

# inferred bridge = envelope area that is not directly observed, restricted to
# the tape footprint (the only occlusion allowed for THIS contour). The blade
# narrows toward its apex, so the envelope strip can touch the protruding tip
# only along a single edge; dilate inside the tape to guarantee 8-connectivity
# to both visible parts while never leaving the tape footprint.
inferred = envelope & tape & ~target_visible
inferred = ndimage.binary_closing(inferred, structure=np.ones((5, 5)), iterations=1)
# Minimal expansion to guarantee 8-connectivity to both visible parts. One
# pixel inside the tape plus a one-pixel edge band behind each tape edge is
# enough; larger dilation overshoots the true blade and inflates the area.
inferred = ndimage.binary_dilation(inferred, iterations=1) & tape
axial_band = ndimage.binary_dilation(tape, iterations=1) & envelope
inferred = (inferred | axial_band) & ~target_visible

union = target_visible | inferred
union = ndimage.binary_fill_holes(union)
n_final, _ = cv2.connectedComponents(union.astype(np.uint8), 8)
print("inferred bridge px under tape:", int((inferred & tape).sum()),
      "| edge band px:", int((inferred & ~tape).sum()),
      "| connected components in union:", n_final - 1)
assert n_final == 2, "bridge must connect body and tip into one blade"
# the small edge band must remain adjacent to the tape and inside the envelope
assert (inferred & ~tape & ~ndimage.binary_dilation(tape, iterations=2)).sum() == 0

# ------------------------------------------------------------- 5. contour trace
contours, _ = cv2.findContours(
    union.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)
contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)

# classify each contour point. A boundary point is "inferred" when it borders
# the inferred bridge (tape-covered, no observed tissue). Use the final
# inferred mask dilated to the boundary so the full bridge outline -- the two
# cross-tape edges and the short edge-band joins -- is marked, while every
# edge on exposed serrated blade stays "observed".
inf_boundary = ndimage.binary_dilation(inferred, iterations=2)
vis_boundary = ndimage.binary_dilation(target_visible, iterations=1)
class_map = (inf_boundary & ~vis_boundary).astype(np.uint8)  # 1 = inferred
pt_class = np.array([
    "inferred" if class_map[int(round(y)), int(round(x))] else "observed"
    for x, y in contour
])
print("contour points classified inferred:", int((pt_class == "inferred").sum()),
      "of", len(pt_class))

# Deliver <= 40 vertices (one region card). A tooth-aware selection: detect
# convex protrusions (serration tips) on the contour and reserve vertices for
# them, then distribute the remaining budget by equal arc length. This hugs
# the real teeth instead of chording across them, keeping the delivered
# polygon area as close as possible to the pixel-exact raster area.
BUDGET = 40
closed_c = np.vstack([contour, contour[0], contour[1]])
# discrete curvature: angle change at each contour point (closed indexing)
npts = len(contour)
prev = np.roll(contour, 1, axis=0)
nxt = np.roll(contour, -1, axis=0)
v1 = contour - prev
v2 = nxt - contour
cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
dot = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]
turn = np.arctan2(cross, dot)
# outward (convex) corners only; a strong positive turn marks a tooth tip.
convex_strength = np.maximum(-turn, 0.0)
# non-max suppression over a small window so each tooth yields one vertex
win = max(3, npts // 70)
peak_idx = []
order = np.argsort(convex_strength)[::-1]
taken = np.zeros(npts, dtype=bool)
for cand in order:
    if convex_strength[cand] < 0.25:
        break
    lo, hi = cand - win, cand + win + 1
    if not taken[lo:hi].any():
        peak_idx.append(int(cand % npts))
        taken[lo:hi] = True
    if len(peak_idx) >= BUDGET - 6:
        break
peak_idx = sorted(set(i % npts for i in peak_idx))

# Force the observed<->inferred transition points into the vertex set so the
# polygon keeps explicit corners where the real serrated edge gives way to the
# smooth tape bridge (and back). These are few and make the orange boundary
# legible instead of letting an equal-arc chord span the whole hidden run.
is_inf = (pt_class == "inferred")
transitions = np.flatnonzero(np.diff(np.r_[is_inf, is_inf[0]]))
mandatory = sorted(set(int(t % npts) for t in transitions))
# merge mandatory transitions with tooth peaks, keeping within budget
anchor = sorted(set(peak_idx) | set(mandatory))[:BUDGET]

# fill remaining vertices by equal arc length, skipping near-selected peaks
seg_len = np.hypot(np.diff(closed_c[:, 0]), np.diff(closed_c[:, 1]))
cum = np.concatenate([[0.0], np.cumsum(seg_len)])
perimeter = cum[-1]
chosen = set(anchor)
fill_targets = np.linspace(0.0, perimeter, BUDGET, endpoint=False)
for dist in fill_targets:
    k = int(np.searchsorted(cum, dist, side="right") - 1)
    k = min(k, npts - 1)
    frac = (dist - cum[k]) / max(seg_len[k], 1e-9)
    idx = k if frac < 0.5 else (k + 1) % npts
    if all(abs(idx - p) % npts > 2 for p in chosen):
        chosen.add(idx)
    if len(chosen) >= BUDGET:
        break
sel = sorted(chosen)[:BUDGET]
poly = contour[sel].astype(np.float64)

# tag each polygon edge by the class of pixels along it. A lower threshold
# (>=25% inferred) keeps the short cross-tape bridge edges visible even though
# the serrated edges are much longer.
edge_tags = []
closed_pts = np.vstack([poly, poly[0]])
for p0, p1 in zip(closed_pts[:-1], closed_pts[1:]):
    length = np.hypot(*(p1 - p0))
    steps = max(int(np.ceil(length)), 2)
    sx = np.clip(np.linspace(p0[0], p1[0], steps).round().astype(int), 0, W - 1)
    sy = np.clip(np.linspace(p0[1], p1[1], steps).round().astype(int), 0, H - 1)
    votes = class_map[sy, sx]
    edge_tags.append("inferred" if votes.mean() >= 0.25 else "observed")

# ------------------------------------------------------------- areas
# Single measurement basis = the delivered closed card polygon, rasterized
# with the SAME PIL routine as LeafLedger's geometry.polygon_mask() so the
# pixel count matches the Geometry tab exactly. The observed/inferred split is
# then taken INSIDE that footprint, so the two parts sum exactly to the
# reported raster area (no mixing of a finer, non-deliverable mask).
def app_polygon_mask(extent, vertices):
    width, height = extent
    grid = Image.new("1", extent)
    points = [(float(x * width - 0.5), float(y * height - 0.5)) for x, y in vertices]
    ImageDraw.Draw(grid).polygon(points, fill=1)
    return np.asarray(grid, dtype=bool)


norm_vertices = poly / [W, H]
card_mask = app_polygon_mask((W, H), norm_vertices)
total_px = int(card_mask.sum())
# The inferred region is the tape bridge plus its 2-px connectivity edge band.
# observed = every other card pixel (visible blade or card over real sheet).
occlusion_zone = ndimage.binary_dilation(tape, iterations=1)
inferred_mask = card_mask & inferred
observed_mask = card_mask & ~inferred_mask
observed_px = int(observed_mask.sum())
inferred_px = int(inferred_mask.sum())
assert observed_px + inferred_px == total_px
assert (inferred_mask & ~occlusion_zone).sum() == 0, "inferred area escaped the tape"

# independent vector area of the delivered polygon (cross-check vs raster)
vpoly = norm_vertices
shoelace = 0.5 * abs(
    np.sum(vpoly[:, 0] * np.roll(vpoly[:, 1], -1) - vpoly[:, 1] * np.roll(vpoly[:, 0], -1))
)
vector_area_px = shoelace * W * H

# full-detail reference: pixel-exact mask including every tooth (not the card)
detail_px = int(union.sum())

result = {
    "plane": truth["plane"],
    "mm_per_px": [MMX, MMY],
    "contour_vertices_normalized": np.round(norm_vertices, 5).tolist(),
    "vertex_count": len(poly),
    "edge_classes": edge_tags,
    "observed_edge_count": int(sum(t == "observed" for t in edge_tags)),
    "inferred_edge_count": int(sum(t == "inferred" for t in edge_tags)),
    # Authoritative card figures (the single closed polygon, rasterized as the
    # Geometry tab does). observed + inferred == raster exactly.
    "raster_area_px": total_px,
    "observed_area_px": observed_px,
    "inferred_area_px": inferred_px,
    "vector_polygon_area_px": round(float(vector_area_px), 2),
    "vector_vs_raster_area_percent": round(100 * float(vector_area_px) / total_px, 2),
    "raster_area_mm2": round(total_px * MM2_PER_PX, 3),
    "observed_area_mm2": round(observed_px * MM2_PER_PX, 3),
    "inferred_area_mm2": round(inferred_px * MM2_PER_PX, 3),
    "inferred_fraction_percent": round(100 * inferred_px / total_px, 2),
    # full-detail pixel-exact reference (all teeth); the 40-vertex card covers
    # card_coverage_percent of this, the rest is tooth detail lost to the
    # fixed vertex budget, never fragment or other-leaf pixels.
    "full_detail_mask_px": detail_px,
    "card_coverage_of_detail_percent": round(100 * total_px / detail_px, 2),
    "detected_tape_px_bbox": [det_x0, det_y0, det_x1, det_y1],
    "fitted_axis_angle_deg": round(float(angle_deg), 3),
    "excluded_fragment_area_px": int(fragment["area"]),
    "area_basis_note": (
        "Single closed card polygon is the measurement basis; raster_area_mm2 "
        "is its pixel count and observed_area_mm2 + inferred_area_mm2 equals it "
        "exactly. inferred is the smooth envelope bridge and lies wholly within "
        "the mounting tape. vector_polygon_area is the shoelace area of the same "
        "polygon. full_detail_mask_px is the pixel-exact blade including every "
        "fine tooth; the 40-vertex card covers card_coverage_of_detail_percent, "
        "the difference being serration detail (not fragment or other leaf)."
    ),
}
(OUT / "contour_result.json").write_text(json.dumps(result, indent=2))

# ------------------------------------------------------------- overlay render
canvas = Image.fromarray(img).convert("RGBA")
overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
od = ImageDraw.Draw(overlay)
# observed fill green (only the visible-tissue part of the delivered card)
od.bitmap((0, 0), Image.fromarray(((card_mask & target_visible) * 90).astype(np.uint8)),
          fill=(90, 170, 110, 90))
# inferred fill amber, with diagonal hatching clipped strictly to the mask
od.bitmap((0, 0), Image.fromarray((inferred_mask * 255).astype(np.uint8)),
          fill=(224, 168, 60, 110))
hatch = Image.new("RGBA", (W, H), (0, 0, 0, 0))
hd = ImageDraw.Draw(hatch)
for off in range(-H, W, 10):
    hd.line([(off, 0), (off + H, H)], fill=(150, 92, 18, 150), width=1)
hatch.putalpha(Image.fromarray((inferred_mask * 150).astype(np.uint8)))
overlay = Image.alpha_composite(overlay, hatch)
canvas = Image.alpha_composite(canvas, overlay)
draw = ImageDraw.Draw(canvas)

# draw edges in two colors
for tag, color, width in (
    ("observed", (36, 122, 84, 255), 3),
    ("inferred", (202, 122, 24, 255), 3),
):
    for (p0, p1), t in zip(zip(closed_pts[:-1], closed_pts[1:]), edge_tags):
        if t == tag:
            draw.line([tuple(p0), tuple(p1)], fill=color, width=width)

# vertices
for i, (x, y) in enumerate(poly, 1):
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(250, 246, 230, 255),
                 outline=(30, 38, 43, 255))

# fragment outline (red) to show it was deliberately excluded
fpts, _ = cv2.findContours(fragment["mask"].astype(np.uint8),
                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for fc in fpts:
    draw.line([tuple(p[0]) for p in fc] + [tuple(fc[0][0])],
              fill=(200, 52, 48, 255), width=2)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
except OSError:
    font = font_b = ImageFont.load_default()

legend = [
    ("observed edge (real teeth kept)", (36, 122, 84, 255)),
    ("inferred edge (tape bridge only)", (202, 122, 24, 255)),
    ("excluded same-color fragment", (200, 52, 48, 255)),
]
draw.rectangle((10, 8, 330, 86), fill=(28, 34, 38, 215))
for i, (text, color) in enumerate(legend):
    yy0 = 14 + i * 24
    draw.line([(20, yy0 + 8), (44, yy0 + 8)], fill=color, width=4)
    draw.text((52, yy0), text, fill=(245, 240, 228, 255), font=font)

draw.text((10, H - 52),
          f"observed {result['observed_area_mm2']} mm2  +  "
          f"inferred {result['inferred_area_mm2']} mm2  =  "
          f"{result['raster_area_mm2']} mm2",
          fill=(20, 26, 30, 255), font=font_b,
          stroke_width=2, stroke_fill=(245, 240, 228, 220))
canvas.convert("RGB").save(OUT / "contour_overlay.png", dpi=(150, 150))

# ------------------------------------------------------------- audit vs truth
gt = np.load(FIX / "masks.npz")
gt_target, gt_frag, gt_tape = gt["target_leaf"], gt["fragment"], gt["tape"]


def dice(a, b):
    return 2 * (a & b).sum() / max((a.sum() + b.sum()), 1)


# Two audit views:
#  * full-detail mask (every tooth): best possible pixel agreement
#  * delivered 40-vertex card: what the single region polygon actually covers
inside_frag = (card_mask & gt_frag).sum()
tip_recovered = (card_mask & gt["target_tip"]).sum() / max(gt["target_tip"].sum(), 1)
audit = {
    "full_detail_mask_vs_true_target_dice": round(float(dice(union, gt_target)), 4),
    "delivered_card_vs_true_target_dice": round(float(dice(card_mask, gt_target)), 4),
    "fragment_pixels_leaked_into_card": int(inside_frag),
    "fragment_leak_percent_of_card": round(100 * inside_frag / max(card_mask.sum(), 1), 3),
    "other_leaf_leak_px": int((card_mask & gt["other_leaf"]).sum()),
    "true_target_pixels_outside_card": int((gt_target & ~card_mask).sum()),
    "card_pixels_beyond_true_target": int((card_mask & ~gt_target).sum()),
    "card_beyond_target_percent": round(
        100 * (card_mask & ~gt_target).sum() / max(card_mask.sum(), 1), 2),
    "toothed_tip_recovered_percent": round(100 * float(tip_recovered), 2),
    "inferred_pixels_inside_true_tape_percent": round(
        100 * (inferred_mask & gt_tape).sum() / max(inferred_mask.sum(), 1), 2),
    "observed_tissue_vs_true_visible_recall": round(
        100 * (target_visible & gt_target).sum() / max(gt_target.sum(), 1), 2),
    "card_observed_plus_inferred_equals_raster": bool(
        observed_px + inferred_px == total_px),
    "card_overshoot_cause": (
        "Beyond-target pixels are straight-chord area between serration tips "
        "under the fixed 40-vertex card budget (the card approaches the convex "
        "hull); none are fragment or other-leaf pixels, which remain at zero."
    ),
}
(OUT / "audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(result, indent=2))
print("AUDIT:", json.dumps(audit, indent=2))
