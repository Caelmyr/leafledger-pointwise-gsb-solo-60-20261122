"""Synthesize a rectified herbarium folio that matches the measurement brief.

Registered tile plane: 620 x 472 px representing 320 mm x 240 mm.

Scene
-----
Target (LEFT) leaf: a serrated blade tilted so its long axis crosses a vertical
mounting tape strip mid-height. Its right margin (one flank + the apex region)
runs under the tape; on the tape's far side a tooth-bearing apex of the SAME
leaf protrudes again.

Same-color fragment: a separate fragment leaf pressed on top on the far side of
the tape, with a narrow neck that almost reaches the tape edge. In ground truth
it is a different object; the neck does not connect to the target.

Another, unrelated leaf sits on the same sheet and must be measured later, so
no global occlusion region is created.

Ground-truth masks are written for auditing the chosen contour; the contour
work itself is driven by the rendered image plus hand-marked zones, not by the
masks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

OUT = Path(__file__).resolve().parents[1] / "fixture"
WIDTH, HEIGHT = 620, 472
MM_PER_PX = (320.0 / WIDTH, 240.0 / HEIGHT)

rng = np.random.default_rng(20261122)
yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)

# ---------------------------------------------------------------- sheet base
sheet_rgb = np.array([0.74, 0.69, 0.575], dtype=np.float32)
grain = rng.normal(0, 0.013, (HEIGHT, WIDTH)).astype(np.float32)
warm = 0.028 * np.sin(xx / 41.0) + 0.02 * (yy / HEIGHT)
base = np.clip(sheet_rgb + (grain + warm)[..., None], 0, 1)
fiber = (np.sin(xx / 2.7 + rng.normal(0, 1, (HEIGHT, 1))) * 0.5 + 0.5) * 0.012
base += fiber[..., None] * np.array([0.6, 0.55, 0.45])


def serrated_leaf_mask(cx, cy, length, half_width, angle_deg, teeth_amp, teeth_n,
                       tip_bias=0.0, seed=0, base_inset=0.0):
    """Boolean mask of a serrated leaf; local u runs base -> tip."""
    local_rng = np.random.default_rng(seed)
    theta = np.deg2rad(angle_deg)
    u = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
    v = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
    along = np.clip(u / length + 0.5, 0.0, 1.0)
    taper = np.sin(np.pi * np.clip(along, 0, 1)) ** 0.72
    phase = along * np.pi * teeth_n + local_rng.uniform(0, 0.4)
    teeth = teeth_amp * (0.72 * np.sin(phase) + 0.28 * np.sin(2.13 * phase + 1.2))
    width = half_width * taper * (1.0 + tip_bias * (along - 0.5))
    width = np.maximum(width, 0.6)
    serrated_half = width * (1.0 + teeth)
    inside = (np.abs(v) <= serrated_half) & (u >= -length / 2) & (u <= length / 2)
    if base_inset:
        inside &= (u + length / 2) > base_inset * (1 - np.clip(np.abs(v) / (half_width + 1e-6), 0, 1))
    return inside, (u, v, along)


def leaf_paint(mask, tone, coords, seed=0):
    rng_local = np.random.default_rng(seed + 7)
    tex = ndimage.gaussian_filter(
        rng_local.normal(0, 0.02, (HEIGHT, WIDTH)).astype(np.float32), 1.1
    )
    img = np.broadcast_to(np.array(tone, dtype=np.float32), (HEIGHT, WIDTH, 3)).copy()
    img += tex[..., None]
    u, v, along = coords
    midrib = np.exp(-(v / 1.6) ** 2) * mask
    img += midrib[..., None] * np.array([0.05, 0.045, 0.02])
    for root in np.linspace(0.18, 0.82, 9):
        center = root + 0.5 * np.abs(v) / (np.ptp(u[mask]) + 1e-6)
        ridge = np.exp(-((u / max(float(np.ptp(u[mask])), 1.0) - center) / 0.004) ** 2) * mask
        img += ridge[..., None] * np.array([0.028, 0.024, 0.012])
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------- target leaf
# Blade axis (32 deg) crosses the vertical tape strip at mid-height while the
# blade still has a wide half-width there, so a real vertical run of the
# margin is pinned. The toothed apex reappears beyond the tape's far edge on
# the SAME axis; the tape hides only the connecting run between them.
TARGET_CX, TARGET_CY = 156.0, 143.0
TARGET_LEN, TARGET_HW = 360.0, 47.0
TARGET_ANGLE = 32.0
target_main, tcoords = serrated_leaf_mask(
    TARGET_CX, TARGET_CY, TARGET_LEN, TARGET_HW, TARGET_ANGLE,
    teeth_amp=0.17, teeth_n=26, base_inset=1.4, seed=11,
)

# Tooth-bearing apex of the SAME leaf: centered on the blade axis, beyond the
# tape's right edge. Geometry is collinear by construction.
TIP_CX, TIP_CY = 300.0, 233.0
tip_full, tip_coords = serrated_leaf_mask(
    TIP_CX, TIP_CY, 60.0, 20.0, TARGET_ANGLE,
    teeth_amp=0.21, teeth_n=9, tip_bias=0.5, base_inset=0.8, seed=12,
)
ta = tip_coords[2]
target_tip = tip_full & (ta > 0.20)

target_leaf = target_main | target_tip

# --------------------------------------------------------------- fragment leaf
# Separate same-color fragment on the tape's far side, clearly off the blade
# axis. Its narrow neck points back toward the hidden margin but stops short
# of the tape edge (a real gap), so it never connects to the target leaf.
FRAG_CX, FRAG_CY = 360, 130
frag_main, fcoords = serrated_leaf_mask(
    FRAG_CX, FRAG_CY, 108.0, 31.0, -26.0,
    teeth_amp=0.16, teeth_n=14, base_inset=1.0, seed=23,
)
neck_full, ncoords = serrated_leaf_mask(
    300, 158, 40.0, 6.0, -26.0, teeth_amp=0.05, teeth_n=6, seed=24,
)
na = ncoords[2]
neck = neck_full & (na > 0.22) & (na < 0.94)
fragment = frag_main | neck

# ----------------------------------------------- second, unrelated leaf (right)
other_leaf, ocoords = serrated_leaf_mask(
    486, 350, 175.0, 48.0, -66.0,
    teeth_amp=0.15, teeth_n=21, base_inset=1.2, seed=31,
)

# ------------------------------------------------------------------ paint blade
TARGET_TONE = (0.36, 0.42, 0.235)
FRAG_TONE = (0.345, 0.405, 0.225)     # deliberately near-identical color
OTHER_TONE = (0.30, 0.37, 0.215)

image = base.copy()
image[target_leaf] = leaf_paint(target_leaf, TARGET_TONE, tcoords, seed=11)[target_leaf]
image[other_leaf] = leaf_paint(other_leaf, OTHER_TONE, ocoords, seed=31)[other_leaf]
image[fragment] = leaf_paint(fragment, FRAG_TONE, fcoords, seed=23)[fragment]

# soft drop shadow where the fragment presses on top
frag_shadow = (
    ndimage.gaussian_filter(fragment.astype(np.float32), 2.4)
    - ndimage.gaussian_filter(fragment.astype(np.float32), 0.8)
)
image = np.clip(image - np.clip(frag_shadow, 0, 1)[..., None] * 0.10, 0, 1)

# ---------------------------------------------------------------- mounting tape
TAPE_X0, TAPE_X1 = 232, 263
TAPE_Y0, TAPE_Y1 = 90, 330
tape = np.zeros((HEIGHT, WIDTH), dtype=bool)
tape[TAPE_Y0:TAPE_Y1, TAPE_X0:TAPE_X1] = True
edge_jitter = ndimage.gaussian_filter(rng.normal(0, 1.0, (HEIGHT, WIDTH)), 2.2)
tape &= ~((np.abs(xx - TAPE_X0) < 2.2) & (edge_jitter > 0.8))
tape &= ~((np.abs(xx - TAPE_X1) < 2.2) & (edge_jitter > 0.9))

tape_tone = np.array([0.80, 0.745, 0.62], dtype=np.float32)
tape_tex = rng.normal(0, 0.012, (HEIGHT, WIDTH)).astype(np.float32)
tape_paint = np.clip(tape_tone + tape_tex[..., None], 0, 1)
blend = 0.82
image[tape] = blend * tape_paint[tape] + (1 - blend) * image[tape]
shadow = (
    ndimage.gaussian_filter(tape.astype(np.float32), 1.6)
    - ndimage.gaussian_filter(tape.astype(np.float32), 0.6)
)
image = np.clip(image - np.clip(shadow, 0, 1)[..., None] * 0.12, 0, 1)

# ------------------------------------------------------------- annotation marks
pil = Image.fromarray(np.rint(image * 255).astype(np.uint8))
clean = pil.copy()  # image evidence without the hand-drawn circles
draw = ImageDraw.Draw(pil, "RGBA")

bar_px = 30.0 / MM_PER_PX[0]
bx0, by0 = 470, 428
draw.rectangle((bx0, by0, bx0 + bar_px, by0 + 7), fill=(38, 34, 28, 255))
draw.text((bx0, by0 - 14), "30 mm", fill=(38, 34, 28, 255))


def dashed_ellipse(box, color, width=2, dash=7):
    for start in range(0, 360, dash * 2):
        draw.arc(box, start=start, end=start + dash, fill=color, width=width)


# user's three hand marks: main body, covered run, fragment-confirmed edge
draw.ellipse((60, 96, 226, 262), outline=(40, 96, 180, 235), width=3)
draw.text((70, 82), "main body", fill=(20, 60, 130, 255))
draw.ellipse((214, 168, 282, 246), outline=(214, 148, 36, 240), width=3)
draw.text((196, 154), "covered run", fill=(150, 96, 12, 255))
dashed_ellipse((312, 84, 414, 178), (200, 52, 48, 240), width=3)
draw.text((322, 68), "fragment edge", fill=(160, 30, 28, 255))
draw.text((288, 250), "toothed tip\n(same leaf)", fill=(60, 80, 30, 255))

pil = pil.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))

# ---------------------------------------------------------------------- outputs
OUT.mkdir(parents=True, exist_ok=True)
pil.save(OUT / "folio_rectified.png")
clean.save(OUT / "folio_clean.png")

truth = {
    "plane": {"width_px": WIDTH, "height_px": HEIGHT, "width_mm": 320, "height_mm": 240},
    "mm_per_px": list(MM_PER_PX),
    "target_leaf_main": {"center": [TARGET_CX, TARGET_CY], "length": TARGET_LEN,
                         "half_width": TARGET_HW, "angle_deg": TARGET_ANGLE},
    "target_tip": {"center": [TIP_CX, TIP_CY]},
    "fragment": {"center": [FRAG_CX, FRAG_CY]},
    "tape": {"x0": TAPE_X0, "x1": TAPE_X1, "y0": TAPE_Y0, "y1": TAPE_Y1},
    "hand_marks": {
        "main_body_circle": [60 / WIDTH, 96 / HEIGHT, 226 / WIDTH, 262 / HEIGHT],
        "covered_run_circle": [214 / WIDTH, 168 / HEIGHT, 282 / WIDTH, 246 / HEIGHT],
        "fragment_edge_ellipse": [312 / WIDTH, 84 / HEIGHT, 414 / WIDTH, 178 / HEIGHT],
    },
}
(OUT / "scene_truth.json").write_text(json.dumps(truth, indent=2))

audit = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
audit[target_main] = (90, 170, 110)
audit[target_tip] = (60, 220, 170)
audit[fragment] = (220, 90, 80)
audit[tape] = (235, 210, 120)
audit[other_leaf] = (120, 140, 220)
Image.fromarray(audit).save(OUT / "ground_truth_masks.png")

np.savez(
    OUT / "masks.npz",
    target_leaf=target_leaf,
    target_main=target_main,
    target_tip=target_tip,
    fragment=fragment,
    tape=tape,
    other_leaf=other_leaf,
)

# ------------------------------------------------------------- sanity numbers
def min_gap(a, b):
    if (a & b).any():
        return 0.0
    dist = ndimage.distance_transform_edt(~a)
    return float(dist[b].min()) if b.any() else float("inf")


print("target px:", int(target_leaf.sum()),
      "| fragment px:", int(fragment.sum()),
      "| tape px:", int(tape.sum()),
      "| other px:", int(other_leaf.sum()))
print("target under tape px:", int((target_leaf & tape).sum()))
print("fragment under tape px:", int((fragment & tape).sum()))
print("tip px right of tape:", int((target_tip & (xx > TAPE_X1)).sum()),
      "| tip px under/left of tape:", int((target_tip & (xx <= TAPE_X1)).sum()))
print("min gap target<->fragment (px):", round(min_gap(target_leaf, fragment), 2))
tape_right_band = tape & (xx >= TAPE_X1 - 1.0)
print("min gap neck<->tape right edge (px):", round(min_gap(neck, tape_right_band), 2))
print("min gap fragment<->tape (px):", round(min_gap(fragment, tape), 2))
print("other leaf overlaps target/fragment:",
      int((other_leaf & (target_leaf | fragment)).sum()))
