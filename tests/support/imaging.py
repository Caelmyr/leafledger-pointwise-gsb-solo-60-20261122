"""Synthetic flat folio imagery for macro / detail-alignment tests.

These helpers render a deterministic, undistorted (front-on) specimen image and
a higher-resolution local crop that shares an exact affine relationship with
it. They are test utilities only and never read uploaded bytes.
"""
import numpy as np
from PIL import Image, ImageDraw


SHEET = (214, 205, 178)
BLADE = (96, 138, 78)
BLADE_DARK = (72, 110, 60)
PAPER_LINE = (150, 138, 110)


def make_plane_image(width=900, height=680, seed=7):
    """A flat herbarium sheet with a blade, discoloration patch, and vein row."""
    rng = np.random.default_rng(seed)
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :] = SHEET
    # Subtle paper grain so registration has texture to lock onto.
    grain = rng.normal(0, 5, size=(height, width, 1))
    array = np.clip(array.astype(np.int16) + grain.astype(np.int16), 0, 255).astype(np.uint8)
    image = Image.fromarray(array, "RGB")
    draw = ImageDraw.Draw(image)
    # Mounted blade body.
    draw.ellipse((120, 250, 760, 430), fill=BLADE, outline=BLADE_DARK, width=4)
    draw.line((110, 340, 770, 340), fill=BLADE_DARK, width=6)  # midrib
    # Lateral veins toward the right, including the "vein tips" to re-check.
    for x in range(220, 720, 60):
        draw.line((x, 340, x + 40, 300), fill=BLADE_DARK, width=3)
        draw.line((x, 340, x + 40, 380), fill=BLADE_DARK, width=3)
    # Rare discoloration patch (priority target).
    draw.ellipse((300, 285, 470, 395), fill=(196, 150, 92), outline=(150, 104, 60), width=3)
    for x, y in ((330, 320), (400, 350), (440, 310)):
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=(120, 74, 44))
    # A ruled paper tag and scale marks outside the blade.
    draw.rectangle((600, 480, 820, 600), outline=PAPER_LINE, width=3)
    for x in range(610, 815, 18):
        draw.line((x, 485, x, 505), fill=PAPER_LINE, width=2)
    # Three distinct, high-contrast fiducial flecks used as matching sites.
    for x, y in ((250, 470), (520, 250), (650, 420)):
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=(40, 44, 48), outline=(240, 240, 235), width=3)
    return image


def affine_matrix(plane_points, capture_points):
    """Affine map plane-normalized -> capture-normalized, as in geometry.affine_fit."""
    source = np.asarray(plane_points, dtype=np.float64)
    target = np.asarray(capture_points, dtype=np.float64)
    design = np.c_[source, np.ones(len(source))]
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    matrix = np.eye(3)
    matrix[:2, :] = coefficients.T
    return matrix


def render_detail(plane_image, matrix, detail_size=(640, 480)):
    """Render a local capture by sampling the flat plane through an affine map.

    The detail is a different, higher-resolution photograph whose normalized
    coordinates relate to the plane via ``matrix`` (plane -> detail).
    """
    plane = np.asarray(plane_image, dtype=np.float32) / 255.0
    width, height = detail_size
    inverse = np.linalg.inv(matrix)
    gy, gx = np.indices((height, width), dtype=np.float32)
    u = (gx + 0.5) / width
    v = (gy + 0.5) / height
    denom = inverse[2, 0] * u + inverse[2, 1] * v + inverse[2, 2]
    pu = (inverse[0, 0] * u + inverse[0, 1] * v + inverse[0, 2]) / denom
    pv = (inverse[1, 0] * u + inverse[1, 1] * v + inverse[1, 2]) / denom
    plane_h, plane_w = plane.shape[:2]
    coords = (pv * plane_h - 0.5, pu * plane_w - 0.5)
    from scipy import ndimage
    channels = [
        ndimage.map_coordinates(plane[:, :, c], coords, order=1, mode="constant", cval=0.0)
        for c in range(3)
    ]
    detail = np.clip(np.stack(channels, axis=-1) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(detail, "RGB")
