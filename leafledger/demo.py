from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy import ndimage

from .geometry import DEFAULT_CORNERS, canvas_dimensions, homography
from .photos import linear_to_srgb, png_bytes, srgb_to_linear
from .store import make_id
from .photos import store_capture


DEMO_KEY = "leafledger.training.v1"
PLANE_SIZE = (620, 472)
PHOTO_SIZE = (766, 598)
DEMO_ROLES = ("diffuse", "parallel", "cross", "rake-left", "rake-right")
POSES = {
    "diffuse": [[0.092, 0.086], [0.911, 0.112], [0.894, 0.912], [0.081, 0.889]],
    "parallel": [[0.122, 0.107], [0.910, 0.084], [0.925, 0.871], [0.101, 0.916]],
    "cross": [[0.079, 0.101], [0.908, 0.119], [0.877, 0.916], [0.081, 0.909]],
    "rake-left": [[0.089, 0.094], [0.894, 0.088], [0.902, 0.905], [0.064, 0.911]],
    "rake-right": [[0.094, 0.095], [0.913, 0.094], [0.926, 0.906], [0.112, 0.893]],
}


@dataclass(frozen=True)
class DemoSpecimen:
    seed: int
    name: str
    body: str
    glaze: str
    firing: str
    batch: str
    colors: tuple
    lamp_strength: float = 0.035


SERIES = (
    DemoSpecimen(81, "Willow-leaved aster", "Asteraceae", "Lanceolate", "March imaging", "Folio 24",
                ((91, 92, 48), (139, 145, 76), (170, 172, 99))),
    DemoSpecimen(114, "Ribbed dock", "Polygonaceae", "Oblong", "March imaging", "Folio 24",
                ((64, 77, 39), (106, 116, 55), (141, 147, 87))),
    DemoSpecimen(129, "Speckled alder", "Betulaceae", "Elliptic", "April imaging", "Folio 31",
                ((77, 82, 47), (116, 118, 70), (140, 142, 83))),
    DemoSpecimen(94, "Red-edged pear", "Rosaceae", "Ovate", "April imaging", "Folio 31",
                ((81, 72, 43), (132, 126, 72), (158, 151, 98))),
    DemoSpecimen(203, "Split-lamp goldenrod", "Asteraceae", "Lanceolate", "Dual-lamp imaging", "Folio 47",
                ((88, 94, 41), (132, 138, 66), (166, 168, 96)), lamp_strength=0.14),
)


def random_field(generator, size, y_space, x_space):
    height, width = size
    rows, cols = int(np.ceil(height / y_space)) + 4, int(np.ceil(width / x_space)) + 4
    coarse = generator.normal(size=(rows, cols)).astype(np.float32)
    y, x = np.indices(size, dtype=np.float32)
    field = ndimage.map_coordinates(coarse, (y / y_space + 1, x / x_space + 1), order=3)
    return field.astype(np.float32)


@dataclass(frozen=True)
class LeafField:
    diffuse: np.ndarray
    height: np.ndarray
    leaf_fraction: np.ndarray
    matte_veins: np.ndarray
    roughness: np.ndarray
    u: np.ndarray
    v: np.ndarray


def surface_palette(colors, values):
    linear_colors = srgb_to_linear(np.array(colors, dtype=np.float32) / 255)
    stops = (-1.2, 0.05, 1.25)
    return np.stack(
        [np.interp(values, stops, linear_colors[:, channel]) for channel in range(3)],
        axis=2,
    ).astype(np.float32)


def neutral_card(diffuse, roughness, x, y, bounds):
    left, right, top, bottom = bounds
    sample = (x >= left) & (x <= right) & (y >= top) & (y <= bottom)
    border = (
        (x >= left - 0.007) & (x <= right + 0.007)
        & (y >= top - 0.009) & (y <= bottom + 0.007)
        & ~sample
    )
    diffuse[sample] = np.array([0.194, 0.172, 0.155])
    diffuse[border] *= 0.70
    roughness[sample] = 0.76


LEFT_CARD = (0.070, 0.165, 0.844, 0.961)
RIGHT_CARD = (0.826, 0.921, 0.844, 0.961)


def blade_outline(x, y, seed):
    axis = .477 + .021 * np.sin((x - .20) * 5) + .023 * (x - .35)
    along = np.clip((x - .16) / .69, 0., 1.)
    taper = np.maximum(np.sin(np.pi * along), 0.0) ** (.65 if seed in (129, 94) else .83)
    teeth = .007 * np.sin(along * np.pi * 37) + .003 * np.sin(along * np.pi * 73)
    width = taper * (.219 if seed in (129, 94) else .179) + taper * teeth
    height = y - axis
    width_upper = width * (.85 + .18 * along)
    width_lower = width * (1.07 - .12 * along)
    limit = np.where(height < 0, width_upper, width_lower)
    blade = np.clip((limit - abs(height)) * 370 + .2, 0, 1)
    blade *= (x > .16) & (x < .85)
    stem = np.exp(-(height / .0038) ** 2) * ((x > .104) & (x < .187))
    return along, height, np.maximum(blade, stem)


def veins(x, height, along, blade):
    main = np.exp(-(height / (.0043 - .0025 * along))**2)
    feeder = np.zeros_like(main)
    fine_grid = np.zeros_like(main)
    for root in np.linspace(.225, .699, 10):
        lateral = np.abs(height)
        center = root + .45 * lateral + 1.18 * lateral ** 2
        ridge = np.exp(-((x - center) / (.0024 + .0012 * (1 - along))) ** 2)
        feeder = np.maximum(feeder, ridge * np.clip(1 - .95 * lateral, 0, 1))
        thin = np.exp(-((x - center + .011 + .048 * lateral) / .0016) ** 2)
        fine_grid = np.maximum(fine_grid, thin * (lateral > .033) * .23)
    return np.clip((.91 * main + .49 * feeder + fine_grid) * blade, 0, 1)


def folio_marks(x, y, rng):
    lower_tag = (x > .70) & (x < .94) & (y > .785) & (y < .943)
    lines = ((np.floor(y * 235).astype(int) % 6) == 0) & lower_tag & (x > .717) & (x < .919)
    scale = (x > .48) & (x < .66) & (y > .886) & (y < .938)
    dark_ticks = (np.floor((x - .48) * 225).astype(int) % 2 == 0) & scale
    folio_number = (x > .763) & (x < .875) & (y > .810) & (y < .818)
    tape_1 = (x > .250) & (x < .276) & (y > .372) & (y < .517)
    tape_2 = (x > .613) & (x < .629) & (y > .475) & (y < .635)
    sheet = np.broadcast_to(np.array([.68, .62, .49], dtype=np.float32), (*x.shape, 3)).copy()
    paper = rng.normal(0, .014, x.shape).astype(np.float32)
    warm = (.027 * np.sin(4*x+3*y) + .042*x) * (1 - y)
    sheet += (paper + warm)[..., None]
    sheet[lower_tag | scale] += [.12, .13, .14]
    sheet[lines | dark_ticks | folio_number] = [.14, .15, .12]
    return sheet, tape_1 | tape_2


def bake_field(specimen):
    width, height = PLANE_SIZE
    shape = (height, width)
    generator = np.random.default_rng(specimen.seed)
    y, x = np.indices(shape, dtype=np.float32)
    y /= height
    x /= width
    broad = random_field(generator, shape, 47, 68)
    mottling = random_field(generator, shape, 11, 19)
    grain = generator.normal(size=shape).astype(np.float32)
    along, off_axis, blade = blade_outline(x, y, specimen.seed)
    vascular = veins(x, off_axis, along, blade)
    sheet, tabs = folio_marks(x, y, generator)
    pigment = surface_palette(specimen.colors, .60 * broad + .25 * mottling + .20 * grain)
    pigment *= (1 - .31 * np.clip((x - .59) * 2.9, 0, 1))[..., None]
    pigment += vascular[..., None] * np.array([.145, .128, .039], dtype=np.float32)
    specks = np.zeros(shape, dtype=np.float32)
    stipple = np.zeros_like(specks)
    for index in range(42):
        px, py = generator.uniform(.22, .80), generator.uniform(.27, .694)
        diameter = generator.uniform(.002, .009)
        distance = ((x - px) / diameter) ** 2 + ((y - py) / (diameter * .79)) ** 2
        if index % 3 == 0:
            stipple = np.maximum(stipple, np.exp(-distance) * blade)
        else:
            specks = np.maximum(specks, np.exp(-distance) * blade)
    pigment += stipple[...,None] * np.array([.172, .157, .083])
    pigment *= (1 - .74 * specks)[..., None]
    crease_center = .543 - .18 * (x - .50) + .008*np.sin(x*46)
    folded = np.exp(-((y-crease_center)/.006)**2) * blade * (x > .39) * (x < .59)
    pigment *= (1 - .28 * folded)[..., None]
    diffuser = sheet * (1 - blade[..., None]) + pigment * blade[..., None]
    diffuser[tabs] = .58 * diffuser[tabs] + np.array([.31, .29, .209])
    roughness = np.clip(.85 + .045 * mottling - .29 * blade - .22 * tabs, .19, .96)
    neutral_card(diffuser, roughness, x, y, LEFT_CARD)
    neutral_card(diffuser, roughness, x, y, RIGHT_CARD)
    relief = blade * (.06 + .033 * mottling) + .29 * vascular + .17 * tabs - .17 * specks
    relief += .22 * folded + .014 * broad
    return LeafField(np.clip(diffuser, 0, 1), relief, blade, vascular, roughness, x, y)


def gloss_reflection(field, role):
    x, y = field.u, field.v
    hotspot = np.exp(-(((x - 0.64 + 0.17 * y) / 0.21) ** 2 + ((y - 0.38) / 0.25) ** 2))
    elongated = np.exp(-(((x - 0.64 - 0.38 * (y - 0.3)) / 0.075) ** 2 + ((y - 0.49) / 0.31) ** 2))
    low = (1 - field.roughness) * field.leaf_fraction * (0.5 + 0.5 * np.sin(8 * x + 13 * y) ** 2)
    intensity = 0.079 * low if role == "diffuse" else 0.34 * low + 0.24 * hotspot + 0.12 * elongated
    if role == "cross":
        intensity = (0.009 + hotspot * 0.018) * low
    return intensity * (1 - field.matte_veins * 0.58)


def lamp_split(field, strength):
    """Cool lamp toward the left, warm lamp toward the right, blended at mid-plane.

    strength scales the per-channel color imbalance. Mild training folios stay
    inside the 5% neutrality line after a global fit; the stress folio does not.
    """
    x = field.u
    side = np.clip(0.5 - x, -0.5, 0.5) * 2.0  # +1 at left edge, -1 at right edge
    red = 1.0 - strength * side
    green = np.ones_like(x)
    blue = 1.0 + strength * side
    return np.stack((red, green, blue), axis=2).astype(np.float32)


def light_field(field, role, lamp_strength=0.035):
    x, y = field.u, field.v
    dy, dx = np.gradient(0.92 * field.height)
    broad = 0.93 + 0.13 * (1 - y) - 0.08 * (x - 0.5) ** 2
    if role in ("rake-left", "rake-right"):
        direction = 1 if role == "rake-left" else -1
        side = 0.91 + direction * 0.13 * (0.5 - x)
        shading = np.clip(0.66 + (dx * direction - dy * 0.23) * 3.9, 0.26, 1.32)
        rgb = field.diffuse * (broad * side * shading)[:, :, None]
    else:
        side = np.clip(0.98 + dx * 0.38 - dy * 0.31, 0.64, 1.2)
        reflection = gloss_reflection(field, role)
        exposure = 0.88 if role == "cross" else 1.09 if role == "parallel" else 1.0
        rgb = field.diffuse * (broad * side * exposure)[:, :, None]
        rgb *= lamp_split(field, lamp_strength)
        rgb += reflection[:, :, None] * np.array([0.69, 0.79, 0.82], dtype=np.float32)
    return np.clip(rgb, 0, 1)


def desk_background(width, height, seed):
    generator = np.random.default_rng(seed + 500)
    noise = random_field(generator, (height, width), 21, 30) * 0.018
    y, x = np.indices((height, width), dtype=np.float32)
    grad = 0.12 + 0.19 * np.exp(-((x / width - 0.24) ** 2 + (y / height - 0.10) ** 2) * 2.4)
    grain = generator.normal(size=(height, width)).astype(np.float32) * 0.008
    neutral = np.maximum(grad + grain + noise, 0.035)
    return neutral[:, :, None] * np.array([0.50, 0.57, 0.58], dtype=np.float32)


def surface_alpha(x, y):
    horizontal = np.abs(x - 0.5) - 0.461
    vertical = np.abs(y - 0.5) - 0.452
    corner = np.hypot(np.maximum(horizontal, 0), np.maximum(vertical, 0))
    nearest = np.minimum(np.maximum(horizontal, vertical), 0)
    sdf = corner + nearest - 0.035
    return np.clip(0.5 - sdf * 410, 0, 1)


def warp_to_photo(array, corners, photo_size, seed):
    width, height = photo_size
    photo_to_tile = np.linalg.inv(homography(DEFAULT_CORNERS, corners))
    photo_y, photo_x = np.indices((height, width), dtype=np.float32)
    photo_y = (photo_y + 0.5) / height
    photo_x = (photo_x + 0.5) / width
    denominator = photo_to_tile[2, 0] * photo_x + photo_to_tile[2, 1] * photo_y + photo_to_tile[2, 2]
    tile_u = (photo_to_tile[0, 0] * photo_x + photo_to_tile[0, 1] * photo_y + photo_to_tile[0, 2]) / denominator
    tile_v = (photo_to_tile[1, 0] * photo_x + photo_to_tile[1, 1] * photo_y + photo_to_tile[1, 2]) / denominator
    tile_h, tile_w = array.shape[:2]
    coords = (tile_v * tile_h - 0.5, tile_u * tile_w - 0.5)
    tile_image = np.stack(
        [ndimage.map_coordinates(array[:, :, component], coords, order=1, mode="nearest") for component in range(3)],
        axis=2,
    )
    alpha = surface_alpha(tile_u, tile_v)
    shadow = ndimage.shift(ndimage.gaussian_filter(alpha, 7), (9, 5), order=1, mode="constant")
    stage = desk_background(width, height, seed)
    stage *= (1 - 0.61 * shadow)[:, :, None]
    rim = np.clip(alpha - ndimage.minimum_filter(alpha, size=8), 0, 1)
    tile_image *= 1 - 0.24 * rim[:, :, None]
    mixed = stage * (1 - alpha[:, :, None]) + tile_image * alpha[:, :, None]
    return linear_to_srgb(mixed)


def render_photo(field, role, specimen):
    linear = light_field(field, role, specimen.lamp_strength)
    image = warp_to_photo(linear, POSES[role], PHOTO_SIZE, specimen.seed)
    array = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array)


def rectangle(name, purpose, left, top, right, bottom):
    return {
        "id": make_id(), "name": name, "purpose": purpose,
        "vertices": [[left, top], [right, top], [right, bottom], [left, bottom]],
    }


def training_regions():
    return [
        rectangle("Mounted leaf and notes", "inspection", 0.06, 0.07, 0.946, 0.818),
        rectangle("Intact blade", "glazed", 0.409, 0.385, 0.458, 0.428),
        rectangle("Mounting sheet", "bare", 0.32, 0.69, 0.55, 0.755),
        rectangle("Left neutral card", "neutral", 0.086, 0.858, 0.15, 0.943),
        rectangle("Right neutral card", "neutral", 0.842, 0.858, 0.906, 0.943),
        rectangle("Near-leaf-tip fading reference", "tip", 0.762, 0.430, 0.838, 0.560),
        rectangle("Image scale", "ignore", 0.478, 0.883, 0.663, 0.941),
    ]


def training_specimen(store, specimen):
    field = bake_field(specimen)
    photos = []
    for role in DEMO_ROLES:
        image = render_photo(field, role, specimen)
        photo = store_capture(
            store.blobs, png_bytes(image), role,
            f"{specimen.name.lower().replace(' ', '-')}-{role}.png",
            "Synthetic image of a pressed leaf, its mount, and a color reference.",
            make_id(),
            synthetic=True,
        )
        photos.append(photo)
    base = photos[0]
    size = canvas_dimensions(base, POSES["diffuse"], 320, 240)
    return {
        "name": specimen.name,
        "batch": specimen.batch,
        "body": specimen.body,
        "glaze": specimen.glaze,
        "firing": specimen.firing,
        "notes": "Synthetic herbarium folio. Plant, mounting tabs and illumination are rendered locally.",
        "width_mm": 320,
        "height_mm": 240,
        "synthetic": True,
        "photos": photos,
        "plane": {"photo_id": base["id"], "corners": POSES["diffuse"], "size": list(size)},
        "regions": training_regions(),
        "registrations": {
            photo["id"]: {"base_photo_id": base["id"], "corners": POSES[photo["role"]], "support_percent": 100}
            for photo in photos[1:]
        },
    }


def load_training(store):
    existing = store.seeded_tiles(DEMO_KEY)
    if existing:
        return existing, False
    rows = [training_specimen(store, specimen) for specimen in SERIES]
    return store.seed_tiles(DEMO_KEY, rows), True


@lru_cache(maxsize=1)
def preview_photo():
    specimen = SERIES[0]
    rendered = render_photo(bake_field(specimen), "diffuse", specimen)
    return png_bytes(rendered)
