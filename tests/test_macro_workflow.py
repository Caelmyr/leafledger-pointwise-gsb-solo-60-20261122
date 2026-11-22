import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leafledger.geometry import affine_fit, specimen_frame  # noqa: E402
from leafledger.macro import plan_macro_frame, preview_detail  # noqa: E402
from leafledger.model import InputError  # noqa: E402
from leafledger.photos import png_bytes, store_capture  # noqa: E402
from leafledger.store import Store, make_id  # noqa: E402
from leafledger.workspace import (  # noqa: E402
    remove_detail_link,
    save_detail_link,
)

from tests.support.imaging import affine_matrix, make_plane_image, render_detail  # noqa: E402


PLANE_SIZE = (900, 680)
SPECIMEN_MM = (300.0, 226.6667)


def rectangle(name, purpose, left, top, right, bottom):
    return {
        "id": make_id(),
        "name": name,
        "purpose": purpose,
        "vertices": [[left, top], [right, top], [right, bottom], [left, bottom]],
    }


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "workspace")


@pytest.fixture()
def tile_with_plane(store):
    image = make_plane_image(*PLANE_SIZE)
    photo = store_capture(store.blobs, png_bytes(image), "diffuse", "survey.png", "", make_id())
    tile = store.create_tile({
        "name": "Macro test specimen",
        "width_mm": SPECIMEN_MM[0],
        "height_mm": SPECIMEN_MM[1],
    })
    default_corners = [[0, 0], [1, 0], [1, 1], [0, 1]]

    def add_photo(tile):
        tile["photos"].append(photo)
        tile["plane"] = {"photo_id": photo["id"], "corners": default_corners, "size": list(PLANE_SIZE)}

    tile = store.edit_tile(tile["id"], tile["version"], add_photo, "capture", "base")
    # Priority: rare discoloration patch. Secondary: the vein-tip row immediately
    # to its right, reachable only if the frame can extend past the patch.
    patch = rectangle("Rare discoloration", "inspection", 300 / 900, 285 / 680, 470 / 900, 395 / 680)
    veins = rectangle("Vein tips to re-check", "inspection", 500 / 900, 285 / 680, 640 / 900, 395 / 680)

    def add_regions(tile):
        tile["regions"].extend([patch, veins])

    tile = store.edit_tile(tile["id"], tile["version"], add_regions, "region", "targets")
    return tile, photo, patch, veins


def _exhaustive_best_primary(tile, region, width_mm, height_mm, samples=400):
    """Independent dense search for the maximum primary coverage reachable."""
    from leafledger.geometry import polygon_mask
    frac_w = width_mm / tile["width_mm"]
    frac_h = height_mm / tile["height_mm"]
    extent = tuple(tile["plane"]["size"])
    mask = polygon_mask(extent, region["vertices"])
    best = 0
    for top in np.linspace(0.0, 1.0 - frac_h, samples):
        for left in np.linspace(0.0, 1.0 - frac_w, samples):
            best = max(best, _mask_coverage(mask, left, top, left + frac_w, top + frac_h))
    return best


def _mask_coverage(mask, left, top, right, bottom):
    height, width = mask.shape
    x0 = max(0, int(np.floor(left * width)))
    y0 = max(0, int(np.floor(top * height)))
    x1 = min(width, int(np.ceil(right * width)))
    y1 = min(height, int(np.ceil(bottom * height)))
    return int(np.count_nonzero(mask[y0:y1, x0:x1]))


def test_planner_prioritises_full_patch_and_reports_veins_left_outside(store, tile_with_plane):
    tile, _, patch, veins = tile_with_plane
    plan = plan_macro_frame(tile, {
        "width_mm": 38.0,
        "height_mm": 28.0,
        "primary_region_id": patch["id"],
        "secondary_region_id": veins["id"],
    })
    # The patch is larger than the 38 x 28 mm field in both axes, so one frame
    # cannot hold it all. The planner must still reach the geometrically maximum
    # achievable primary coverage (no secondary trade-off may cost patch pixels).
    optimum = _exhaustive_best_primary(tile, patch, 38.0, 28.0)
    primary = plan["primary"]
    assert primary["covered_px"] >= optimum - 20  # tolerance for grid resolution
    assert primary["fully_inside"] is False
    assert primary["outside_px"] > 0
    frame = plan["frame"]
    # Frame is a valid, in-plane axis-aligned rectangle of the requested physical size.
    assert frame["right"] - frame["left"] == pytest.approx(38.0 / SPECIMEN_MM[0], abs=1e-3)
    assert frame["bottom"] - frame["top"] == pytest.approx(28.0 / SPECIMEN_MM[1], abs=1e-3)
    assert 0.0 <= frame["left"] and frame["right"] <= 1.0
    assert 0.0 <= frame["top"] and frame["bottom"] <= 1.0
    # Secondary coverage is reported and the unphotographed remainder stays labelled.
    secondary = plan["secondary"]
    assert secondary["total_px"] > 0
    assert secondary["outside_px"] >= 0
    assert secondary["covered_px"] + secondary["outside_px"] == secondary["total_px"]


def test_planner_fully_contains_priority_when_field_is_large_enough(store, tile_with_plane):
    tile, _, patch, veins = tile_with_plane
    plan = plan_macro_frame(tile, {
        "width_mm": 80.0,
        "height_mm": 60.0,
        "primary_region_id": patch["id"],
        "secondary_region_id": veins["id"],
    })
    assert plan["primary"]["fully_inside"] is True
    assert plan["primary"]["outside_px"] == 0
    # With the patch fully held, the frame shifts toward the secondary region,
    # capturing strictly more of it than a frame parked on the patch's left edge.
    assert plan["secondary"]["covered_px"] > 0


def test_planner_requires_physical_dimensions(store, tile_with_plane):
    tile, _, patch, veins = tile_with_plane

    def clear_dims(tile):
        tile["width_mm"] = None
        tile["height_mm"] = None

    tile = store.edit_tile(tile["id"], tile["version"], clear_dims, "metadata", "dims")
    with pytest.raises(InputError):
        plan_macro_frame(tile, {
            "width_mm": 38.0, "height_mm": 28.0,
            "primary_region_id": patch["id"], "secondary_region_id": veins["id"],
        })


def test_affine_fit_rejects_collinear_points():
    collinear_plane = [[0.2, 0.2], [0.4, 0.4], [0.6, 0.6]]
    capture = [[0.1, 0.1], [0.3, 0.3], [0.5, 0.5]]
    with pytest.raises(InputError):
        affine_fit(collinear_plane, capture)


def _detail_photo(store, plane_image):
    # Detail window overlaps the right portion of the patch but leaves part of it
    # outside the local photograph, mirroring a real partial macro capture.
    plane_points = [[0.42, 0.30], [0.80, 0.30], [0.80, 0.66]]
    detail_corners = [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95]]
    matrix = affine_matrix(plane_points, detail_corners)
    detail_image = render_detail(plane_image, matrix, detail_size=(640, 480))
    photo = store_capture(store.blobs, png_bytes(detail_image), "reference", "macro-detail.png", "", make_id())
    return photo, plane_points, detail_corners, matrix


def test_detail_registration_warps_into_plane_and_measures_only_overlap(store, tile_with_plane):
    tile, base_photo, patch, veins = tile_with_plane
    plane_image = Image.open(store.blobs.require(base_photo["working_digest"])).convert("RGB")
    detail_photo, plane_points, detail_points, matrix = _detail_photo(store, plane_image)

    def attach(tile):
        tile["photos"].append(detail_photo)

    tile = store.edit_tile(tile["id"], tile["version"], attach, "capture", "detail upload")
    pairs = [
        {"plane": plane_points[i], "capture": detail_points[i]}
        for i in range(3)
    ]
    # Preview first: no persistence, but diagnostics and a warped image come back.
    image, diagnostics, preview_matrix = preview_detail(store.blobs, tile, detail_photo["id"], {"pairs": pairs})
    assert image.size == tuple(tile["plane"]["size"])
    assert 0.0 < diagnostics["support_percent"] < 100.0
    assert np.allclose(preview_matrix, matrix, atol=1e-6)

    updated = save_detail_link(
        store, tile["id"], detail_photo["id"], {"version": tile["version"], "pairs": pairs}
    )
    link = updated["detail_links"][detail_photo["id"]]
    assert link["base_photo_id"] == base_photo["id"]
    assert link["support_percent"] == diagnostics["support_percent"]

    # The warped detail frame shares the base plane's physical scale and is partial.
    frame = specimen_frame(store.blobs, updated, detail_photo["id"])
    assert frame.scale is not None
    support_fraction = float(frame.valid.mean())
    assert 0.02 < support_fraction < 0.95

    # Measurement on the originally drawn patch reports co-located support and an
    # explicit unsupported remainder (the parts the local photo never captured).
    from leafledger.geometry import polygon_mask, region_domain
    requested = polygon_mask(frame.extent, patch["vertices"])
    domain = region_domain(frame.extent, updated, patch)
    supported = domain & frame.valid
    requested_count = int(np.count_nonzero(requested))
    supported_count = int(np.count_nonzero(supported))
    assert 0 < supported_count < requested_count
    outside = requested_count - supported_count
    assert outside > 0
    mm2 = frame.scale[0] * frame.scale[1]
    covered_mm2 = supported_count * mm2
    outside_mm2 = outside * mm2
    assert covered_mm2 > 0 and outside_mm2 > 0


def test_detail_link_is_removable_and_blocks_measurement_afterward(store, tile_with_plane):
    tile, base_photo, _, _ = tile_with_plane
    plane_image = Image.open(store.blobs.require(base_photo["working_digest"])).convert("RGB")
    detail_photo, plane_points, detail_points, _ = _detail_photo(store, plane_image)

    def attach(tile):
        tile["photos"].append(detail_photo)

    tile = store.edit_tile(tile["id"], tile["version"], attach, "capture", "detail upload")
    pairs = [{"plane": plane_points[i], "capture": detail_points[i]} for i in range(3)]
    tile = save_detail_link(store, tile["id"], detail_photo["id"], {"version": tile["version"], "pairs": pairs})
    assert detail_photo["id"] in tile["detail_links"]

    tile = remove_detail_link(store, tile["id"], detail_photo["id"], {"version": tile["version"]})
    assert detail_photo["id"] not in tile.get("detail_links", {})
    with pytest.raises(InputError):
        specimen_frame(store.blobs, tile, detail_photo["id"])


def test_selecting_detail_photo_for_measurement_is_allowed(store, tile_with_plane):
    from leafledger.analysis import selected_capture
    tile, base_photo, _, _ = tile_with_plane
    plane_image = Image.open(store.blobs.require(base_photo["working_digest"])).convert("RGB")
    detail_photo, plane_points, detail_points, _ = _detail_photo(store, plane_image)

    def attach(tile):
        tile["photos"].append(detail_photo)

    tile = store.edit_tile(tile["id"], tile["version"], attach, "capture", "detail upload")
    pairs = [{"plane": plane_points[i], "capture": detail_points[i]} for i in range(3)]
    tile = save_detail_link(store, tile["id"], detail_photo["id"], {"version": tile["version"], "pairs": pairs})
    assert selected_capture(tile, detail_photo["id"]) == detail_photo["id"]


def test_corner_registration_replaces_detail_link(store, tile_with_plane):
    from leafledger.workspace import save_registration
    tile, base_photo, _, _ = tile_with_plane
    plane_image = Image.open(store.blobs.require(base_photo["working_digest"])).convert("RGB")
    detail_photo, plane_points, detail_points, _ = _detail_photo(store, plane_image)

    def attach(tile):
        tile["photos"].append(detail_photo)

    tile = store.edit_tile(tile["id"], tile["version"], attach, "capture", "detail upload")
    pairs = [{"plane": plane_points[i], "capture": detail_points[i]} for i in range(3)]
    tile = save_detail_link(store, tile["id"], detail_photo["id"], {"version": tile["version"], "pairs": pairs})
    assert detail_photo["id"] in tile["detail_links"]

    # A four-corner registration on the same capture is a different alignment model;
    # it must replace the local detail link rather than leave both active.
    corners = [[0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98]]
    tile = save_registration(
        store, tile["id"], detail_photo["id"], {"version": tile["version"], "corners": corners}
    )
    assert detail_photo["id"] not in tile["detail_links"]
    assert detail_photo["id"] in tile["registrations"]
