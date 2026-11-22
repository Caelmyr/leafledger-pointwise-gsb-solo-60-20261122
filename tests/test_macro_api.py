import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leafledger.photos import png_bytes, store_capture  # noqa: E402
from leafledger.store import make_id  # noqa: E402
from leafledger.web import create_app  # noqa: E402

from tests.support.imaging import affine_matrix, make_plane_image, render_detail  # noqa: E402


PLANE_SIZE = (900, 680)


def rectangle(name, purpose, left, top, right, bottom):
    return {
        "id": make_id(),
        "name": name,
        "purpose": purpose,
        "vertices": [[left, top], [right, top], [right, bottom], [left, bottom]],
    }


@pytest.fixture()
def client(tmp_path):
    app = create_app(data_dir=str(tmp_path / "data"))
    app.testing = True
    return app.test_client()


@pytest.fixture()
def specimen(client):
    created = client.post("/api/specimens", json={
        "name": "HTTP macro specimen", "width_mm": 300.0, "height_mm": 226.6667,
    })
    assert created.status_code == 201
    tile = created.get_json()
    image = make_plane_image(*PLANE_SIZE)
    data = {
        "photo": (io.BytesIO(png_bytes(image)), "survey.png"),
        "version": str(tile["version"]),
        "role": "diffuse",
    }
    uploaded = client.post(f"/api/specimens/{tile['id']}/photos", data=data, content_type="multipart/form-data")
    assert uploaded.status_code == 201
    tile = uploaded.get_json()
    base_photo = tile["photos"][0]
    # Save the full-plane corners explicitly.
    saved = client.put(f"/api/specimens/{tile['id']}/plane", json={
        "version": tile["version"],
        "photo_id": base_photo["id"],
        "corners": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "respect_scale": True,
    })
    assert saved.status_code == 200
    tile = saved.get_json()
    patch = rectangle("Rare discoloration", "inspection", 300 / 900, 285 / 680, 470 / 900, 395 / 680)
    veins = rectangle("Vein tips", "inspection", 500 / 900, 285 / 680, 640 / 900, 395 / 680)
    for region in (patch, veins):
        response = client.post(f"/api/specimens/{tile['id']}/regions", json={
            "version": tile["version"], "name": region["name"],
            "purpose": "inspection", "vertices": region["vertices"],
        })
        assert response.status_code == 201
        tile = response.get_json()
        region["id"] = next(r["id"] for r in tile["regions"] if r["name"] == region["name"])
    return tile, base_photo, patch, veins, image


def test_macro_plan_endpoint_returns_json_and_overlay(client, specimen):
    tile, _, patch, veins, _ = specimen
    response = client.post(f"/api/specimens/{tile['id']}/macro-plan", json={
        "width_mm": 38.0, "height_mm": 28.0,
        "primary_region_id": patch["id"], "secondary_region_id": veins["id"],
    })
    assert response.status_code == 200
    plan = response.get_json()
    assert plan["field_mm"] == [38.0, 28.0]
    assert set(plan["frame"]) == {"left", "top", "right", "bottom"}
    assert plan["primary"]["outside_px"] > 0

    overlay = client.post(
        f"/api/specimens/{tile['id']}/macro-plan?overlay=1",
        json={
            "width_mm": 38.0, "height_mm": 28.0,
            "primary_region_id": patch["id"], "secondary_region_id": veins["id"],
        },
    )
    assert overlay.status_code == 200
    assert overlay.mimetype == "image/png"
    with Image.open(io.BytesIO(overlay.data)) as picture:
        assert picture.size == tuple(tile["plane"]["size"])


def test_macro_plan_rejects_oversized_field(client, specimen):
    tile, _, patch, veins, _ = specimen
    response = client.post(f"/api/specimens/{tile['id']}/macro-plan", json={
        "width_mm": 999.0, "height_mm": 28.0,
        "primary_region_id": patch["id"], "secondary_region_id": veins["id"],
    })
    assert response.status_code == 422
    assert "larger than" in response.get_json()["error"]


def _upload_detail(client, tile, plane_image):
    plane_points = [[0.42, 0.30], [0.80, 0.30], [0.80, 0.66]]
    detail_corners = [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95]]
    matrix = affine_matrix(plane_points, detail_corners)
    detail = render_detail(plane_image, matrix, detail_size=(640, 480))
    response = client.post(f"/api/specimens/{tile['id']}/photos", data={
        "photo": (io.BytesIO(png_bytes(detail)), "macro-detail.png"),
        "version": str(tile["version"]),
        "role": "reference",
    }, content_type="multipart/form-data")
    assert response.status_code == 201
    updated = response.get_json()
    detail_photo = updated["photos"][-1]
    pairs = [{"plane": plane_points[i], "capture": detail_corners[i]} for i in range(3)]
    return updated, detail_photo, pairs


def test_detail_link_preview_save_and_delete(client, specimen):
    tile, base_photo, patch, veins, plane_image = specimen
    tile, detail_photo, pairs = _upload_detail(client, tile, plane_image)

    preview = client.post(
        f"/api/specimens/{tile['id']}/detail-links/{detail_photo['id']}/preview",
        json={"pairs": pairs},
    )
    assert preview.status_code == 200
    assert preview.mimetype == "image/png"
    diagnostics = preview.headers["X-Detail-Alignment"]
    assert "support_percent" in diagnostics

    saved = client.put(f"/api/specimens/{tile['id']}/detail-links/{detail_photo['id']}", json={
        "version": tile["version"], "pairs": pairs,
    })
    assert saved.status_code == 200
    linked = saved.get_json()
    assert detail_photo["id"] in linked["detail_links"]
    photo_state = next(p for p in linked["photos"] if p["id"] == detail_photo["id"])
    assert photo_state["registration_state"] == "detail"

    # The warped detail is viewable through the standard registered-viewer route.
    viewer = client.get(f"/api/specimens/{tile['id']}/viewer/{detail_photo['id']}?layer=raw")
    assert viewer.status_code == 200
    assert viewer.mimetype == "image/png"

    removed = client.delete(f"/api/specimens/{tile['id']}/detail-links/{detail_photo['id']}", json={
        "version": linked["version"],
    })
    assert removed.status_code == 200
    assert detail_photo["id"] not in removed.get_json()["detail_links"]


def test_detail_link_rejects_collinear_points(client, specimen):
    tile, _, _, _, plane_image = specimen
    tile, detail_photo, _ = _upload_detail(client, tile, plane_image)
    bad = [
        {"plane": [0.2, 0.2], "capture": [0.1, 0.1]},
        {"plane": [0.4, 0.4], "capture": [0.3, 0.3]},
        {"plane": [0.6, 0.6], "capture": [0.5, 0.5]},
    ]
    response = client.put(f"/api/specimens/{tile['id']}/detail-links/{detail_photo['id']}", json={
        "version": tile["version"], "pairs": bad,
    })
    assert response.status_code == 422
    assert "collinear" in response.get_json()["error"].lower()


def test_measurement_on_detail_reports_unsupported_remainder(client, specimen):
    tile, _, patch, _, plane_image = specimen
    tile, detail_photo, pairs = _upload_detail(client, tile, plane_image)
    saved = client.put(f"/api/specimens/{tile['id']}/detail-links/{detail_photo['id']}", json={
        "version": tile["version"], "pairs": pairs,
    })
    tile = saved.get_json()
    # Color variation measured on the detail capture over the originally drawn patch.
    result = client.post(f"/api/specimens/{tile['id']}/results", json={
        "kind": "color",
        "photo_id": detail_photo["id"],
        "region_id": patch["id"],
        "apply_neutral": False,
        "options": {"grid": 5},
    })
    assert result.status_code == 201
    record = result.get_json()
    support = record["support"]
    assert support["available_pixels"] < support["requested_pixels"]
    assert support["available_percent"] < 100.0
    assert record["provenance"]["photo"]["id"] == detail_photo["id"]
