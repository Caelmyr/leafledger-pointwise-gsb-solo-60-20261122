from .color import robust_neutral
from .geometry import (
    DEFAULT_CORNERS,
    canvas_dimensions,
    corners_parameter,
    homography,
    region_domain,
    specimen_frame,
    validate_polygon,
)
from .macro import detail_transform
from .model import (
    ConflictError,
    InputError,
    MAX_REGIONS,
    MAX_TILE_IMAGES,
    ROLE_LABELS,
    dictionary,
    enumeration,
    flag,
    identity,
    match_slot,
    model_version,
    region_fields,
    scalar,
    string,
    tile_fields,
)
from .photos import diagnostic_flags, store_capture
from .store import make_id


def describe_photo(photo, plane, registrations, detail_links=None):
    info = {**photo}
    info["role_label"] = ROLE_LABELS[photo["role"]]
    info["diagnostics"] = diagnostic_flags(photo)
    is_base = plane is not None and plane["photo_id"] == photo["id"]
    detail = bool(detail_links and photo["id"] in detail_links)
    if is_base:
        state = "base"
    elif detail:
        state = "detail"
    elif photo["id"] in registrations:
        state = "registered"
    else:
        state = "unregistered"
    info["registration_state"] = state
    info["preview_url"] = f"/api/captures/{photo['id']}/preview"
    return info


def summarize_tile(tile):
    base = tile.get("plane")
    photos = tile["photos"]
    cover = next((item for item in photos if base and item["id"] == base["photo_id"]), None)
    if cover is None and photos:
        cover = photos[0]
    summary = {
        key: tile[key] for key in (
            "id", "name", "batch", "body", "glaze", "firing", "width_mm", "height_mm", "version", "synthetic",
        )
    }
    return {
        **summary,
        "photo_count": len(photos),
        "region_count": len(tile["regions"]),
        "cover_url": f"/api/specimens/{tile['id']}/photos/{cover['id']}/thumbnail" if cover else None,
        "registered_count": int(bool(base)) + len(tile["registrations"]) + len(tile.get("detail_links", {})),
        "updated_at": tile["updated_at"],
    }


def present_tile(tile):
    registrations = tile["registrations"]
    detail_links = tile.get("detail_links", {})
    photos = []
    for record in tile["photos"]:
        data = describe_photo(record, tile["plane"], registrations, detail_links)
        data["preview_url"] = f"/api/specimens/{tile['id']}/photos/{record['id']}/preview"
        data["thumbnail_url"] = f"/api/specimens/{tile['id']}/photos/{record['id']}/thumbnail"
        data["original_url"] = f"/api/specimens/{tile['id']}/photos/{record['id']}/original"
        if data["registration_state"] != "unregistered":
            data["registered_url"] = f"/api/specimens/{tile['id']}/viewer/{record['id']}?v={tile['version']}"
        if record["id"] in detail_links:
            data["detail_support_percent"] = detail_links[record["id"]].get("support_percent")
        photos.append(data)
    native = tile.get("plane")
    return {
        **tile,
        "photos": photos,
        "canvas_size": native["size"] if native else None,
    }


def edit_specimen(store, identifier, payload):
    version = model_version(payload)

    def change(tile):
        updated = tile_fields(payload, tile)
        tile.update(updated)

    return store.edit_tile(identifier, version, change, "metadata", "Updated specimen details")


def import_photo(store, tile_id, expected, binary, filename, role, note):
    validated_role = enumeration(role, ROLE_LABELS, "Lighting role")
    validated_note = string(note, "Capture notes", maximum=500, required=False)
    current = store.tile(tile_id)
    store.require_version(current, expected)
    if len(current["photos"]) >= MAX_TILE_IMAGES:
        raise InputError(f"A specimen supports at most {MAX_TILE_IMAGES} photographs.")
    record = store_capture(store.blobs, binary, validated_role, filename, validated_note, make_id())
    if any(photo["source_digest"] == record["source_digest"] and photo["role"] == validated_role for photo in current["photos"]):
        raise InputError("This image is already on the specimen with the same lighting role.")

    def change(tile):
        tile["photos"].append(record)
        if tile["plane"] is None and len(tile["photos"]) == 1:
            size = canvas_dimensions(record, DEFAULT_CORNERS)
            tile["plane"] = {"photo_id": record["id"], "corners": DEFAULT_CORNERS, "size": list(size)}

    return store.edit_tile(tile_id, expected, change, "capture", f"Imported {ROLE_LABELS[validated_role]} · {record['name']}")


def edit_photo(store, tile_id, capture_id, payload):
    expected = model_version(payload)
    capture_id = identity(capture_id, "Photo ID")

    def change(tile):
        source = match_slot(tile["photos"], capture_id, "photo")
        revised = enumeration(payload.get("role", source["role"]), ROLE_LABELS, "Lighting role")
        if any(
            item["id"] != capture_id
            and item["source_digest"] == source["source_digest"]
            and item["role"] == revised
            for item in tile["photos"]
        ):
            raise InputError("The same image already has this role on the specimen.")
        source["role"] = revised
        source["note"] = string(
            payload.get("note", source["note"]),
            "Capture notes",
            maximum=500,
            required=False,
        )

    return store.edit_tile(tile_id, expected, change, "capture", "Updated photo notes and lighting label")


def remove_photo(store, tile_id, capture_id, payload):
    expected = model_version(payload)
    capture_id = identity(capture_id, "Photo ID")
    reset = flag(payload.get("reset_plane", False), "Reset specimen plane")

    def change(tile):
        photo = match_slot(tile["photos"], capture_id, "photo")
        if tile["plane"] and tile["plane"]["photo_id"] == capture_id:
            if (tile["registrations"] or tile["regions"]) and not reset:
                raise ConflictError("Removing the base photo requires resetting its current sampling plane.")
            tile["plane"] = None
            tile["calibration"] = None
            tile["registrations"] = {}
            tile["regions"] = []
        tile["registrations"].pop(capture_id, None)
        tile.get("detail_links", {}).pop(capture_id, None)
        tile["photos"].remove(photo)

    return store.edit_tile(tile_id, expected, change, "capture", "Removed a photo from the live specimen")


def preview_transform(store, tile_id, photo_id, payload, mode):
    current = store.tile(tile_id)
    photo_id = identity(photo_id, "Photo ID")
    marked = corners_parameter(payload.get("corners"))
    photo = match_slot(current["photos"], photo_id, "photo")
    if mode == "base":
        width = current["width_mm"] if payload.get("respect_scale", True) else None
        height = current["height_mm"] if payload.get("respect_scale", True) else None
        dims = canvas_dimensions(photo, marked, width, height)
        updated = {**current, "plane": {"photo_id": photo_id, "corners": marked, "size": list(dims)}}
    else:
        if current["plane"] is None or current["plane"]["photo_id"] == photo_id:
            raise InputError("Set another photo as the base before checking registration.")
        change = {"base_photo_id": current["plane"]["photo_id"], "corners": marked}
        updated = {**current, "registrations": {**current["registrations"], photo_id: change}}
    return specimen_frame(store.blobs, updated, photo_id)


def save_plane(store, tile_id, payload):
    dictionary(payload, "Specimen plane")
    expected = model_version(payload)
    photo_id = identity(payload.get("photo_id"), "Base photo ID")
    corners = corners_parameter(payload.get("corners"))
    reset = flag(payload.get("reset_regions", False), "Reset specimen plane")
    respect_scale = flag(payload.get("respect_scale", True), "Respect specimen dimensions")
    homography(DEFAULT_CORNERS, corners)

    def change(tile):
        photo = match_slot(tile["photos"], photo_id, "photo")
        size = canvas_dimensions(
            photo, corners,
            tile["width_mm"] if respect_scale else None,
            tile["height_mm"] if respect_scale else None,
        )
        updated = {"photo_id": photo_id, "corners": corners, "size": list(size)}
        if tile["plane"] == updated:
            return
        if (tile["registrations"] or tile["regions"]) and not reset:
            raise ConflictError("Changing the specimen plane requires confirming reset of its live regions and registrations.")
        tile["plane"] = updated
        tile["registrations"].clear()
        tile.get("detail_links", {}).clear()
        tile["regions"].clear()
        tile["calibration"] = None

    return store.edit_tile(tile_id, expected, change, "plane", "Saved the photographed tile plane")


def save_registration(store, identifier, photo_id, payload):
    expected = model_version(payload)
    photo_id = identity(photo_id, "Photo ID")
    corners = corners_parameter(payload.get("corners"))
    homography(DEFAULT_CORNERS, corners)
    current = store.tile(identifier)
    store.require_version(current, expected)
    frame = preview_transform(store, identifier, photo_id, payload, "registration")
    supported = float(frame.valid.mean())
    if supported < 0.15:
        raise InputError("Mark a region with at least 15% visible support.")

    def change(tile):
        tile.get("detail_links", {}).pop(photo_id, None)
        tile["registrations"][photo_id] = {
            "base_photo_id": tile["plane"]["photo_id"],
            "corners": corners,
            "support_percent": round(100 * supported, 2),
        }

    return store.edit_tile(identifier, expected, change, "registration", "Registered a cross-light capture")


def save_region(store, identifier, payload, region_id=None):
    expected = model_version(payload)
    region = region_fields(payload)
    validate_polygon(region["vertices"])
    chosen = identity(region_id, "Region ID") if region_id is not None else make_id()
    region["id"] = chosen

    def change(tile):
        if not tile["plane"]:
            raise InputError("Set a base photo before defining a sampling region.")
        if region_id is None:
            if len(tile["regions"]) >= MAX_REGIONS:
                raise InputError(f"A specimen supports at most {MAX_REGIONS} regions.")
            tile["regions"].append(region)
        else:
            previous = match_slot(tile["regions"], chosen, "sampling region")
            index = tile["regions"].index(previous)
            tile["regions"][index] = region
        if tile["calibration"]:
            reasons = {region["purpose"], previous["purpose"]} if region_id is not None else {region["purpose"]}
            if tile["calibration"]["region_id"] == chosen or "ignore" in reasons:
                tile["calibration"] = None

    return store.edit_tile(identifier, expected, change, "region", f"Saved sampling region {region['name']}")


def remove_region(store, identifier, region_id, payload):
    expected = model_version(payload)
    chosen = identity(region_id, "Region ID")

    def change(tile):
        region = match_slot(tile["regions"], chosen, "sampling region")
        tile["regions"].remove(region)
        if tile["calibration"] and (tile["calibration"]["region_id"] == chosen or region["purpose"] == "ignore"):
            tile["calibration"] = None

    return store.edit_tile(identifier, expected, change, "region", "Removed a live sampling region")


def calibrate(store, identifier, payload):
    expected = model_version(payload)
    region_id = identity(payload.get("region_id"), "Neutral card region")
    target = scalar(payload.get("target", 0.18), "Neutral reflectance target", lower=0.08, upper=0.8)
    before = store.tile(identifier)
    store.require_version(before, expected)
    selected = match_slot(before["regions"], region_id, "neutral card")
    if selected["purpose"] != "neutral":
        raise InputError("Select a region marked as a neutral color card.")
    frame = specimen_frame(store.blobs, before)
    area = region_domain(frame.extent, before)
    record = robust_neutral(frame, selected, area, target)
    record["plane_size"] = before["plane"]["size"]
    record["plane_corners"] = before["plane"]["corners"]
    record["region_vertices"] = selected["vertices"]

    def change(tile):
        tile["calibration"] = record

    return store.edit_tile(identifier, expected, change, "calibration", "Saved neutral-card image gains")


def clear_calibration(store, identifier, payload):
    expected = model_version(payload)

    def change(tile):
        tile["calibration"] = None

    return store.edit_tile(identifier, expected, change, "calibration", "Cleared neutral-card image gains")


MIN_DETAIL_SUPPORT = 0.02


def save_detail_link(store, identifier, photo_id, payload):
    expected = model_version(payload)
    current = store.tile(identifier)
    store.require_version(current, expected)
    photo, matrix, residuals, pairs = detail_transform(current, photo_id, payload)
    # Build the warped frame through the public path after a provisional write so
    # the same validation and support rules apply as for full-plane registrations.
    provisional = {
        **current,
        "detail_links": {
            **current.get("detail_links", {}),
            photo_id: {
                "base_photo_id": current["plane"]["photo_id"],
                "transform": [round(float(value), 10) for value in matrix.ravel()],
            },
        },
    }
    preview = specimen_frame(store.blobs, provisional, photo_id)
    supported = float(preview.valid.mean())
    if supported < MIN_DETAIL_SUPPORT:
        raise InputError(
            f"The aligned detail covers only {supported * 100:.2f}% of the plane; mark matching points more accurately."
        )
    max_residual = float(max(residuals)) if residuals else 0.0

    def change(tile):
        links = tile.setdefault("detail_links", {})
        # A detail capture uses local point alignment instead of four plane corners.
        tile["registrations"].pop(photo_id, None)
        links[photo_id] = {
            "base_photo_id": tile["plane"]["photo_id"],
            "transform": [round(float(value), 10) for value in matrix.ravel()],
            "point_pairs": pairs,
            "point_residuals": residuals,
            "max_residual": round(max_residual, 6),
            "support_percent": round(100 * supported, 3),
        }

    return store.edit_tile(
        identifier,
        expected,
        change,
        "registration",
        f"Aligned local detail capture · {photo['name']}",
    )


def remove_detail_link(store, identifier, photo_id, payload):
    expected = model_version(payload)

    def change(tile):
        links = tile.get("detail_links", {})
        if photo_id not in links:
            raise InputError("This capture has no local detail alignment.")
        links.pop(photo_id)

    return store.edit_tile(identifier, expected, change, "registration", "Removed a local detail alignment")
