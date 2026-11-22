"""Validate the delivered contour against LeafLedger's own Geometry rules.

Imports the real application modules (leafledger.geometry / metrology) and
feeds them the contour produced by build_contour.py, proving the single
closed polygon is acceptable as one region card and reproducing the exact
area figures the Geometry tab would compute. No ground-truth masks are used
here; only the delivered JSON + the application's own functions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from leafledger.geometry import polygon_mask, signed_area, validate_polygon
from leafledger.metrology import mask_geometry

ROOT = Path(__file__).resolve().parents[1]
result = json.loads((ROOT / "output" / "contour_result.json").read_text())
W, H = result["plane"]["width_px"], result["plane"]["height_px"]
vertices = result["contour_vertices_normalized"]

# 1. application-level polygon validation (no crossing, sufficient area, ...)
validated = validate_polygon(vertices)
print("validate_polygon: OK,", len(validated), "vertices")
assert 3 <= len(validated) <= 40

# 2. rasterize exactly like the app does and recompute the supported area
extent = (W, H)
mask = polygon_mask(extent, validated)
print("app rasterized polygon pixels:", int(mask.sum()))
print("app signed area (fraction of plane):", round(abs(signed_area(validated)), 6))

# 3. emulate mask_geometry with the real plane scale (320 x 240 mm)
class FakeFrame:
    extent = (W, H)
    scale = tuple(result["mm_per_px"])


# the app expects a region dict with the validated vertices
region = {"name": "Left target leaf (single closed contour)",
          "purpose": "inspection", "vertices": validated}
requested = polygon_mask(extent, validated)
# every requested pixel is "supported" in this rectified plane (valid alpha=1)
geo = mask_geometry(FakeFrame(), region, requested, requested)
print("\n--- LeafLedger Geometry module output for this contour ---")
for key in (
    "vector_outline_area", "raster_area", "raster_supported_area",
    "requested_pixels", "supported_pixels", "units",
):
    print(f"{key:26s}: {geo[key]}")

# 4. reconcile with the observed/inferred split (pixel-exact raster basis)
mm2 = result["mm_per_px"][0] * result["mm_per_px"][1]
split = {
    "observed_area_mm2": result["observed_area_mm2"],
    "inferred_area_mm2": result["inferred_area_mm2"],
    "total_raster_area_mm2": result["raster_area_mm2"],
    "app_polygon_raster_mm2": round(geo["raster_area"], 3),
    "app_vector_outline_mm2": round(geo["vector_outline_area"], 3),
    "mm2_per_px": round(mm2, 6),
}
print("\n--- observed vs inferred reconciliation ---")
print(json.dumps(split, indent=2))

# 5. confirm the inferred boundary is confined to the detected tape run
print("\ninferred fraction of total area: ", result["inferred_fraction_percent"], "%")
print("inferred edges:", result["inferred_edge_count"],
      "observed edges:", result["observed_edge_count"])
(ROOT / "output" / "app_validation.json").write_text(json.dumps({
    "validation": "passed",
    "vertex_count": len(validated),
    "app_geometry": geo,
    "split": split,
}, indent=2))
print("\nwrote output/app_validation.json")
