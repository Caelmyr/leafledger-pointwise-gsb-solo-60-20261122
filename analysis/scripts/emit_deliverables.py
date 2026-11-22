"""Emit the user-facing deliverables from the chosen contour.

Produces:
  region_card.json        - the single closed polygon in LeafLedger region
                            format (normalized 0..1 vertices, <=40, inspection
                            purpose), ready to paste into a sampling region.
  occlusion_ledger.json   - this contour's occlusion bookkeeping ONLY. The
                            inferred (covered) area is scoped to this one leaf
                            so other leaves measured on the same sheet are
                            untouched. It records observed vs inferred area in
                            px and mm^2 and the exact inferred edge indices.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / "output" / "contour_result.json").read_text())

region_card = {
    "name": "Left target leaf - single closed contour",
    "purpose": "inspection",
    "vertices": r["contour_vertices_normalized"],
}
(ROOT / "output" / "region_card.json").write_text(json.dumps(region_card, indent=2))

inferred_edges = [i + 1 for i, tag in enumerate(r["edge_classes"]) if tag == "inferred"]
occlusion_ledger = {
    "scope": "left_target_leaf_only",
    "applies_to_other_leaves_on_sheet": False,
    "note": (
        "The inferred area is the smooth envelope bridge behind the mounting "
        "tape that connects the visible blade body to the collinear toothed "
        "apex. It is confined to the tape footprint (plus a 1-px connectivity "
        "edge band) and is charged to this contour alone."
    ),
    "detected_tape_bbox_px": r["detected_tape_px_bbox"],
    "inferred_polygon_edge_indices_1based": inferred_edges,
    "observed_edge_count": r["observed_edge_count"],
    "inferred_edge_count": r["inferred_edge_count"],
    "areas": {
        "raster_total_mm2": r["raster_area_mm2"],
        "observed_mm2": r["observed_area_mm2"],
        "inferred_mm2": r["inferred_area_mm2"],
        "inferred_fraction_percent": r["inferred_fraction_percent"],
        "vector_outline_mm2": round(
            r["vector_polygon_area_px"]
            * r["mm_per_px"][0] * r["mm_per_px"][1], 3),
        "units": "mm^2",
        "mm2_per_px": round(r["mm_per_px"][0] * r["mm_per_px"][1], 6),
    },
    "consistency_check": {
        "observed_plus_inferred_equals_total": (
            abs(r["observed_area_mm2"] + r["inferred_area_mm2"]
                - r["raster_area_mm2"]) < 0.002
        ),
    },
}
(ROOT / "output" / "occlusion_ledger.json").write_text(
    json.dumps(occlusion_ledger, indent=2))

print("region vertices:", len(region_card["vertices"]))
print("inferred edges (1-based):", inferred_edges)
print("observed %.3f + inferred %.3f = %.3f mm^2" % (
    r["observed_area_mm2"], r["inferred_area_mm2"], r["raster_area_mm2"]))
print("wrote region_card.json and occlusion_ledger.json")
