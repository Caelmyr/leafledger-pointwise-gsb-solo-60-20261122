# LeafLedger

LeafLedger is a local image workbench for mounted herbarium specimens. It keeps original files, aligns views in a measured folio plane, and saves each image measurement with its sources and observed support.

## Start

Run `docker compose up --build --wait` and visit `http://127.0.0.1:19965`. Set `LEAFLEDGER_PORT` to change the loopback port. The service keeps the measurements, images, and exported records in a dedicated named volume. `docker compose stop` keeps this data.

For a native start, install `requirements.txt` under Python 3.11 and run `./start.sh`. The Docker build can source the exact locked wheels from an existing verified image by setting `LEAFLEDGER_DEPS_IMAGE`. This selects the dependency stage alone.

## Survey a mounted specimen

Load the four **training folios** to explore explicitly synthetic specimens. Each has diffuse, parallel-polarized, cross-polarized and two raking-light photographs. The same mount, reference card, bright flecks, paper tags, folded tissue and tape appear in each registered view.

- **Captures** imports the photo and retains the original bytes. Mark four fiducials around the plane, and register the same marks in additional captures. Compare the registered photos in split, blend, or difference views.
- **Macro frame planning** turns the measured specimen size and a macro field (for example 38 × 28 mm) into one reachable camera rectangle in the survey plane. It holds the priority region (a rare discoloration patch) as completely as the physical field allows, then maximizes a secondary region (a vein-tip row to re-check). The overlay and JSON report mark exactly which pixels stay outside the frame, so nothing outside the reachable rectangle is implied to be photographed.
- **Local detail alignment** handles a partial macro photograph whose corners do not appear in the survey photo. Mark the same three (up to six) sites on both images and an affine least-squares transform warps the detail into the specimen plane. Existing **Measure** and **Geometry** runs then use the detail directly: only co-located, supported pixels are measured, and every part the detail did not capture stays explicitly labelled unsupported/unmeasured.
- **Sampling regions** marks the blade, background sheet, neutral card and exclusions. Save neutral-card balance when you need comparable appearance across captures.
- **Measure** can assess color changes, blade coverage, review small appearance observations, find glare and direction-dependent relief, map local texture and fine lines, verify mount margins, sample a line, compare against an approved reference, scan the visible midrib and mount seam, and locate pale flecks. **Blade width profile** reads cross-sections on a visually marked blade axis using the mount and blade samples. Unsupported cross-sections are explicitly labeled.
- **Geometry** saves visible area and trace length with physical dimensions when supplied. **Reference library** freezes an approved image sample. **Folio comparisons** group saved image observations by plant family, form, visit or folio.
- **Evidence** keeps a dated report or ZIP with its measured layers and optional originals. **Received evidence** verifies the archive manifest before opening a local copy.

## Limits

Captures may be PNG, JPEG, WebP or single-frame TIFF and are limited to 24 MiB and 12 megapixels. The registered plane is at most 1,800 pixels along its longest edge. Each specimen can hold up to 10 images and 24 region definitions. Received evidence accepts archives up to 24 MiB compressed and 72 MiB examined, with up to twelve measurements.

The workbench uses supported image appearance and physical plane scales, with approximate color and edge readings. Use the original images and measurement support to interpret spots, dried tissue, mounting material and glare. No plant identification or biological diagnosis is performed. Input handling honors embedded ICC profiles and orientation, treating unprofiled RGB images as sRGB.

The local PaveDesk optical registration, measurement and evidence modules provided the base for this workbench. LeafLedger has its own specimen imagery, measured blade cross-sections, and folio-specific workflows. Project `scripts/count_backend.py` can recount effective backend lines.
