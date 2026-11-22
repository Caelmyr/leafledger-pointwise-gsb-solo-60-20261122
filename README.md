# LeafLedger

LeafLedger is a local image workbench for mounted herbarium specimens. It keeps original files, aligns views in a measured folio plane, and saves each image measurement with its sources and observed support.

## Start

Run `docker compose up --build --wait` and visit `http://127.0.0.1:19965`. Set `LEAFLEDGER_PORT` to change the loopback port. The service keeps the measurements, images, and exported records in a dedicated named volume. `docker compose stop` keeps this data.

For a native start, install `requirements.txt` under Python 3.11 and run `./start.sh`. The Docker build can source the exact locked wheels from an existing verified image by setting `LEAFLEDGER_DEPS_IMAGE`. This selects the dependency stage alone.

## Survey a mounted specimen

Load the **training folios** to explore explicitly synthetic specimens, including a split-lamp folio with a cool lamp on the left, a warm lamp on the right and one neutral card at each end. Each has diffuse, parallel-polarized, cross-polarized and two raking-light photographs. The same mount, reference cards, bright flecks, paper tags, folded tissue and tape appear in each registered view.

- **Captures** imports the photo and retains the original bytes. Mark four fiducials around the plane, and register the same marks in additional captures. Compare the registered photos in split, blend, or difference views.
- **Sampling regions** marks the blade, background sheet, neutral cards, a near-tip fading reference and exclusions. Save one whole-specimen neutral balance across every marked card: a single set of channel gains corrects the full plane so genuine leaf color differences survive. Each card is checked against the 5% red/blue-to-green neutrality line; when no global fit satisfies every card, the best-effort correction is still saved and the plane is split into trusted color zones and explicitly hatched calibration-limited zones. The near-tip region always stays trusted as the future fading reference, and Reference match reuses the same zones so a failed card is never mistaken for a color difference.
- **Measure** can assess color changes, blade coverage, review small appearance observations, find glare and direction-dependent relief, map local texture and fine lines, verify mount margins, sample a line, compare against an approved reference, scan the visible midrib and mount seam, and locate pale flecks. **Blade width profile** reads cross-sections on a visually marked blade axis using the mount and blade samples. Unsupported cross-sections are explicitly labeled.
- **Geometry** saves visible area and trace length with physical dimensions when supplied. **Reference library** freezes an approved image sample. **Folio comparisons** group saved image observations by plant family, form, visit or folio.
- **Evidence** keeps a dated report or ZIP with its measured layers and optional originals. **Received evidence** verifies the archive manifest before opening a local copy.

## Limits

Captures may be PNG, JPEG, WebP or single-frame TIFF and are limited to 24 MiB and 12 megapixels. The registered plane is at most 1,800 pixels along its longest edge. Each specimen can hold up to 10 images and 24 region definitions. Received evidence accepts archives up to 24 MiB compressed and 72 MiB examined, with up to twelve measurements.

The workbench uses supported image appearance and physical plane scales, with approximate color and edge readings. Use the original images and measurement support to interpret spots, dried tissue, mounting material and glare. No plant identification or biological diagnosis is performed. Input handling honors embedded ICC profiles and orientation, treating unprofiled RGB images as sRGB.

The local PaveDesk optical registration, measurement and evidence modules provided the base for this workbench. LeafLedger has its own specimen imagery, measured blade cross-sections, and folio-specific workflows. Project `scripts/count_backend.py` can recount effective backend lines.
