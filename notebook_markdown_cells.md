# Notebook Markdown Cells — Riparian Encroachment Detection Pipeline

Each section below is a self-contained Markdown cell. Copy-paste directly into your `.ipynb`, positioned immediately above the code cell it documents (indicated in the cell comment).

---

## CELL 1 — Insert at the top of the notebook (Title + Problem Statement)

```markdown
# Riparian Encroachment Detection & Structure Segmentation

### Automated Mapping of Informal Settlements Within Regulated River Buffer Zones — Kasarani, Nairobi

---

## 1. Problem Statement

Kenya's Water Act designates a mandatory riparian buffer along all rivers and streams — a
setback zone in which permanent construction is legally prohibited. In rapidly urbanizing
areas such as **Kasarani, Nairobi**, informal settlements have expanded directly into these
buffer zones, increasing flood risk, degrading water quality, and complicating enforcement
for county planning authorities.

Manual field surveys cannot keep pace with this growth. This notebook builds an automated,
satellite-imagery-driven pipeline to **detect, count, and measure** individual structures
encroaching on the riparian buffer, producing a GIS-ready output for compliance reporting.

## 2. The Core Technical Challenge: "The Blob Limitation"

Informal settlements are characterized by **high roof density** — corrugated iron (mabati)
sheets are frequently butted directly against neighboring structures, with little to no
visible gap between rooftops.

A standard single-stage computer vision model (e.g. a segmentation network run directly on
raw imagery, or basic connected-component analysis on a material mask) treats spectrally
and spatially continuous roof material as **one object**. The result:

- Ten adjacent one-room dwellings are detected as **a single building footprint**.
- Building **counts are severely undercounted**, sometimes by an order of magnitude.
- Footprint area calculations are meaningless, since one "structure" spans multiple households.

> **Key Takeaway:** For encroachment analytics to be legally and administratively useful,
> the pipeline must resolve *individual* structures, not just detect "built-up area." This
> is the design driver behind the two-step methodology below.

## 3. Methodology Overview

This pipeline decouples **material classification** from **instance boundary delineation**,
using a purpose-built model for each task:

| Step | Model | Role |
|------|-------|------|
| 1. Material Highlighter | Random Forest (pixel-level) | Isolates corrugated iron roof pixels from vegetation, bare soil, and water |
| 2. Structure Boundary Tracer | YOLOv8-seg (instance segmentation) | Splits touching roofs into distinct, countable structures using shadow lines, ridgelines, and corner cues |

The output of Step 1 (a binary material mask) becomes the label source for Step 2. Step 2's
output polygons are then georeferenced and converted into planning-grade metrics.
```

---

## CELL 2 — Insert before the study area / data acquisition cells (rivers + satellite composite)

```markdown
## Data Acquisition: Study Area & Source Imagery

Before classification can begin, the pipeline assembles its two core spatial inputs:

- **River network (vector):** Pulled from OpenStreetMap via `osmnx`, filtered to
  `waterway` tags (`river`, `stream`) for the Kasarani, Nairobi study area. This defines
  the centerlines from which the regulated buffer zone is constructed.
- **Satellite composite (raster):** A multi-band optical composite retrieved and stacked
  via `pystac-client` / `stackstac`, clipped to the riparian buffer extent using `rioxarray`.

> **Key Takeaway:** All downstream classification is only as good as this alignment step —
> the raster and vector layers must share a consistent CRS before any pixel-level analysis
> is trustworthy.
```

---

## CELL 3 — Insert directly above the Random Forest training cell (Step 1: Material Highlighter)

```markdown
## Step 1 — Material Highlighter: Pixel-Level Spectral Classification

**Objective:** Produce a clean binary mask isolating corrugated iron roofing from all other
land cover in the scene.

### Why Random Forest (and not a deep model) for this step

Corrugated iron has a distinctive, consistent spectral signature relative to vegetation,
bare soil, and water across the visible/NIR bands. This is a **low-dimensional, tabular
classification problem** at the per-pixel level — exactly the regime where a Random Forest
classifier is fast to train, resistant to overfitting on small labeled samples, and easy to
validate, without the labeled-data volume a deep segmentation network would require.

### Process

1. **Training samples** are digitized as polygons (`training_data.shp`) over known examples
   of each class: `corrugated_iron`, `vegetation`, `bare_soil`, `water`.
2. Pixel values are extracted from the source raster within each labeled polygon using
   `rasterio.mask`.
3. A `RandomForestClassifier` (scikit-learn) is trained on these labeled pixel vectors.
4. The trained model is applied across the full raster extent, producing a **binary
   material mask** (`corrugated_iron_mask.tif`) — 1 = candidate roof pixel, 0 = everything else.

> **Key Takeaway:** This step answers *"where is roofing material?"* — it does **not**
> answer *"how many separate buildings are there?"*. That question is deferred entirely to
> Step 2, which is why the mask is allowed to contain large, fused blobs at this stage.
```

---

## CELL 4 — Insert directly above the tiling / label-generation cell (Step 2 intro + preprocessing)

```markdown
## Step 2 — Structure Boundary Tracer: Instance Segmentation with YOLOv8-seg

**Objective:** Take the fused material mask from Step 1 and split it into individually
countable structure polygons.

### Why instance segmentation, not contour detection alone

Running `cv2.findContours` directly on the binary mask would simply trace the outline of
each *connected* blob — which is exactly the failure mode described in the Problem
Statement. Instead, contours from the mask are used to generate **training labels**, and a
YOLOv8-seg model is trained to learn the *visual* cues that separate adjacent roofs even
when their material footprint is contiguous:

- Fine shadow lines cast between adjoining roof edges
- Ridgeline and gutter geometry
- Corner and corrugation-direction discontinuities at roof boundaries

Because these cues are learned from pixel context (not just the mask), the trained model
generalizes to fused blobs where a pure mask-contour approach would fail.

### 2.1 Data Preprocessing & Tiling

YOLOv8 expects fixed-size inputs, so the source raster and its mask are tiled together:

- The satellite image and `corrugated_iron_mask.tif` are sliced into **640×640 px** tiles
  in lockstep, ensuring each image tile has a spatially matching mask tile.
- Rasters smaller than the tile size are zero-padded before slicing.
- For each mask tile, `cv2.findContours` extracts per-structure boundary contours (small
  noise contours below an area threshold are discarded).
- Each contour is converted into a **normalized YOLO segmentation polygon**:

  ```
  class x1 y1 x2 y2 x3 y3 ... xn yn
  ```

  where every `x`/`y` coordinate pair is divided by the tile size (`640`) so all values
  fall in the `[0, 1]` range, as required by the YOLOv8-seg label format.

> **Key Takeaway:** The Random Forest mask isn't the final answer — it's **repurposed as
> free, automatically-generated training labels** for the segmentation network. This avoids
> the cost of manually annotating thousands of individual rooftops by hand.
```

---

## CELL 5 — Insert directly above the `dataset.yaml` generation cell

```markdown
### 2.2 Dataset Configuration

A `dataset.yaml` file is auto-generated to point Ultralytics' YOLOv8 training routine at
the tiled dataset:

- `path` — root of the generated `dataset/` folder
- `train` / `val` — relative paths to the image tiles (validation reuses the training
  split here as a baseline; a held-out split should be introduced for production runs)
- `names` — single class, `corrugated_iron_structure`

This keeps the pipeline **self-configuring**: re-running the tiling cell regenerates both
the dataset and its manifest without manual bookkeeping.
```

---

## CELL 6 — Insert directly above the `model.train(...)` cell

```markdown
### 2.3 Model Training

The segmentation model is initialized from the pretrained `yolov8n-seg` checkpoint (the
nano variant, chosen for fast iteration and low compute overhead) and fine-tuned on the
tiled rooftop dataset:

- **Epochs:** 50
- **Image size:** 640×640 (matching the tile size used during preprocessing)
- **Batch size:** 16

Training and validation metrics (box/mask loss, precision, recall) are logged automatically
to `riparian_encroachment/structure_tracer/`, allowing quick visual inspection of predicted
mask quality against the input tiles before moving to full-scene inference.

> **Key Takeaway:** At this stage the model outputs **pixel-space polygons per tile** — they
> have no real-world coordinates yet. Georeferencing is the final translation step from
> "detected shapes in an image" to "mapped structures on the ground."
```

---

## CELL 7 — Insert before the georeferencing / analytics step (add this cell even if the corresponding code cell is still to be written)

```markdown
## Georeferencing & Encroachment Analytics

**Objective:** Convert tile-space YOLO polygons into real-world, analysis-ready vector data.

### Process

1. **Reprojection:** Each tile's origin offset (its pixel position within the full raster)
   is combined with the source GeoTIFF's **affine transformation matrix** to convert every
   polygon vertex from normalized tile coordinates → full-raster pixel coordinates →
   geographic coordinates in **EPSG:4326**.
2. **Area calculation:** Each reprojected polygon's footprint area is computed in square
   meters (reprojecting to a local equal-area or UTM CRS for the area calculation itself,
   since EPSG:4326 is not equal-area).
3. **Buffer-zone intersection:** Structure polygons are spatially joined against the
   regulated riparian buffer geometry (derived from the OSM river centerlines) to flag
   which detected structures fall **inside** the restricted zone.
4. **Export:** Flagged structures, their footprint areas, and buffer-distance attributes are
   written to a single vector file: `encroaching_structures.geojson`.

> **Key Takeaway:** This is the step that turns a computer-vision output into a
> **planning-grade dataset** — every polygon is now a real-world object with a location,
> an area, and a compliance status, not just a shape in a 640×640 image tile.
```

---

## CELL 8 — Insert at the end of the notebook (Deliverables & Practical Value)

```markdown
## Key Deliverables & Practical Value

### Output

- **`encroaching_structures.geojson`** — a GIS-ready vector layer, directly loadable into
  **QGIS**, **ArcGIS**, or any web mapping stack (Leaflet, Mapbox GL, deck.gl), containing
  one feature per detected structure with geographic geometry, footprint area (m²), and
  buffer-encroachment status.

### Why the two-step design matters for real-world use

| Metric | Blob-based approach | This pipeline |
|---|---|---|
| Building count | Severely undercounted in dense clusters | Individually resolved structures |
| Footprint area | Meaningless (merged geometry) | Per-structure, in m² |
| GIS compatibility | Requires manual cleanup | Export-ready `.geojson` |

### Practical applications

- **Environmental compliance reporting** — exact counts and areas of unauthorized
  structures within the legal riparian setback, suitable for county environmental audits.
- **City planning & resource allocation** — quantifying informal settlement density and
  footprint growth over time (by re-running the pipeline on successive imagery dates).
- **Rapid dashboarding** — the `.geojson` output drops directly into existing web-based
  monitoring dashboards without additional format conversion.

> **Key Takeaway:** The pipeline's value isn't the detection alone — it's that every output
> record is **countable, measurable, and mappable**, which is the minimum bar for a dataset
> to be usable in a legal or administrative planning context.
```
