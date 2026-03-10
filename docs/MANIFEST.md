# Split manifest and sections

This document describes the **split manifest** (section-split exports under `outputs/`): file format, how to read it, and how sections relate to the source image and to each other. For extraction manifests (`data/extractions/`), see [ARCHITECTURE.md](ARCHITECTURE.md).

## Overview

When you **split an image into sections** (rectangles or quadrilaterals), the app writes:

- **One folder per source image** under `outputs/<stem>/` (see [FILE-NAMING.md](FILE-NAMING.md)).
- **manifest.json** — single source of truth: source image info, every section’s geometry, orientation, and layout (reading order, relations).
- **Section images** — `section-0.png`, `section-1.png`, … in the orientation the manifest expects.
- **composite.png** — sections placed back (OpenCV).
- **composite-recreated.png** — same layout, built from manifest + section images using Pillow (verifies reconstruction).

You can **recreate the composite from the manifest alone** (no need for the original photo): load `manifest.json`, load each section image by `filename`, then place them using `bounds` (rect) or `corners` (quad).

---

## Manifest file: `manifest.json`

Location: `outputs/<stem>/manifest.json`. JSON object with the following top-level keys.

### Top-level keys

| Key | Type | Description |
|-----|------|--------------|
| `source_filename` | string | Original image file name (e.g. `photo.jpg`). |
| `source_width` | number | Width of the source image in pixels. |
| `source_height` | number | Height of the source image in pixels. |
| `sections` | array | One object per section (see [Section object](#section-object) below). |
| `composite_filename` | string | Name of the composite image file in this folder (`composite.png`). |
| `composite_recreated_filename` | string | Name of the Pillow-built composite (`composite-recreated.png`). |
| `layout` | object | Reading order and section relations (see [Layout](#layout)). |

### Section object

Each element of `sections` describes one section: where it came from, how it was cut, and how to place it back.

| Key | Type | Description |
|-----|------|--------------|
| `index` | number | Zero-based section index (matches `section-<index>.<ext>`). |
| `filename` | string | Section image file name in this folder (e.g. `section-0.png`). |
| `bounds` | object | Axis-aligned box in source image: `x`, `y`, `width`, `height` (pixels). |
| `rotation_degrees` | number | 0, 90, 180, or 270 (currently always 0). |
| `source_width` | number | Source image width (repeated for convenience). |
| `source_height` | number | Source image height (repeated for convenience). |
| `section_type` | string | `"rect"` or `"quad"` — how the section was extracted and how to place it. |
| `output_width_px` | number | Width of the saved section image in pixels. |
| `output_height_px` | number | Height of the saved section image in pixels. |
| `origin_x`, `origin_y` | number | Same as `bounds.x`, `bounds.y`. |
| `width_px`, `height_px` | number | Same as `bounds.width`, `bounds.height`. |
| `corners` | array | **Quad only.** Four points `[x,y]` in source image: TL, TR, BR, BL. Omitted for rect. |
| `position_rank` | number | Reading-order rank (0 = first). |
| `centroid_x`, `centroid_y` | number | Center of the section in source image (for layout). |

#### Rect vs quad

- **`section_type: "rect"`**  
  Section is an axis-aligned crop. To place it back: paste the section image at `(bounds.x, bounds.y)`. No rotation.

- **`section_type: "quad"`**  
  Section was perspective-warped from a quadrilateral. The saved section image is an upright rectangle (top-left, top-right, bottom-right, bottom-left). To place it back: use a perspective transform that maps the section rectangle onto the four `corners` in the composite (source) space. Corner order: **TL, TR, BR, BL** (same as in `core/image.py` and detection).

### Layout

`layout` describes order and spatial relations between sections (for UI or scripts).

| Key | Type | Description |
|-----|------|--------------|
| `reading_order` | array | Section indices in top-to-bottom, left-to-right order (by centroid). |
| `description` | string | Short explanation of `reading_order`. |
| `section_relations` | array | One object per section (same index as `sections`). Each has: |
| `section_relations[i].left_of` | array | Section indices that are to the left of section `i`. |
| `section_relations[i].right_of` | array | Section indices to the right. |
| `section_relations[i].above` | array | Section indices above. |
| `section_relations[i].below` | array | Section indices below. |

Relations are derived from centroids with a small tolerance so you can build grids or ordered lists.

---

## How to read the manifest

### Loading in Python

```python
from pathlib import Path
from image_processor import SplitManifest

path = Path("outputs/my_image/manifest.json")
manifest = SplitManifest.load_json(path)

print(manifest.source_filename)   # e.g. "photo.jpg"
print(manifest.source_width, manifest.source_height)
print(len(manifest.sections))

for s in manifest.sections:
    print(s.index, s.filename, s.section_type, s.bounds.x, s.bounds.y)
    if s.corners:
        print("  corners:", s.corners)
```

### Recreating the composite

Use the built-in function so placement matches the manifest:

```python
from image_processor import recreate_composite_from_manifest

recreate_composite_from_manifest(Path("outputs/my_image/manifest.json"))
# Writes composite.png and composite-recreated.png in the same folder.
```

### Manual placement (rect)

For each section with `section_type == "rect"`:

1. Load the image from `outputs/<stem>/<section.filename>`.
2. Paste it at `(bounds.x, bounds.y)` on a canvas of size `(source_width, source_height)`.
3. Clip to canvas if the crop extends past the edges.

### Manual placement (quad)

For each section with `section_type == "quad"` and four `corners`:

1. Load the section image (upright rectangle).
2. Build a perspective transform from the section rectangle `[0,0], [W,0], [W,H], [0,H]` to `corners` in composite space.
3. Warp the section onto the composite canvas and composite (e.g. with a mask so only the quad is opaque).

(Implementation: see `_write_composite_image` and `_write_composite_recreated` in `image_processor.py`.)

---

## Section image orientation

Section files are saved in the orientation the manifest assumes:

- **Rect:** Axis-aligned crop; no rotation. Section image pixels correspond 1:1 to the region `bounds` in the source image.
- **Quad:** Perspective-warped to an upright rectangle. Corner order in the section image: top-left → top-right → bottom-right → bottom-left. When recreating, the same corner order in `corners` places the section back correctly.

Keeping this consistent avoids mismatches when recreating the full image from sections.

---

## Example manifest (minimal)

```json
{
  "source_filename": "wall.jpg",
  "source_width": 1200,
  "source_height": 800,
  "sections": [
    {
      "index": 0,
      "filename": "section-0.png",
      "bounds": { "x": 0, "y": 0, "width": 600, "height": 400 },
      "rotation_degrees": 0,
      "source_width": 1200,
      "source_height": 800,
      "section_type": "rect",
      "output_width_px": 600,
      "output_height_px": 400,
      "position_rank": 0,
      "centroid_x": 300,
      "centroid_y": 200
    }
  ],
  "composite_filename": "composite.png",
  "composite_recreated_filename": "composite-recreated.png",
  "layout": {
    "reading_order": [0],
    "description": "Section indices in top-to-bottom, left-to-right order by centroid.",
    "section_relations": [{ "left_of": [], "right_of": [], "above": [], "below": [] }]
  }
}
```

Quad sections will also include `"corners": [[x,y], [x,y], [x,y], [x,y]]` (TL, TR, BR, BL) in source image coordinates.
