# File naming conventions

This project uses **folder-scoped** naming for split outputs: the folder name identifies the source image; files inside use fixed, simple names. No repeated stems in filenames.

---

## Outputs folder: `outputs/`

Section-split results live under **`outputs/<stem>/`**, where `<stem>` is a safe, filesystem-friendly version of the source image name (e.g. `photo.jpg` → `photo`, `My Image.heic` → `My_Image`). All files for one source image are in that one folder.

### Standard files in `outputs/<stem>/`

| File | Description |
|------|-------------|
| **manifest.json** | Split manifest: source dimensions, sections (bounds/corners, orientation, layout). Single source of truth for reconstruction. |
| **section-0.png** | First section image (index 0). |
| **section-1.png** | Second section, etc. |
| **section-&lt;N&gt;.&lt;ext&gt;** | Section image for index `N`. Extension is usually `png`; can be set at split time. |
| **composite.png** | Full image recreated by placing section images at their positions (OpenCV). Same size as source. |
| **composite-recreated.png** | Same layout as `composite.png`, but built from manifest + section images using Pillow (verifies reconstruction). |

### Naming rules

- **Manifest:** Always **`manifest.json`** in the folder. The app discovers split outputs by searching for `manifest.json` under `outputs/`.
- **Sections:** **`section-<i>.<ext>`** with zero-based index `i` (e.g. `section-0.png`, `section-1.png`). The manifest’s `sections[i].filename` matches this.
- **Composites:** **`composite.png`** and **`composite-recreated.png`**. Names are fixed; the manifest stores them in `composite_filename` and `composite_recreated_filename`.

No stem prefix: the folder name already identifies the source, so filenames stay short and consistent across all exports.

---

## Other directories

| Path | Naming |
|------|--------|
| **data/inputs/** | Uploaded or copied input images; names preserved (e.g. `photo.jpg`). |
| **data/extractions/&lt;source_id&gt;/** | `painting_0.png`, `painting_1.png`, …, `manifest.json`, `overlay.png`. Different schema from split manifest (see ARCHITECTURE.md). |
| **data/runs/&lt;stem&gt;/&lt;run_id&gt;/** | `manifest.json`, `overlay.png`, `original.<ext>`. Run manifests have a different schema (detection runs). |
| **assets/** | Legacy input folder; filenames as-is. |

---

## Constants in code

In **`image_processor.py`** the output names are defined as:

- `MANIFEST_FILENAME = "manifest.json"`
- `SECTION_FILENAME_TEMPLATE = "section-{i}.{ext}"`
- `COMPOSITE_FILENAME = "composite.png"`
- `COMPOSITE_RECREATED_FILENAME = "composite-recreated.png"`

Use these when building paths or when documenting behavior so code and docs stay in sync.

---

## Resolving paths

- **Manifest path:** `outputs/<stem>/manifest.json`. To list all splits: `outputs_dir.rglob("manifest.json")`.
- **Section path:** `outputs/<stem>/<section.filename>` (e.g. `outputs/photo/section-0.png`).
- **Composite path:** `outputs/<stem>/composite.png` and `outputs/<stem>/composite-recreated.png`.

When recreating from a manifest, the stem is the **parent directory name** of `manifest.json` (e.g. path `outputs/my_image/manifest.json` → stem `my_image`).
