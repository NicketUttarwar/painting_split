# Painting Split

Retrieve just the painting images from photos. Upload a photo (wall, floor, carpet, or cluttered scene); the app detects only the painting(s), excludes background and objects, and saves extracted images to `data/extractions/`.

**Requires Python 3.9+.**

## Install

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Dependencies**: numpy, opencv-python-headless, Pillow, scikit-image, flask, werkzeug, openai, pyyaml.

### OpenAI API key (for Retrieve paintings)

The **Retrieve paintings** flow uses OpenAI vision with **painting-only** prompts and iterative refinement (5 passes) to find painting boundaries and exclude wall, floor, carpet, and art supplies. To enable it:

1. Copy `secrets.example.yaml` to `secrets.yaml`.
2. Set `openai.api_key` in `secrets.yaml` (from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)).

## Web app

From the project root:

```bash
cd /path/to/painting_split && source .venv/bin/activate && python app.py
```

Or use `./run_web.sh`. Open the printed URL (e.g. `http://localhost:5001`) in your browser.

- **Inputs**: Upload photos to **data/inputs/** (drop zone or list). You can also pick from legacy **assets/**.
- **Retrieve paintings**: Select an image and click **Retrieve paintings**. The app detects only the painting(s), extracts them (perspective-corrected), and writes to **data/extractions/<source_id>/** (`painting_0.png`, `painting_1.png`, …, `manifest.json`, `overlay.png`). View and download extracted images on the page.
- **Export sections**: Draw sections (or use AI), then export to `outputs/<stem>/`. You get `manifest.json`, `section-0.png`, … `composite.png`, and `composite-recreated.png`. See [docs/MANIFEST.md](docs/MANIFEST.md) and [docs/FILE-NAMING.md](docs/FILE-NAMING.md).

## File layout

| Path | Purpose |
|------|--------|
| `data/inputs/` | Uploaded photos (input images). |
| `data/extractions/<source_id>/` | Extracted painting images, `manifest.json`, `overlay.png`. |
| `data/runs/<source_id>/<run_id>/` | Detection run (manifest, overlay, original copy). |
| `assets/` | Legacy input folder (still supported). |
| `outputs/<stem>/` | Section-split export: `manifest.json`, `section-0.png`, … `composite.png`, `composite-recreated.png`. |

- **Split outputs:** See [docs/FILE-NAMING.md](docs/FILE-NAMING.md) for naming and [docs/MANIFEST.md](docs/MANIFEST.md) for reading the manifest and sections.
- **Pipeline and API:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Scripts

**Retrieve paintings from one image** (detect + extract, or extract from JSON):

```bash
python scripts/retrieve_one.py <image_path>                  # detect then extract
python scripts/retrieve_one.py <image_path> <sections.json>  # extract from sections JSON
```

**Batch retrieval** (all images in data/inputs/):

```bash
python scripts/retrieve_paintings_batch.py [--inputs-dir DIR] [--extractions-dir DIR]
```

**Split from sections JSON** (rects or quads to output dir):

```bash
python scripts/split_from_sections.py <image_path> '<sections_json>' [output_dir]
```

Sections JSON: array of `{ "x", "y", "width", "height" }` (rects) or `{ "corners": [[x,y], ...] }` (4 corners per section, quads). Outputs go to `outputs/<stem>/` with `manifest.json`, `section-0.png`, … and composites (see [docs/FILE-NAMING.md](docs/FILE-NAMING.md)).

## API (summary)

- `GET/POST /api/inputs` — list / upload to data/inputs/
- `POST /api/retrieve-paintings` — run detection + extraction; returns `source_id`, `paintings[]`, `manifest_url`
- `POST /api/retrieve-paintings-stream` — same as NDJSON stream (progress + extraction_done)
- `GET /api/extractions` — list extractions; `GET /api/extractions/<source_id>/<filename>` — serve file
- `GET /api/runs`, `GET /api/runs/<stem>/<run_id>/...` — detection runs
- `POST /api/split` — split image into sections (rects or quads); writes to `outputs/<stem>/` (manifest, section images, composites)
- `GET /api/outputs` — list split outputs (each entry has path to `manifest.json` and folder)
- `POST /api/outputs/recreate` — recreate `composite.png` and `composite-recreated.png` from `manifest.json` (body: `{ "path": "folder/manifest.json" }`)
- `GET /api/outputs/<path>` — serve a file from outputs/ (e.g. `my_image/section-0.png`)

## Documentation

| Doc | Description |
|-----|--------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline, components, API. |
| [docs/FILE-NAMING.md](docs/FILE-NAMING.md) | Output file naming (`outputs/<stem>/`, `manifest.json`, `section-N.png`, composites). |
| [docs/MANIFEST.md](docs/MANIFEST.md) | How to read the split manifest and sections; schema, layout, and reconstruction. |

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Tests cover core, extraction, detection (parsing), and API. Some image_processor tests are skipped if no sample image is in `assets/`.
