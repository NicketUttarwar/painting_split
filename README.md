# Painting Split

Detect painting edges in camera-roll photos, correct perspective, and split multi-canvas images into the smallest usable sections.

**Requires Python 3.13.** The project config explicitly uses Python 3.13; see [Config](#config) below.

## Install

From the project root:

```bash
# Use Python 3.13 on your machine (e.g. from Homebrew or python.org)
python3.13 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Or install from the dependency list only (no editable package):

```bash
pip install -r requirements.txt
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

**Dependencies** (in `pyproject.toml` and `requirements.txt`): numpy, opencv-python, Pillow, scikit-image, scipy, matplotlib. The `requirements.txt` file is aligned with `pyproject.toml` so either install method reproduces the same environment.

## Run

```bash
painting-split <input_dir> <output_dir>
```

Example: process every image in `assets` and write sections to `output`:

```bash
painting-split assets output
```

- **input_dir**: Folder of images (e.g. `assets/` or camera roll exports). **Every image** in this directory is processed; by default subdirectories are included (use `--no-recursive` for top-level only). Supports JPEG, PNG, BMP, WebP, TIFF.
- **output_dir**: Where to save cropped, perspective-corrected images (and split sections when internal boundaries are detected).

Output files are named `{original_stem}_section_0.png`, `_section_1.png`, etc.

### Options

| Option | Description |
|--------|-------------|
| `--no-multi-canvas` | Disable splitting into multiple canvases; output one image per painting only. |
| `--no-deskew` | Disable deskew step; do not correct small rotation of the canvas. |
| `--max-deskew-degrees DEGREES` | Maximum rotation to apply when deskewing (default: 8.0). |
| `--min-line-length-ratio RATIO` | Minimum line length as fraction of image for deskew (default: 0.12). |
| `--min-area RATIO` | Minimum contour area as fraction of image (default: 0.05). |
| `--canny-low`, `--canny-high` | Canny edge detection thresholds (defaults: 50, 150). |
| `--padding N` | Pixels of padding around each corrected image (default: 0). |
| `--extension {png,jpg,jpeg}` | Output format (default: png). |
| `--no-recursive` | Only process images in the top-level of input_dir, not in subdirectories. |
| `--visualize` | Save debug images (deskewed canvas) to `output_dir/debug/<stem>_debug.png`. |

Example:

```bash
painting-split ./photos ./output --min-area 0.08 --padding 2
```

## Config

The repo includes a **config file** that mentions the Python version and interpreter:

- **`config.toml`**  
  - `python_version = "3.13"`  
  - `python_interpreter_path` — Optional. Set to the path of Python 3.13 on your machine (e.g. `/opt/homebrew/bin/python3.13` or `C:\Python313\python.exe`) if you want tooling or scripts to use that interpreter explicitly. Leave empty to use whichever `python3.13` is on your PATH.

Create the virtual environment with that interpreter:

```bash
# If you set python_interpreter_path in config.toml, use it:
/path/to/python3.13 -m venv .venv
# Otherwise:
python3.13 -m venv .venv
```

You can add a `[tuning]` section to `config.toml` with `min_area_ratio`, `canny_low`, `canny_high`, `padding`, `max_deskew_degrees`, or `min_line_length_ratio` to change default CLI values (CLI flags override).

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## How it works

1. **Load** every image from the input directory (recursively by default). If OpenCV cannot read a file, Pillow is used with EXIF orientation applied so camera-roll photos appear upright.
2. **Detect** the painting boundary (largest quad from edges/contours).
3. **Perspective-correct** so the painting is front-facing (aspect ratio preserved).
4. **Deskew**: The corrected canvas is analyzed for dominant line angles; a small rotation (clamped by `--max-deskew-degrees`) is applied so that horizontal/vertical lines align, improving later section detection.
5. **Section split**: On the deskewed canvas, the pipeline finds **strong straight lines** in any direction and builds sections. If internal boundaries are found, the image is split into the smallest sections; otherwise the single corrected image is saved.
6. **Save** each section as `{source_stem}_section_{i}.png` (or chosen extension).

If no painting quad is found in an image, it is skipped and a warning is printed.
