"""Tests for image_processor: load, crop, split (no auto-detect; detection is OpenAI vision)."""
from pathlib import Path
from typing import Optional

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_processor import (
    split_image_from_rects,
    load_image,
    SectionBounds,
    crop_section,
)


ASSETS = Path(__file__).resolve().parent.parent / "assets"
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"

# Sample image: prefer 4panel, else the recursive-test asset
def _sample_image_path() -> Optional[Path]:
    for name in ("2017-10-22_18-41-22_438-4panel.png", "2017-10-22_18-41-22_438-f99608c9-4707-404d-a3a0-5a2b84bde722.png"):
        p = ASSETS / name
        if p.is_file():
            return p
    return None


def _four_quadrant_rects(w: int, h: int) -> list[dict]:
    """Return 4 axis-aligned rects splitting image into quadrants."""
    hw, hh = w // 2, h // 2
    return [
        {"x": 0, "y": 0, "width": float(hw), "height": float(hh), "rotation_degrees": 0},
        {"x": float(hw), "y": 0, "width": float(w - hw), "height": float(hh), "rotation_degrees": 0},
        {"x": 0, "y": float(hh), "width": float(hw), "height": float(h - hh), "rotation_degrees": 0},
        {"x": float(hw), "y": float(hh), "width": float(w - hw), "height": float(h - hh), "rotation_degrees": 0},
    ]


def test_split_produces_files_and_manifest():
    """Split an image with 4 quadrant rects and check section files and manifest."""
    path = _sample_image_path()
    if path is None:
        pytest.skip("Sample image not in assets/")
    img, _ = load_image(path)
    assert img is not None
    h, w = img.shape[:2]
    rects = _four_quadrant_rects(w, h)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    manifest = split_image_from_rects(path, rects=rects, output_dir=OUTPUTS)
    assert len(manifest.sections) == 4
    from image_processor import safe_stem
    stem = safe_stem(path.name)
    section_dir = OUTPUTS / stem
    for i, s in enumerate(manifest.sections):
        assert s.filename == f"section-{i}.png"
        assert (section_dir / s.filename).is_file()
    assert (section_dir / "manifest.json").is_file()


def test_load_image_returns_bgr_array():
    """load_image returns non-empty BGR array and mimetype."""
    path = _sample_image_path()
    if path is None:
        pytest.skip("Sample image not in assets/")
    img, mime = load_image(path)
    assert img is not None
    assert img.size > 0
    assert len(img.shape) == 3 and img.shape[2] == 3
    assert "image" in mime


def test_crop_section_bounds():
    """crop_section respects bounds and returns non-empty crop."""
    path = _sample_image_path()
    if path is None:
        pytest.skip("Sample image not in assets/")
    img, _ = load_image(path)
    h, w = img.shape[:2]
    bounds = SectionBounds(x=10, y=20, width=min(100, w - 10), height=min(80, h - 20))
    crop = crop_section(img, bounds, rotation_degrees=0)
    assert crop.size > 0
    assert crop.shape[0] <= int(bounds.height) + 1 and crop.shape[1] <= int(bounds.width) + 1
