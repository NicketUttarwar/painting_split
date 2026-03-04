"""Tests for extraction.extractor: extract_paintings writes manifest and images."""
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.image import load_image, save_image
from extraction.extractor import extract_paintings


def _make_temp_image(tmp_path: Path, size=(100, 80)) -> Path:
    import numpy as np
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = (80, 80, 80)
    p = tmp_path / "source.png"
    save_image(img, p)
    return p


def test_extract_paintings_one_quad(tmp_path):
    img_path = _make_temp_image(tmp_path, (100, 80))
    quads = [[[10, 5], [90, 5], [90, 75], [10, 75]]]
    out_root = tmp_path / "out"
    result = extract_paintings(img_path, quads, out_root)
    assert result["source_id"]
    assert len(result["paintings"]) == 1
    assert result["paintings"][0]["filename"] == "painting_0.png"
    assert (out_root / result["source_id"] / "painting_0.png").is_file()
    assert (out_root / result["source_id"] / "manifest.json").is_file()
    assert (out_root / result["source_id"] / "overlay.png").is_file()


def test_extract_paintings_two_quads(tmp_path):
    img_path = _make_temp_image(tmp_path, (200, 100))
    quads = [
        [[0, 0], [95, 0], [95, 99], [0, 99]],
        [[105, 0], [199, 0], [199, 99], [105, 99]],
    ]
    out_root = tmp_path / "out"
    result = extract_paintings(img_path, quads, out_root)
    assert len(result["paintings"]) == 2
    assert (out_root / result["source_id"] / "painting_0.png").is_file()
    assert (out_root / result["source_id"] / "painting_1.png").is_file()
    import json
    manifest = json.loads((out_root / result["source_id"] / "manifest.json").read_text())
    assert manifest["source_width"] == 200
    assert manifest["source_height"] == 100
    assert len(manifest["paintings"]) == 2
