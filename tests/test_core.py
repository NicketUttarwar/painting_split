"""Tests for core.image: load, crop, warp, safe_stem."""
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.image import (
    load_image,
    save_image,
    SectionBounds,
    safe_stem,
    quad_to_pts,
    quad_output_size,
    crop_section,
    warp_quad_to_rect,
    IMAGE_EXTENSIONS,
)


def test_safe_stem():
    assert safe_stem("foo.jpg") == "foo"
    assert safe_stem("a/b/c.png") == "c"
    assert " " not in safe_stem("hello world.x")
    assert len(safe_stem("a" * 100)) <= 80


def test_quad_to_pts():
    quad = [[0, 0], [10, 0], [10, 20], [0, 20]]
    pts = quad_to_pts(quad)
    assert pts.shape == (4, 2)
    np.testing.assert_array_almost_equal(pts[0], [0, 0])
    np.testing.assert_array_almost_equal(pts[1], [10, 0])
    quad_dict = [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": 0, "y": 1}]
    pts2 = quad_to_pts(quad_dict)
    assert pts2.shape == (4, 2)


def test_quad_output_size():
    pts = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
    w, h = quad_output_size(pts)
    assert w == 100
    assert h == 50


def test_crop_section_synthetic():
    img = np.zeros((60, 80, 3), dtype=np.uint8)
    img[:] = (100, 100, 100)
    bounds = SectionBounds(x=10, y=5, width=20, height=15)
    crop = crop_section(img, bounds, rotation_degrees=0)
    assert crop.shape == (15, 20, 3)
    crop90 = crop_section(img, bounds, rotation_degrees=90)
    assert crop90.shape == (20, 15, 3)


def test_warp_quad_to_rect_synthetic():
    img = np.zeros((50, 80, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)
    quad = [[10, 10], [70, 10], [70, 40], [10, 40]]
    warped = warp_quad_to_rect(img, quad)
    assert warped.shape[0] > 0 and warped.shape[1] > 0
    assert warped.shape[2] == 3


def test_section_bounds_from_dict():
    d = {"x": 1.0, "y": 2.0, "width": 10.0, "height": 20.0}
    b = SectionBounds.from_dict(d)
    assert b.x == 1 and b.y == 2 and b.width == 10 and b.height == 20
    assert b.to_dict() == d


def test_load_image_requires_file():
    with pytest.raises(Exception):
        load_image(Path("/nonexistent/image.png"))


def test_load_and_save_roundtrip(tmp_path):
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[2:8, 5:15] = (255, 0, 0)
    path = tmp_path / "test_core_roundtrip.png"
    save_image(img, path)
    assert path.is_file()
    loaded, mime = load_image(path)
    assert loaded is not None
    assert loaded.shape == img.shape
    np.testing.assert_array_equal(loaded, img)
