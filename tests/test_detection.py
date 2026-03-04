"""Tests for detection module (parse helpers; full detection requires API key)."""
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Test parsing and module structure without calling OpenAI
from detection.painting_detector import (
    _extract_json,
    _parse_sections_json,
    _parse_self_assessment,
    _corners_to_normalized,
)


def test_extract_json():
    assert _extract_json('{"a":1}') == {"a": 1}
    assert _extract_json('pre {"sections":[]} post') == {"sections": []}
    assert _extract_json('```json\n{"x":0}\n```') == {"x": 0}
    assert _extract_json("") is None
    assert _extract_json("no json here") is None


def test_parse_sections_json():
    text = '{"sections":[{"corners":[[0,0],[1000,0],[1000,1000],[0,1000]]}],"self_assessment":{}}'
    out = _parse_sections_json(text, 800, 600)
    assert len(out) == 1
    assert len(out[0]["corners"]) == 4
    assert out[0]["x"] == 0 and out[0]["y"] == 0
    assert out[0]["width"] == 800 and out[0]["height"] == 600


def test_parse_self_assessment():
    data = {"self_assessment": {"refinement_notes": " Moved corner 1 left."}}
    assert _parse_self_assessment(data) == "Moved corner 1 left."
    assert _parse_self_assessment({}) == ""
    assert _parse_self_assessment({"self_assessment": {}}) == ""


def test_corners_to_normalized():
    sections = [{"corners": [[0, 0], [100, 0], [100, 50], [0, 50]]}]
    out = _corners_to_normalized(sections, 100, 50)
    assert len(out) == 1
    assert out[0]["corners"][0] == [0, 0]
    assert out[0]["corners"][1] == [1000, 0]
    assert out[0]["corners"][2] == [1000, 1000]
    assert out[0]["corners"][3] == [0, 1000]
