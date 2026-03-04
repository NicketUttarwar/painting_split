"""Tests for Flask API: inputs, extractions, config."""
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import app after path is set
from app import app, INPUTS_DIR, EXTRACTIONS_DIR


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_api_config(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    data = r.get_json()
    assert "openai_configured" in data
    assert "retrieve_paintings_available" in data


def test_api_list_inputs_empty(client):
    r = client.get("/api/inputs")
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, list)


def test_api_list_extractions(client):
    r = client.get("/api/extractions")
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, list)


def test_api_retrieve_paintings_missing_path(client):
    r = client.post("/api/retrieve-paintings", json={}, content_type="application/json")
    assert r.status_code == 400
    data = r.get_json()
    assert "error" in data


def test_api_retrieve_paintings_file_not_found(client):
    r = client.post(
        "/api/retrieve-paintings",
        json={"path": "inputs/nonexistent.png"},
        content_type="application/json",
    )
    assert r.status_code == 404
