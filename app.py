"""
Painting Split — manually outline multiple canvases in a photo and export them.
No AI. Upload/select image, draw or adjust canvas quads, export to outputs/.
"""
from __future__ import annotations

import json as json_module
from pathlib import Path
from typing import Optional
from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename

from core.image import load_image, safe_stem, IMAGE_EXTENSIONS
from image_processor import (
    split_image_from_rects,
    split_image_from_quads,
    SplitManifest,
    recreate_composite_from_manifest,
    COMPOSITE_FILENAME,
    COMPOSITE_RECREATED_FILENAME,
)

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
INPUTS_DIR = DATA_ROOT / "inputs"
EXTRACTIONS_DIR = DATA_ROOT / "extractions"
RUNS_DIR = DATA_ROOT / "runs"
# Legacy (backward compatibility)
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB


def allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in IMAGE_EXTENSIONS or ext == ".heic"


def _resolve_input_path(path_str: str) -> Optional[Path]:
    """Resolve path string to file under data/inputs/ or assets/. Returns None if not found."""
    path_str = (path_str or "").strip().lstrip("/")
    path_str = path_str.replace("inputs/", "").replace("assets/", "")
    for base in (INPUTS_DIR, ASSETS_DIR):
        path = base / path_str
        if path.is_file():
            try:
                path.resolve().relative_to(base.resolve())
                return path
            except ValueError:
                continue
    return None


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/assets")
def list_assets():
    """List image files in assets/ (and optionally uploads)."""
    out = []
    for path in sorted(ASSETS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in {*IMAGE_EXTENSIONS, ".heic"}:
            out.append({
                "name": path.name,
                "path": f"assets/{path.name}",
            })
    return jsonify(out)


@app.route("/api/assets/<path:filename>")
def get_asset(filename):
    """Serve an image from assets."""
    path = ASSETS_DIR / filename
    if not path.is_file() or path.resolve().parent != ASSETS_DIR.resolve():
        return jsonify({"error": "Not found"}), 404
    return send_file(path, mimetype="image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png")


@app.route("/api/upload", methods=["POST"])
def upload():
    """Accept a single image; save to assets/ and return path."""
    if "file" not in request.files and "image" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files.get("file") or request.files.get("image")
    if not f or f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "File type not allowed"}), 400
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    name = secure_filename(Path(f.filename).name)
    if not name:
        name = "upload.png"
    dest = ASSETS_DIR / name
    f.save(dest)
    return jsonify({"path": f"assets/{name}", "name": name})


@app.route("/api/split", methods=["POST"])
def api_split():
    """
    Split image into sections. Body: { path, sections, corrected_path?, ... }.
    If corrected_path is set, split from that image; otherwise from asset path.
    """
    data = request.get_json() or {}
    path_str = data.get("path") or data.get("asset")
    corrected_path_str = data.get("corrected_path")
    sections = data.get("sections") or []
    if not path_str and not corrected_path_str:
        return jsonify({"error": "Missing path, asset, or corrected_path"}), 400
    if corrected_path_str:
        corrected_path_str = corrected_path_str.replace("outputs/", "").lstrip("/")
        path = OUTPUTS_DIR / corrected_path_str
    else:
        path = _resolve_input_path(path_str)
    if not path or not path.is_file():
        return jsonify({"error": "File not found"}), 404
    extension = (data.get("extension") or "png").strip().lstrip(".") or "png"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if not sections:
            img, _ = load_image(path)
            if img is None or img.size == 0:
                return jsonify({"error": "Could not load image"}), 400
            h, w = img.shape[:2]
            sections = [{"x": 0, "y": 0, "width": float(w), "height": float(h), "rotation_degrees": 0.0}]

        use_quads = all(
            isinstance(s.get("corners"), list) and len(s.get("corners")) == 4
            for s in sections
        )
        if use_quads:
            quads = [s["corners"] for s in sections]
            manifest = split_image_from_quads(
                path,
                quads=quads,
                output_dir=OUTPUTS_DIR,
                extension=extension,
            )
        else:
            manifest = split_image_from_rects(
                path,
                rects=sections,
                output_dir=OUTPUTS_DIR,
                extension=extension,
            )
        out = manifest.to_dict()
        stem = safe_stem(manifest.source_filename)
        out["output_folder"] = stem
        if not out.get("composite_filename"):
            out["composite_filename"] = COMPOSITE_FILENAME
            out["composite_recreated_filename"] = COMPOSITE_RECREATED_FILENAME
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/outputs")
def list_outputs():
    """List manifests in outputs/ (including subfolders per source image)."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    manifests = []
    for path in sorted(OUTPUTS_DIR.rglob("manifest.json")):
        try:
            m = SplitManifest.load_json(path)
            # Path relative to outputs/ (e.g. "my_image/manifest.json")
            rel = path.relative_to(OUTPUTS_DIR)
            manifests.append({
                "path": str(rel).replace("\\", "/"),
                "filename": path.name,
                "folder": rel.parent.as_posix() if rel.parent != Path(".") else "",
                "source": m.source_filename,
                "source_width": m.source_width,
                "source_height": m.source_height,
                "sections_count": len(m.sections),
                "composite_filename": m.composite_filename,
                "composite_recreated_filename": m.composite_recreated_filename,
                "layout": m.layout,
            })
        except Exception:
            continue
    return jsonify(manifests)


@app.route("/api/outputs/recreate", methods=["POST"])
def api_recreate_composite():
    """
    Recreate the composite image from a saved manifest. Body: { "path": "folder/manifest.json" } (path relative to outputs/).
    Uses only manifest data so behavior matches how sections were saved.
    """
    data = request.get_json() or {}
    path_str = (data.get("path") or data.get("manifest_path") or "").strip().lstrip("/")
    if not path_str:
        return jsonify({"error": "Missing path or manifest_path"}), 400
    path = (OUTPUTS_DIR / path_str).resolve()
    root = OUTPUTS_DIR.resolve()
    if not path.is_file():
        return jsonify({"error": "Manifest not found"}), 404
    try:
        path.relative_to(root)
    except ValueError:
        return jsonify({"error": "Not found"}), 404
    try:
        recreate_composite_from_manifest(path)
        return jsonify({"ok": True, "path": path_str})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/outputs/<path:filename>")
def get_output(filename):
    """Serve a file from outputs/ (supports subfolders, e.g. my_image/manifest.json or my_image/section-0.png)."""
    path = (OUTPUTS_DIR / filename).resolve()
    root = OUTPUTS_DIR.resolve()
    if not path.is_file():
        return jsonify({"error": "Not found"}), 404
    # Ensure path is under OUTPUTS_DIR (no path traversal)
    try:
        path.relative_to(root)
    except ValueError:
        return jsonify({"error": "Not found"}), 404
    return send_file(path, as_attachment=True, download_name=path.name)


@app.route("/api/image-info", methods=["POST"])
def image_info():
    """Return width/height for an asset (for canvas setup)."""
    data = request.get_json() or {}
    path_str = data.get("path") or data.get("asset")
    if not path_str:
        return jsonify({"error": "Missing path"}), 400
    path_str = path_str.replace("assets/", "").lstrip("/")
    path = ASSETS_DIR / path_str
    if not path.is_file():
        path = _resolve_input_path(path_str)
    if not path or not path.is_file():
        return jsonify({"error": "File not found"}), 404
    try:
        img, _ = load_image(path)
        if img is None:
            return jsonify({"error": "Could not load image"}), 400
        h, w = img.shape[:2]
        return jsonify({"width": w, "height": h})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- New API: data/inputs, retrieve-paintings, extractions, runs ---

@app.route("/api/inputs")
def api_list_inputs():
    """List image files in data/inputs/."""
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for path in sorted(INPUTS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in {*IMAGE_EXTENSIONS, ".heic"}:
            out.append({"name": path.name, "path": f"inputs/{path.name}"})
    return jsonify(out)


@app.route("/api/inputs/<path:filename>")
def api_get_input(filename):
    """Serve an image from data/inputs/."""
    path = (INPUTS_DIR / filename).resolve()
    try:
        path.relative_to(INPUTS_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Not found"}), 404
    if not path.is_file():
        return jsonify({"error": "Not found"}), 404
    mimetype = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return send_file(path, mimetype=mimetype)


@app.route("/api/inputs", methods=["POST"])
def api_upload_input():
    """Upload a single image to data/inputs/."""
    if "file" not in request.files and "image" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files.get("file") or request.files.get("image")
    if not f or f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "File type not allowed"}), 400
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    name = secure_filename(Path(f.filename).name) or "upload.png"
    dest = INPUTS_DIR / name
    f.save(dest)
    return jsonify({"path": f"inputs/{name}", "name": name})


if __name__ == "__main__":
    PORT = 5001
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    url = f"http://localhost:{PORT}"
    print(f"\n  → Open in browser (copy link): {url}\n")
    app.run(host="0.0.0.0", port=PORT, debug=True)
