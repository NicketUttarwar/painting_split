"""
File/folder system for auto-detect runs: track each image, each run, and save
overlay (original + drawn quads) for comparison and edge-coverage assessment.
Layout: auto_detect_runs/<image_stem>/<run_id>/manifest.json, overlay.png, original copy optional.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from image_processor import load_image, safe_stem
except ImportError:
    load_image = None
    safe_stem = lambda n: Path(n).stem[:80].replace(" ", "_")


def get_runs_dir(base_dir: Path) -> Path:
    """Return the root directory for detection runs (create if needed). Uses data/runs/."""
    runs_dir = Path(base_dir) / "data" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def _run_id() -> str:
    """Generate a run id safe for folder names (ISO-ish, no colons)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


# Lazy numpy import for polylines
_np = None


def _ensure_np():
    global _np
    if _np is None:
        import numpy
        _np = numpy
    return _np


def _draw_quads_on_image(img, sections: list[dict[str, Any]], color_bgr=(0, 255, 0), thickness: int = 3):
    """Draw each section's corners as a closed polygon. Modifies img in place."""
    if cv2 is None or img is None:
        return
    np = _ensure_np()
    for s in sections:
        corners = s.get("corners")
        if not isinstance(corners, list) or len(corners) != 4:
            continue
        pts = []
        for c in corners:
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                pts.append([int(round(float(c[0]))), int(round(float(c[1])))])
            else:
                break
        if len(pts) == 4:
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts_arr], isClosed=True, color=color_bgr, thickness=thickness)
    return img


def save_run(
    image_path: Path,
    sections: list[dict[str, Any]],
    refinement_metadata: dict[str, Any],
    run_id: str | None = None,
    runs_base: Path | None = None,
    per_iteration: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Save one auto-detect run: manifest.json and overlay.png.
    Returns the run info (run_id, path, manifest_path, overlay_path) for the caller.
    """
    if load_image is None or cv2 is None:
        return {}
    _ensure_np()
    image_path = Path(image_path)
    if not image_path.is_file():
        return {}
    runs_dir = get_runs_dir(runs_base or image_path.resolve().parent.parent)
    stem = safe_stem(image_path.name)
    run_id = run_id or _run_id()
    run_dir = runs_dir / stem / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load image and draw quads for overlay
    img, _ = load_image(image_path)
    if img is not None and img.size > 0 and sections:
        overlay = img.copy()
        _draw_quads_on_image(overlay, sections)
        overlay_path = run_dir / "overlay.png"
        cv2.imwrite(str(overlay_path), overlay)

    # Copy original for comparison (so we have both original and overlay in same run folder)
    try:
        import shutil
        orig_copy = run_dir / ("original" + image_path.suffix.lower())
        if orig_copy.suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            orig_copy = run_dir / "original.png"
        shutil.copy2(image_path, orig_copy)
    except Exception:
        orig_copy = None

    manifest = {
        "image_path": str(image_path.name),
        "image_stem": stem,
        "run_id": run_id,
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "image_width": int(img.shape[1]) if img is not None else 0,
        "image_height": int(img.shape[0]) if img is not None else 0,
        "iterations": refinement_metadata.get("iterations", 0),
        "sections_count": len(sections),
        "refinement_notes": refinement_metadata.get("refinement_notes", ""),
        "refinement_used": refinement_metadata.get("refinement_used", False),
        "sections": sections,
        "per_iteration": per_iteration or [],
    }
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Placeholder for edge-coverage assessment (compare original vs overlay; fill via separate tool or API).
    assessment_path = run_dir / "assessment.json"
    assessment = {
        "status": "pending",
        "description": "Compare original and overlay to assess whether drawn edges cover canvas edges.",
        "edge_coverage_notes": None,
        "original_path": orig_copy.name if (orig_copy and orig_copy.is_file()) else None,
        "overlay_path": "overlay.png",
    }
    with open(assessment_path, "w") as f:
        json.dump(assessment, f, indent=2)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "overlay_path": str(run_dir / "overlay.png") if (img is not None and sections) else None,
        "original_path": str(orig_copy) if orig_copy and orig_copy.is_file() else None,
    }


def list_runs(runs_base: Path) -> list[dict[str, Any]]:
    """List all runs: by image stem, then by run_id. Each entry has run_id, stem, manifest summary."""
    runs_dir = get_runs_dir(runs_base)
    out = []
    if not runs_dir.is_dir():
        return out
    for stem_dir in sorted(runs_dir.iterdir()):
        if not stem_dir.is_dir():
            continue
        stem = stem_dir.name
        for run_dir in sorted(stem_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                with open(manifest_path) as f:
                    m = json.load(f)
            except Exception:
                continue
            out.append({
                "stem": stem,
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "timestamp_iso": m.get("timestamp_iso", ""),
                "sections_count": m.get("sections_count", 0),
                "iterations": m.get("iterations", 0),
                "image_path": m.get("image_path", ""),
            })
    return out


def list_runs_for_image(runs_base: Path, image_stem: str) -> list[dict[str, Any]]:
    """List runs for a given image stem (same image, different runs)."""
    runs_dir = get_runs_dir(runs_base)
    stem_dir = runs_dir / image_stem
    if not stem_dir.is_dir():
        return []
    out = []
    for run_dir in sorted(stem_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            with open(manifest_path) as f:
                m = json.load(f)
        except Exception:
            continue
        out.append({
            "stem": image_stem,
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "timestamp_iso": m.get("timestamp_iso", ""),
            "sections_count": m.get("sections_count", 0),
            "iterations": m.get("iterations", 0),
        })
    return out
