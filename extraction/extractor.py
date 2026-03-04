"""
Extract painting images from source photo using quads.
Writes to data/extractions/<source_id>/: painting_0.png, painting_1.png, ..., manifest.json, overlay.png.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.image import (
    load_image,
    save_image,
    safe_stem,
    quad_to_pts,
    warp_quad_to_rect,
)


def extract_paintings(
    image_path: Path,
    quads: list[list[list[float]]] | list[dict[str, Any]],
    output_root: Path,
    source_id: str | None = None,
    extension: str = "png",
) -> dict[str, Any]:
    """
    Warp each quad to a rectangle and save as painting_0.png, painting_1.png, ...
    Also writes manifest.json and overlay.png. Uses output_root (e.g. data/extractions).
    Returns manifest dict with source_path, source_width, source_height, paintings[].
    """
    image_path = Path(image_path)
    output_root = Path(output_root)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img, _ = load_image(image_path)
    if img is None or img.size == 0:
        raise ValueError(f"Could not load image: {image_path}")
    h, w = img.shape[:2]

    source_id = source_id or safe_stem(image_path.name)
    out_dir = output_root / source_id
    out_dir.mkdir(parents=True, exist_ok=True)

    paintings: list[dict[str, Any]] = []
    overlay = img.copy()

    for i, quad in enumerate(quads):
        if not isinstance(quad, (list, tuple)) or len(quad) != 4:
            continue
        try:
            warped = warp_quad_to_rect(img, quad)
        except Exception:
            continue
        if warped.size == 0:
            continue

        filename = f"painting_{i}.{extension.lstrip('.')}"
        out_path = out_dir / filename
        save_image(warped, out_path)

        pts = quad_to_pts(quad)
        pts = np.clip(pts, 0, None)
        pts[:, 0] = np.minimum(pts[:, 0], w - 1e-2)
        pts[:, 1] = np.minimum(pts[:, 1], h - 1e-2)
        x_min = float(pts[:, 0].min())
        y_min = float(pts[:, 1].min())
        x_max = float(pts[:, 0].max())
        y_max = float(pts[:, 1].max())
        corners = [[float(pts[j, 0]), float(pts[j, 1])] for j in range(4)]

        paintings.append({
            "index": i,
            "filename": filename,
            "bounds": {"x": x_min, "y": y_min, "width": x_max - x_min, "height": y_max - y_min},
            "corners": corners,
            "rotation_degrees": 0,
        })

        # Draw quad on overlay
        pts_int = np.array(pts, dtype=np.int32)
        cv2.polylines(overlay, [pts_int], isClosed=True, color=(0, 255, 0), thickness=3)

    manifest = {
        "source_path": image_path.name,
        "source_width": w,
        "source_height": h,
        "paintings": paintings,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    overlay_path = out_dir / "overlay.png"
    save_image(overlay, overlay_path)

    return {
        **manifest,
        "source_id": source_id,
        "output_dir": str(out_dir),
    }
