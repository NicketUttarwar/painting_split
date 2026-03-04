"""
Image load/save, coordinate types (rect, quad), crop, perspective warp.
All coordinates: top-left origin, x right, y down, pixels.
"""
from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


@dataclass
class SectionBounds:
    """Axis-aligned rectangle in source image coordinates (pixels)."""
    x: float  # left
    y: float  # top
    width: float
    height: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SectionBounds:
        return cls(
            x=float(d["x"]),
            y=float(d["y"]),
            width=float(d["width"]),
            height=float(d["height"]),
        )


def load_image(path: Path) -> tuple[np.ndarray, str]:
    """Load image from path; return (BGR array, mimetype). Uses OpenCV, scikit-image, and PIL (EXIF)."""
    path = Path(path)
    if path.suffix.lower() in {".heic"}:
        pil = Image.open(path)
        pil = pil.convert("RGB")
        arr = np.array(pil)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr, "image/jpeg"
    img = cv2.imread(str(path))
    if img is not None:
        return img, "image/png" if path.suffix.lower() in {".png"} else "image/jpeg"
    try:
        from skimage import io as skio
        from skimage.util import img_as_ubyte
        rgb = skio.imread(str(path))
        if rgb is not None and rgb.size > 0:
            if rgb.ndim == 2:
                rgb = np.stack([rgb] * 3, axis=-1)
            elif rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            arr = img_as_ubyte(rgb)
            if arr.ndim >= 3 and arr.shape[-1] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr, "image/png" if path.suffix.lower() in {".png"} else "image/jpeg"
    except Exception:
        pass
    pil = Image.open(path)
    if hasattr(pil, "_getexif") and pil._getexif():
        from PIL import ImageOps
        pil = ImageOps.exif_transpose(pil)
    pil = pil.convert("RGB")
    arr = np.array(pil)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr, "image/jpeg"


def save_image(img: np.ndarray, path: Path) -> None:
    """Save BGR image to path. Parent dirs created if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def safe_stem(name: str) -> str:
    """File stem safe for filenames."""
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-.]", "_", stem)
    return stem[:80] or "image"


def quad_to_pts(quad: list[dict[str, float]] | list[list[float]]) -> np.ndarray:
    """Convert quad (4 points with x,y) to numpy array (4,2) float for cv2. Order: TL, TR, BR, BL."""
    pts = []
    for p in quad:
        if isinstance(p, dict):
            pts.append([float(p["x"]), float(p["y"])])
        else:
            pts.append([float(p[0]), float(p[1])])
    return np.array(pts, dtype=np.float32)


def quad_output_size(pts: np.ndarray) -> tuple[int, int]:
    """Compute natural rectangle size from quad (order TL, TR, BR, BL). Returns (width, height)."""
    top = np.linalg.norm(pts[1] - pts[0])
    bottom = np.linalg.norm(pts[2] - pts[3])
    left = np.linalg.norm(pts[3] - pts[0])
    right = np.linalg.norm(pts[2] - pts[1])
    out_w = max(1, int(round((top + bottom) / 2)))
    out_h = max(1, int(round((left + right) / 2)))
    return out_w, out_h


def crop_section(
    img: np.ndarray,
    bounds: SectionBounds,
    rotation_degrees: float = 0,
) -> np.ndarray:
    """Crop a section and optionally rotate. Bounds are clamped to image."""
    h, w = img.shape[:2]
    x1 = max(0, int(bounds.x))
    y1 = max(0, int(bounds.y))
    x2 = min(w, int(bounds.x + bounds.width))
    y2 = min(h, int(bounds.y + bounds.height))
    if x1 >= x2 or y1 >= y2:
        return np.zeros((1, 1, 3), dtype=img.dtype)
    crop = img[y1:y2, x1:x2].copy()
    if rotation_degrees:
        k = int(round(rotation_degrees / 90)) % 4
        if k:
            crop = np.rot90(crop, k=-k)
    return crop


def warp_quad_to_rect(
    img: np.ndarray,
    quad: list[dict[str, float]] | list[list[float]],
    out_w: int | None = None,
    out_h: int | None = None,
) -> np.ndarray:
    """
    Warp the quadrilateral region to a perfect rectangle (perspective correction).
    Quad order: top-left, top-right, bottom-right, bottom-left.
    Returns a BGR image of shape (out_h, out_w, 3). If out_w/out_h omitted, uses natural size from quad edges.
    """
    pts = quad_to_pts(quad)
    if pts.shape[0] != 4:
        raise ValueError("Quad must have exactly 4 points")
    h, w = img.shape[:2]
    pts = pts.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1e-2)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1e-2)
    if out_w is None or out_h is None:
        ow, oh = quad_output_size(pts)
        out_w = ow if out_w is None else out_w
        out_h = oh if out_h is None else out_h
    out_w = max(1, int(out_w))
    out_h = max(1, int(out_h))
    dst = np.array(
        [[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return warped


def extract_quad_region(
    img: np.ndarray,
    quad: list[dict[str, float]] | list[list[float]],
) -> np.ndarray:
    """
    Extract the region inside the quadrilateral. No warping: output is the
    axis-aligned bounding box of the quad with pixels outside the quad
    made transparent. Returns RGBA (4-channel) image.
    """
    pts = quad_to_pts(quad)
    if pts.shape[0] != 4:
        raise ValueError("Quad must have exactly 4 points")
    h, w = img.shape[:2]
    pts = pts.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1e-2)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1e-2)
    x_min = int(np.floor(pts[:, 0].min()))
    y_min = int(np.floor(pts[:, 1].min()))
    x_max = int(np.ceil(pts[:, 0].max())) + 1
    y_max = int(np.ceil(pts[:, 1].max())) + 1
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    if x_max <= x_min or y_max <= y_min:
        return np.zeros((1, 1, 4), dtype=np.uint8)
    crop = img[y_min:y_max, x_min:x_max]
    pts_rel = pts - np.array([x_min, y_min])
    pts_int = np.array(pts_rel, dtype=np.int32)
    mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts_int, 255)
    if len(crop.shape) == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return rgba
