"""
Image splitting with robust 2D coordinate metadata for reconstruction.
Uses core for load/save, rect/quad, crop, warp; adds SectionMetadata, SplitManifest, split_*.
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from core.image import (
    IMAGE_EXTENSIONS,
    SectionBounds,
    load_image,
    safe_stem,
    quad_to_pts,
    quad_output_size,
    crop_section,
    warp_quad_to_rect,
    extract_quad_region,
)

# Re-export for backward compatibility
_quad_to_pts = quad_to_pts
_quad_output_size = quad_output_size


@dataclass
class SectionMetadata:
    """One section: crop bounds and orientation for reconstruction."""
    index: int
    filename: str
    bounds: SectionBounds
    rotation_degrees: float  # 0, 90, 180, 270
    source_width: int
    source_height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "filename": self.filename,
            "bounds": self.bounds.to_dict(),
            "rotation_degrees": self.rotation_degrees,
            "source_width": self.source_width,
            "source_height": self.source_height,
            # Reconstruction: position in original image (top-left of axis-aligned box)
            "origin_x": self.bounds.x,
            "origin_y": self.bounds.y,
            "width_px": self.bounds.width,
            "height_px": self.bounds.height,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SectionMetadata:
        return cls(
            index=int(d["index"]),
            filename=str(d["filename"]),
            bounds=SectionBounds.from_dict(d["bounds"]),
            rotation_degrees=float(d.get("rotation_degrees", 0)),
            source_width=int(d["source_width"]),
            source_height=int(d["source_height"]),
        )


@dataclass
class SplitManifest:
    """Manifest for one source image: sections and coordinates."""
    source_filename: str
    source_width: int
    source_height: int
    sections: list[SectionMetadata] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_filename": self.source_filename,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "sections": [s.to_dict() for s in self.sections],
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> SplitManifest:
        with open(path) as f:
            d = json.load(f)
        return cls(
            source_filename=d["source_filename"],
            source_width=d["source_width"],
            source_height=d["source_height"],
            sections=[SectionMetadata.from_dict(s) for s in d["sections"]],
        )


def crop_image_to_region(
    image_path: Path,
    bounds: SectionBounds,
    output_path: Path,
) -> None:
    """Crop image to the given bounds and save to output_path. No rotation."""
    img, _ = load_image(image_path)
    if img is None or img.size == 0:
        raise ValueError(f"Could not load image: {image_path}")
    crop = crop_section(img, bounds, rotation_degrees=0)
    if crop.size == 0:
        raise ValueError("Crop region is empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), crop)


def canvas_crop_by_quad(
    image_path: Path,
    quad: list[dict[str, float]] | list[list[float]],
    output_path: Path,
) -> None:
    """
    Create canvas-only image: crop to bbox of quad and mask so only pixels
    inside the quad are opaque. Saves RGBA PNG. No warping.
    """
    img, _ = load_image(image_path)
    if img is None or img.size == 0:
        raise ValueError(f"Could not load image: {image_path}")
    rgba = extract_quad_region(img, quad)
    if rgba.size == 0:
        raise ValueError("Quad region is empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), rgba)


def _write_composite_image_rects(
    source_img: np.ndarray,
    sections: list[SectionMetadata],
    section_dir: Path,
    stem: str,
    extension: str,
) -> None:
    """Recreate one full image from rect sections: paste each crop at its origin. Transparent background (RGBA). Writes {stem}_composite.png."""
    source_h, source_w = source_img.shape[:2]
    composite = np.zeros((source_h, source_w, 4), dtype=np.uint8)
    composite[:, :, 3] = 0
    for sec in sections:
        path = section_dir / sec.filename
        if not path.is_file():
            continue
        crop, _ = load_image(path)
        if crop is None or crop.size == 0:
            continue
        ch, cw = crop.shape[:2]
        x = max(0, int(sec.bounds.x))
        y = max(0, int(sec.bounds.y))
        x2 = min(source_w, x + cw)
        y2 = min(source_h, y + ch)
        if x2 <= x or y2 <= y:
            continue
        w_avail, h_avail = x2 - x, y2 - y
        crop = crop[:h_avail, :w_avail]
        if crop.shape[2] == 3:
            composite[y:y + crop.shape[0], x:x + crop.shape[1], :3] = crop
            composite[y:y + crop.shape[0], x:x + crop.shape[1], 3] = 255
        else:
            composite[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
    out_path = section_dir / f"{stem}_composite.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), composite)


def _write_composite_image(
    source_w: int,
    source_h: int,
    quads: list[list[list[float]]],
    section_dir: Path,
    stem: str,
    extension: str,
) -> None:
    """
    Recreate one full image by placing each section cutout at its original quad position.
    Transparent background (RGBA). Slight mask dilation so sections connect with no visible gaps.
    Writes {stem}_composite.png in the same subfolder.
    """
    ext = extension.lstrip(".")
    composite = np.zeros((source_h, source_w, 4), dtype=np.uint8)
    composite[:, :, 3] = 0
    for i, quad in enumerate(quads):
        if len(quad) != 4:
            continue
        section_path = section_dir / f"{stem}_section_{i}.{ext}"
        if not section_path.is_file():
            continue
        warped, _ = load_image(section_path)
        if warped is None or warped.size == 0:
            continue
        hw, ww = warped.shape[:2]
        pts_src = np.ascontiguousarray(
            np.array([[0.0, 0.0], [float(ww), 0.0], [float(ww), float(hw)], [0.0, float(hw)]], dtype=np.float32)
        )
        pts_dst = _quad_to_pts(quad)
        pts_dst = np.clip(
            pts_dst,
            0.0,
            [float(source_w) - 0.01, float(source_h) - 0.01],
        ).astype(np.float32)
        pts_dst = np.ascontiguousarray(pts_dst)
        if pts_src.shape != (4, 2) or pts_dst.shape != (4, 2):
            continue
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped_onto = cv2.warpPerspective(warped, M, (source_w, source_h), flags=cv2.INTER_LINEAR)
        mask = np.zeros((source_h, source_w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts_dst.astype(np.int32), 255)
        mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8))
        for c in range(3):
            composite[:, :, c] = np.where(mask > 0, warped_onto[:, :, c], composite[:, :, c])
        composite[:, :, 3] = np.maximum(composite[:, :, 3], np.where(mask > 0, 255, 0).astype(np.uint8))
    out_path = section_dir / f"{stem}_composite.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), composite)


def split_image_from_rects(
    image_path: Path,
    rects: list[dict[str, float]],
    output_dir: Path,
    extension: str = "png",
) -> SplitManifest:
    """
    Split image into sections using given rectangles. No rotation; all in source image orientation.
    rects: list of {x, y, width, height} in source image pixels.
    Writes section images and a manifest JSON into a subfolder named by the source image stem.
    """
    img, _ = load_image(image_path)
    if img is None or img.size == 0:
        raise ValueError(f"Could not load image: {image_path}")
    h, w = img.shape[:2]
    stem = safe_stem(image_path.name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    section_dir = output_dir / stem
    section_dir.mkdir(parents=True, exist_ok=True)

    sections: list[SectionMetadata] = []
    for i, r in enumerate(rects):
        bounds = SectionBounds(
            x=float(r["x"]),
            y=float(r["y"]),
            width=float(r["width"]),
            height=float(r["height"]),
        )
        crop = crop_section(img, bounds, rotation_degrees=0)
        if crop.size == 0:
            continue
        filename = f"{stem}_section_{i}.{extension.lstrip('.')}"
        out_path = section_dir / filename
        cv2.imwrite(str(out_path), crop)
        sections.append(
            SectionMetadata(
                index=i,
                filename=filename,
                bounds=bounds,
                rotation_degrees=0,
                source_width=w,
                source_height=h,
            )
        )

    manifest = SplitManifest(
        source_filename=image_path.name,
        source_width=w,
        source_height=h,
        sections=sections,
    )
    manifest_path = section_dir / f"{stem}_manifest.json"
    manifest.save_json(manifest_path)
    with open(manifest_path) as f:
        d = json.load(f)
    centroids = []
    for idx, sec in enumerate(d.get("sections", [])):
        bx = sec.get("bounds", {})
        x = float(bx.get("x", 0))
        y = float(bx.get("y", 0))
        ww = float(bx.get("width", 0))
        hh = float(bx.get("height", 0))
        cx, cy = x + ww / 2, y + hh / 2
        sec["centroid_x"] = cx
        sec["centroid_y"] = cy
        centroids.append((cx, cy))
    order_indices = list(range(len(centroids)))
    order_indices.sort(key=lambda i: (centroids[i][1], centroids[i][0]))
    d["layout"] = {
        "reading_order": order_indices,
        "description": "Section indices in top-to-bottom, left-to-right order by centroid.",
    }
    for rank, idx in enumerate(order_indices):
        d["sections"][idx]["position_rank"] = rank
    tol = min(w, h) * 0.05
    section_relations = []
    for i in range(len(centroids)):
        cx_i, cy_i = centroids[i]
        left_of = [j for j in range(len(centroids)) if j != i and centroids[j][0] < cx_i - tol and abs(centroids[j][1] - cy_i) < min(w, h) * 0.3]
        right_of = [j for j in range(len(centroids)) if j != i and centroids[j][0] > cx_i + tol and abs(centroids[j][1] - cy_i) < min(w, h) * 0.3]
        above = [j for j in range(len(centroids)) if j != i and centroids[j][1] < cy_i - tol]
        below = [j for j in range(len(centroids)) if j != i and centroids[j][1] > cy_i + tol]
        section_relations.append({"left_of": left_of, "right_of": right_of, "above": above, "below": below})
    d["layout"]["section_relations"] = section_relations
    d["composite_filename"] = f"{stem}_composite.png"
    with open(manifest_path, "w") as f:
        json.dump(d, f, indent=2)
    # Final recreated painting: all extracted sections composited into one PNG in the same subfolder
    _write_composite_image_rects(img, sections, section_dir, stem, extension)
    return manifest


def split_image_from_quads(
    image_path: Path,
    quads: list[list[dict[str, float]] | list[list[list[float]]]],
    output_dir: Path,
    extension: str = "png",
) -> SplitManifest:
    """
    Split image by perspective-correcting each quadrilateral to a rectangle.
    Each quad (4 corners) is warped so the canvas becomes a perfect rectangle.
    Writes into a subfolder named by the source image stem.
    quads: list of 4-point lists, each point {x,y} or [x,y]. Order: TL, TR, BR, BL.
    """
    img, _ = load_image(image_path)
    if img is None or img.size == 0:
        raise ValueError(f"Could not load image: {image_path}")
    h, w = img.shape[:2]
    stem = safe_stem(image_path.name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # One subfolder per source image so outputs are clustered by source
    section_dir = output_dir / stem
    section_dir.mkdir(parents=True, exist_ok=True)

    sections: list[SectionMetadata] = []
    for i, quad in enumerate(quads):
        if len(quad) != 4:
            continue
        try:
            warped = warp_quad_to_rect(img, quad)
        except Exception:
            continue
        if warped.size == 0:
            continue
        filename = f"{stem}_section_{i}.{extension.lstrip('.')}"
        out_path = section_dir / filename
        cv2.imwrite(str(out_path), warped)

        pts = _quad_to_pts(quad)
        x_min = float(pts[:, 0].min())
        y_min = float(pts[:, 1].min())
        x_max = float(pts[:, 0].max())
        y_max = float(pts[:, 1].max())
        bounds = SectionBounds(
            x=x_min, y=y_min,
            width=x_max - x_min,
            height=y_max - y_min,
        )
        corners = [[float(pts[j, 0]), float(pts[j, 1])] for j in range(4)]
        section_dict: dict[str, Any] = {
            "index": i,
            "filename": filename,
            "bounds": bounds.to_dict(),
            "rotation_degrees": 0.0,
            "source_width": w,
            "source_height": h,
            "corners": corners,
        }
        sections.append(SectionMetadata(
            index=section_dict["index"],
            filename=section_dict["filename"],
            bounds=bounds,
            rotation_degrees=0.0,
            source_width=w,
            source_height=h,
        ))

    manifest = SplitManifest(
        source_filename=image_path.name,
        source_width=w,
        source_height=h,
        sections=sections,
    )
    manifest_path = section_dir / f"{stem}_manifest.json"
    manifest.save_json(manifest_path)
    with open(manifest_path) as f:
        d = json.load(f)

    # Add corners, centroids, and layout (orientation/location relative to each other)
    centroids: list[tuple[float, float]] = []
    for idx, sec in enumerate(d.get("sections", [])):
        if idx < len(quads) and len(quads[idx]) == 4:
            pts = _quad_to_pts(quads[idx])
            sec["corners"] = [[float(pts[j, 0]), float(pts[j, 1])] for j in range(4)]
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            sec["centroid_x"] = cx
            sec["centroid_y"] = cy
            centroids.append((cx, cy))
        else:
            b = sec.get("bounds", {})
            cx = b.get("x", 0) + b.get("width", 0) / 2
            cy = b.get("y", 0) + b.get("height", 0) / 2
            sec["centroid_x"] = cx
            sec["centroid_y"] = cy
            centroids.append((cx, cy))

    # Reading order: top-to-bottom, then left-to-right (by centroid)
    order_indices = list(range(len(centroids)))
    order_indices.sort(key=lambda i: (centroids[i][1], centroids[i][0]))
    d["layout"] = {
        "reading_order": order_indices,
        "description": "Section indices in top-to-bottom, left-to-right order by centroid.",
    }
    for rank, idx in enumerate(order_indices):
        d["sections"][idx]["position_rank"] = rank
    # Relative relations: for each section, which others are left/right/above/below (by centroid)
    tol = min(w, h) * 0.05
    section_relations = []
    for i in range(len(centroids)):
        cx_i, cy_i = centroids[i]
        left_of = [j for j in range(len(centroids)) if j != i and centroids[j][0] < cx_i - tol and abs(centroids[j][1] - cy_i) < min(w, h) * 0.3]
        right_of = [j for j in range(len(centroids)) if j != i and centroids[j][0] > cx_i + tol and abs(centroids[j][1] - cy_i) < min(w, h) * 0.3]
        above = [j for j in range(len(centroids)) if j != i and centroids[j][1] < cy_i - tol]
        below = [j for j in range(len(centroids)) if j != i and centroids[j][1] > cy_i + tol]
        section_relations.append({"left_of": left_of, "right_of": right_of, "above": above, "below": below})
    d["layout"]["section_relations"] = section_relations
    d["composite_filename"] = f"{stem}_composite.png"
    with open(manifest_path, "w") as f:
        json.dump(d, f, indent=2)

    # Final recreated painting: all extracted sections composited into one PNG in the same subfolder
    _write_composite_image(img.shape[1], img.shape[0], quads, section_dir, stem, extension)

    return manifest

