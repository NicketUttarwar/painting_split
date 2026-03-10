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
from PIL import Image as PILImage

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

# Output file naming (folder-scoped: all under outputs/<stem>/)
MANIFEST_FILENAME = "manifest.json"
SECTION_FILENAME_TEMPLATE = "section-{i}.{ext}"  # section-0.png, section-1.png, ...
COMPOSITE_FILENAME = "composite.png"
COMPOSITE_RECREATED_FILENAME = "composite-recreated.png"


def _parse_corners(d: dict[str, Any]) -> list[list[float]] | None:
    """Parse corners from manifest (4 points TL, TR, BR, BL in source image coords)."""
    c = d.get("corners")
    if not isinstance(c, list) or len(c) != 4:
        return None
    out = []
    for p in c:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append([float(p[0]), float(p[1])])
        elif isinstance(p, dict) and "x" in p and "y" in p:
            out.append([float(p["x"]), float(p["y"])])
        else:
            return None
    return out


@dataclass
class SectionMetadata:
    """
    One section: crop bounds, orientation, and reconstruction data.
    Section images are saved in the orientation the manifest expects:
    - rect: axis-aligned crop in source image coords (no rotation).
    - quad: perspective-warped to upright rectangle (TL, TR, BR, BL → [0,0], [w,0], [w,h], [0,h]).
    """
    index: int
    filename: str
    bounds: SectionBounds
    rotation_degrees: float  # 0, 90, 180, 270
    source_width: int
    source_height: int
    # Quad-only: corners in source image pixels (TL, TR, BR, BL). None for rect sections.
    corners: list[list[float]] | None = None
    # "rect" | "quad" — how this section was extracted and how to place it when recreating.
    section_type: str = "rect"
    # Actual dimensions of the saved section image (so recreation can verify/use).
    output_width_px: int = 0
    output_height_px: int = 0
    # Layout: reading order rank (0 = first in reading order).
    position_rank: int | None = None
    centroid_x: float | None = None
    centroid_y: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "index": self.index,
            "filename": self.filename,
            "bounds": self.bounds.to_dict(),
            "rotation_degrees": self.rotation_degrees,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "section_type": self.section_type,
            "output_width_px": self.output_width_px,
            "output_height_px": self.output_height_px,
            "origin_x": self.bounds.x,
            "origin_y": self.bounds.y,
            "width_px": self.bounds.width,
            "height_px": self.bounds.height,
        }
        if self.corners is not None:
            out["corners"] = self.corners
        if self.position_rank is not None:
            out["position_rank"] = self.position_rank
        if self.centroid_x is not None:
            out["centroid_x"] = self.centroid_x
        if self.centroid_y is not None:
            out["centroid_y"] = self.centroid_y
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SectionMetadata:
        corners = _parse_corners(d)
        return cls(
            index=int(d["index"]),
            filename=str(d["filename"]),
            bounds=SectionBounds.from_dict(d["bounds"]),
            rotation_degrees=float(d.get("rotation_degrees", 0)),
            source_width=int(d["source_width"]),
            source_height=int(d["source_height"]),
            corners=corners,
            section_type=str(d.get("section_type", "quad" if corners else "rect")),
            output_width_px=int(d.get("output_width_px", 0)),
            output_height_px=int(d.get("output_height_px", 0)),
            position_rank=int(d["position_rank"]) if d.get("position_rank") is not None else None,
            centroid_x=float(d["centroid_x"]) if d.get("centroid_x") is not None else None,
            centroid_y=float(d["centroid_y"]) if d.get("centroid_y") is not None else None,
        )


@dataclass
class SplitManifest:
    """
    Manifest for one source image: sections, coordinates, orientation, and layout.
    Single source of truth for recreating the composite: image location, section
    orientation (rect vs quad), relation to other sections (reading_order, section_relations),
    and all data needed to place each section image back onto the canvas.
    """
    source_filename: str
    source_width: int
    source_height: int
    sections: list[SectionMetadata] = field(default_factory=list)
    composite_filename: str | None = None
    composite_recreated_filename: str | None = None  # Pillow-built from manifest
    layout: dict[str, Any] | None = None  # reading_order, description, section_relations

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source_filename": self.source_filename,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "sections": [s.to_dict() for s in self.sections],
        }
        if self.composite_filename is not None:
            out["composite_filename"] = self.composite_filename
        if self.composite_recreated_filename is not None:
            out["composite_recreated_filename"] = self.composite_recreated_filename
        if self.layout is not None:
            out["layout"] = self.layout
        return out

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> SplitManifest:
        with open(path) as f:
            d = json.load(f)
        sections = [SectionMetadata.from_dict(s) for s in d["sections"]]
        return cls(
            source_filename=d["source_filename"],
            source_width=d["source_width"],
            source_height=d["source_height"],
            sections=sections,
            composite_filename=d.get("composite_filename"),
            composite_recreated_filename=d.get("composite_recreated_filename"),
            layout=d.get("layout"),
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
    source_w: int,
    source_h: int,
    sections: list[SectionMetadata],
    section_dir: Path,
    stem: str,
    extension: str,
) -> None:
    """
    Recreate full image from rect sections: paste each section image at its bounds origin.
    Section images are in manifest-defined orientation (axis-aligned crop; no rotation).
    Transparent background (RGBA). Writes composite.png. Uses sec.filename from manifest.
    """
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
    out_path = section_dir / COMPOSITE_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), composite)


def _write_composite_image(
    source_w: int,
    source_h: int,
    sections: list[SectionMetadata],
    section_dir: Path,
    stem: str,
    extension: str,
) -> None:
    """
    Recreate full image by placing each quad section at its original corners (from manifest).
    Section images are in manifest-defined orientation (warped upright: TL,TR,BR,BL).
    Uses sec.filename and sec.corners from manifest. Transparent background (RGBA).
    Writes composite.png.
    """
    ext = extension.lstrip(".")
    composite = np.zeros((source_h, source_w, 4), dtype=np.uint8)
    composite[:, :, 3] = 0
    for sec in sections:
        if not sec.corners or len(sec.corners) != 4:
            continue
        section_path = section_dir / sec.filename
        if not section_path.is_file():
            continue
        warped, _ = load_image(section_path)
        if warped is None or warped.size == 0:
            continue
        hw, ww = warped.shape[:2]
        pts_src = np.ascontiguousarray(
            np.array([[0.0, 0.0], [float(ww), 0.0], [float(ww), float(hw)], [0.0, float(hw)]], dtype=np.float32)
        )
        pts_dst = np.array(sec.corners, dtype=np.float32)
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
    out_path = section_dir / COMPOSITE_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), composite)


def _write_composite_recreated(
    source_w: int,
    source_h: int,
    sections: list[SectionMetadata],
    section_dir: Path,
    stem: str,
    extension: str,
) -> None:
    """
    Build composite_recreated.png by combining section images according to the manifest
    using Pillow. Same layout as composite.png but built from manifest data (bounds or
    corners) so it verifies reconstruction. Saved as composite-recreated.png.
    """
    canvas = PILImage.new("RGBA", (source_w, source_h), (0, 0, 0, 0))
    use_quads = any(s.corners is not None and len(s.corners) == 4 for s in sections)

    if use_quads:
        for sec in sections:
            if not sec.corners or len(sec.corners) != 4:
                continue
            section_path = section_dir / sec.filename
            if not section_path.is_file():
                continue
            warped, _ = load_image(section_path)
            if warped is None or warped.size == 0:
                continue
            hw, ww = warped.shape[:2]
            pts_src = np.ascontiguousarray(
                np.array([[0.0, 0.0], [float(ww), 0.0], [float(ww), float(hw)], [0.0, float(hw)]], dtype=np.float32)
            )
            pts_dst = np.array(sec.corners, dtype=np.float32)
            pts_dst = np.clip(pts_dst, 0.0, [float(source_w) - 0.01, float(source_h) - 0.01]).astype(np.float32)
            pts_dst = np.ascontiguousarray(pts_dst)
            if pts_src.shape != (4, 2) or pts_dst.shape != (4, 2):
                continue
            M = cv2.getPerspectiveTransform(pts_src, pts_dst)
            warped_onto = cv2.warpPerspective(warped, M, (source_w, source_h), flags=cv2.INTER_LINEAR)
            mask = np.zeros((source_h, source_w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts_dst.astype(np.int32), 255)
            mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8))
            if warped_onto.ndim == 2:
                warped_onto = cv2.cvtColor(warped_onto, cv2.COLOR_GRAY2BGR)
            if warped_onto.shape[2] == 3:
                rgba = np.zeros((source_h, source_w, 4), dtype=np.uint8)
                rgba[:, :, :3] = cv2.cvtColor(warped_onto, cv2.COLOR_BGR2RGB)
                rgba[:, :, 3] = mask
            else:
                rgba = cv2.cvtColor(warped_onto, cv2.COLOR_BGRA2RGBA)
                rgba[:, :, 3] = np.minimum(rgba[:, :, 3], mask)
            layer = PILImage.fromarray(rgba, mode="RGBA")
            canvas = PILImage.alpha_composite(canvas, layer)
    else:
        for sec in sections:
            section_path = section_dir / sec.filename
            if not section_path.is_file():
                continue
            try:
                pil_sec = PILImage.open(section_path).convert("RGBA")
            except Exception:
                continue
            arr = np.array(pil_sec)
            if arr.size == 0:
                continue
            ch, cw = arr.shape[:2]
            x = max(0, int(sec.bounds.x))
            y = max(0, int(sec.bounds.y))
            x2 = min(source_w, x + cw)
            y2 = min(source_h, y + ch)
            if x2 <= x or y2 <= y:
                continue
            w_avail, h_avail = x2 - x, y2 - y
            crop = pil_sec.crop((0, 0, w_avail, h_avail))
            canvas.paste(crop, (x, y), crop)

    out_path = section_dir / COMPOSITE_RECREATED_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path), "PNG")


def split_image_from_rects(
    image_path: Path,
    rects: list[dict[str, float]],
    output_dir: Path,
    extension: str = "png",
) -> SplitManifest:
    """
    Split image into sections using given rectangles. Sections are saved in
    manifest-defined orientation: axis-aligned crop (no rotation), same as source.
    rects: list of {x, y, width, height} in source image pixels.
    Writes section images and one manifest with all reconstruction data (bounds,
    orientation, layout, section relations). Composite can be recreated from manifest only.
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
        ch, cw = crop.shape[:2]
        ext = extension.lstrip(".")
        filename = SECTION_FILENAME_TEMPLATE.format(i=i, ext=ext)
        out_path = section_dir / filename
        cv2.imwrite(str(out_path), crop)
        cx = bounds.x + bounds.width / 2
        cy = bounds.y + bounds.height / 2
        sections.append(
            SectionMetadata(
                index=i,
                filename=filename,
                bounds=bounds,
                rotation_degrees=0,
                source_width=w,
                source_height=h,
                corners=None,
                section_type="rect",
                output_width_px=cw,
                output_height_px=ch,
                position_rank=None,  # set below after reading_order
                centroid_x=cx,
                centroid_y=cy,
            )
        )

    # Layout: reading order and section relations (orientation/position relative to others)
    centroids = [(s.centroid_x or 0, s.centroid_y or 0) for s in sections]
    order_indices = list(range(len(centroids)))
    order_indices.sort(key=lambda i: (centroids[i][1], centroids[i][0]))
    tol = min(w, h) * 0.05
    section_relations = []
    for i in range(len(centroids)):
        cx_i, cy_i = centroids[i]
        left_of = [j for j in range(len(centroids)) if j != i and centroids[j][0] < cx_i - tol and abs(centroids[j][1] - cy_i) < min(w, h) * 0.3]
        right_of = [j for j in range(len(centroids)) if j != i and centroids[j][0] > cx_i + tol and abs(centroids[j][1] - cy_i) < min(w, h) * 0.3]
        above = [j for j in range(len(centroids)) if j != i and centroids[j][1] < cy_i - tol]
        below = [j for j in range(len(centroids)) if j != i and centroids[j][1] > cy_i + tol]
        section_relations.append({"left_of": left_of, "right_of": right_of, "above": above, "below": below})
    for rank, idx in enumerate(order_indices):
        sections[idx].position_rank = rank

    layout = {
        "reading_order": order_indices,
        "description": "Section indices in top-to-bottom, left-to-right order by centroid.",
        "section_relations": section_relations,
    }
    manifest = SplitManifest(
        source_filename=image_path.name,
        source_width=w,
        source_height=h,
        sections=sections,
        composite_filename=COMPOSITE_FILENAME,
        composite_recreated_filename=COMPOSITE_RECREATED_FILENAME,
        layout=layout,
    )
    manifest_path = section_dir / MANIFEST_FILENAME
    manifest.save_json(manifest_path)
    _write_composite_image_rects(w, h, sections, section_dir, stem, extension)
    _write_composite_recreated(w, h, sections, section_dir, stem, extension)
    return manifest


def split_image_from_quads(
    image_path: Path,
    quads: list[list[dict[str, float]] | list[list[list[float]]]],
    output_dir: Path,
    extension: str = "png",
) -> SplitManifest:
    """
    Split image by perspective-correcting each quadrilateral to a rectangle.
    Sections are saved in manifest-defined orientation: warped to upright rectangle
    (TL, TR, BR, BL in source → [0,0], [w,0], [w,h], [0,h] in section image).
    Writes section images and one manifest with all reconstruction data (corners,
    bounds, orientation, layout, section relations). Composite can be recreated from manifest only.
    quads: list of 4-point lists, each point {x,y} or [x,y]. Order: TL, TR, BR, BL.
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
    for i, quad in enumerate(quads):
        if len(quad) != 4:
            continue
        try:
            warped = warp_quad_to_rect(img, quad)
        except Exception:
            continue
        if warped.size == 0:
            continue
        wh, ww = warped.shape[:2]
        ext = extension.lstrip(".")
        filename = SECTION_FILENAME_TEMPLATE.format(i=i, ext=ext)
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
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        sections.append(SectionMetadata(
            index=i,
            filename=filename,
            bounds=bounds,
            rotation_degrees=0.0,
            source_width=w,
            source_height=h,
            corners=corners,
            section_type="quad",
            output_width_px=ww,
            output_height_px=wh,
            position_rank=None,
            centroid_x=cx,
            centroid_y=cy,
        ))

    # Layout: reading order and section relations
    centroids = [(s.centroid_x or 0, s.centroid_y or 0) for s in sections]
    order_indices = list(range(len(centroids)))
    order_indices.sort(key=lambda i: (centroids[i][1], centroids[i][0]))
    tol = min(w, h) * 0.05
    section_relations = []
    for i in range(len(centroids)):
        cx_i, cy_i = centroids[i]
        left_of = [j for j in range(len(centroids)) if j != i and centroids[j][0] < cx_i - tol and abs(centroids[j][1] - cy_i) < min(w, h) * 0.3]
        right_of = [j for j in range(len(centroids)) if j != i and centroids[j][0] > cx_i + tol and abs(centroids[j][1] - cy_i) < min(w, h) * 0.3]
        above = [j for j in range(len(centroids)) if j != i and centroids[j][1] < cy_i - tol]
        below = [j for j in range(len(centroids)) if j != i and centroids[j][1] > cy_i + tol]
        section_relations.append({"left_of": left_of, "right_of": right_of, "above": above, "below": below})
    for rank, idx in enumerate(order_indices):
        sections[idx].position_rank = rank

    layout = {
        "reading_order": order_indices,
        "description": "Section indices in top-to-bottom, left-to-right order by centroid.",
        "section_relations": section_relations,
    }
    manifest = SplitManifest(
        source_filename=image_path.name,
        source_width=w,
        source_height=h,
        sections=sections,
        composite_filename=COMPOSITE_FILENAME,
        composite_recreated_filename=COMPOSITE_RECREATED_FILENAME,
        layout=layout,
    )
    manifest_path = section_dir / MANIFEST_FILENAME
    manifest.save_json(manifest_path)
    _write_composite_image(w, h, sections, section_dir, stem, extension)
    _write_composite_recreated(w, h, sections, section_dir, stem, extension)
    return manifest


def recreate_composite_from_manifest(manifest_path: Path, extension: str | None = None) -> None:
    """
    Recreate the composite image from a saved manifest and its section images.
    Uses only manifest data (source dimensions, section filenames, corners or bounds)
    so behavior is consistent with how sections were saved. Infers section_dir and stem
    from manifest_path; uses extension from first section filename if not provided.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = SplitManifest.load_json(manifest_path)
    section_dir = manifest_path.parent
    stem = section_dir.name
    ext = extension or "png"
    if manifest.sections:
        first_name = manifest.sections[0].filename
        if "." in first_name:
            ext = first_name.rsplit(".", 1)[-1]
    use_quads = any(s.corners is not None and len(s.corners) == 4 for s in manifest.sections)
    if use_quads:
        _write_composite_image(
            manifest.source_width,
            manifest.source_height,
            manifest.sections,
            section_dir,
            stem,
            ext,
        )
    else:
        _write_composite_image_rects(
            manifest.source_width,
            manifest.source_height,
            manifest.sections,
            section_dir,
            stem,
            ext,
        )
    _write_composite_recreated(
        manifest.source_width,
        manifest.source_height,
        manifest.sections,
        section_dir,
        stem,
        ext,
    )

