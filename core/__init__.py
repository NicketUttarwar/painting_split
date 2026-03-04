"""
Core image operations: load/save, coordinate types (rect, quad), crop, perspective warp.
No AI, no HTTP. Used by detection and extraction.
"""
from core.image import (
    IMAGE_EXTENSIONS,
    SectionBounds,
    load_image,
    save_image,
    safe_stem,
    quad_to_pts,
    quad_output_size,
    crop_section,
    warp_quad_to_rect,
    extract_quad_region,
)

__all__ = [
    "IMAGE_EXTENSIONS",
    "SectionBounds",
    "load_image",
    "save_image",
    "safe_stem",
    "quad_to_pts",
    "quad_output_size",
    "crop_section",
    "warp_quad_to_rect",
    "extract_quad_region",
]
