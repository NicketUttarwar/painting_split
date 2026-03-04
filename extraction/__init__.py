"""
Extraction: warp each painting quad to a rectangle and write to data/extractions/<source_id>/.
"""
from extraction.extractor import extract_paintings

__all__ = ["extract_paintings"]
