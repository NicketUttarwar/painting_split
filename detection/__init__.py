"""
Painting detection: localization + boundary refinement.
Input: image path. Output: list of quads (4 corners per painting).
"""
from detection.painting_detector import (
    detect_paintings,
    detect_paintings_iterative,
    detect_paintings_iterative_stream,
)

__all__ = [
    "detect_paintings",
    "detect_paintings_iterative",
    "detect_paintings_iterative_stream",
]
