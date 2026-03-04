#!/usr/bin/env python3
"""
Split a source image into sections using a given JSON sections list (rects or quads).
For painting retrieval (detect + extract), use scripts/retrieve_one.py or scripts/retrieve_paintings_batch.py.
Usage: python scripts/split_from_sections.py <image_path> <sections_json> [output_dir]
  sections_json: array of {x,y,width,height} or {corners: [[x,y],...]} (4 corners per section).
  If sections have "corners", uses perspective warp and writes to output_dir (e.g. outputs/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from image_processor import split_image_from_rects, split_image_from_quads


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/split_from_sections.py <image_path> <sections_json> [output_dir]")
        sys.exit(1)
    image_path = Path(sys.argv[1])
    sections_arg = sys.argv[2]
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("outputs")

    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        sys.exit(1)

    data = json.loads(sections_arg)
    sections = data.get("sections", data) if isinstance(data, dict) else data
    if not isinstance(sections, list):
        sections = [data]

    use_quads = all(
        isinstance(s.get("corners"), list) and len(s.get("corners")) == 4
        for s in sections
    )
    if use_quads:
        quads = [s["corners"] for s in sections]
        manifest = split_image_from_quads(image_path, quads=quads, output_dir=output_dir)
    else:
        manifest = split_image_from_rects(image_path, rects=sections, output_dir=output_dir)
    print(f"Wrote {len(manifest.sections)} sections to {output_dir}/")
    for s in manifest.sections:
        print(f"  {s.filename}")


if __name__ == "__main__":
    main()
