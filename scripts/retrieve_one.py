#!/usr/bin/env python3
"""
Retrieve paintings from a single image: run detection + extraction, or extract from given sections JSON.
Usage:
  python scripts/retrieve_one.py <image_path>                    # detect + extract
  python scripts/retrieve_one.py <image_path> <sections.json>   # extract from JSON (sections with corners)
Output: data/extractions/<source_id>/painting_0.png, ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/retrieve_one.py <image_path> [sections.json]")
        sys.exit(1)
    image_path = Path(sys.argv[1])
    sections_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not image_path.is_file():
        print(f"Error: image not found: {image_path}")
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent
    extractions_dir = project_root / "data" / "extractions"
    extractions_dir.mkdir(parents=True, exist_ok=True)

    if sections_path is not None:
        if not sections_path.is_file():
            print(f"Error: sections file not found: {sections_path}")
            sys.exit(1)
        with open(sections_path) as f:
            data = json.load(f)
        sections = data.get("sections", data) if isinstance(data, dict) else data
        if not isinstance(sections, list):
            print("Error: sections must be a list (or object with 'sections' key)")
            sys.exit(1)
        quads = []
        for s in sections:
            corners = s.get("corners") if isinstance(s, dict) else None
            if isinstance(corners, list) and len(corners) == 4:
                quads.append(corners)
        if not quads:
            print("Error: no valid sections with 4 corners each")
            sys.exit(1)
        from extraction.extractor import extract_paintings
        result = extract_paintings(image_path, quads, extractions_dir)
    else:
        try:
            from secrets_loader import get_openai_api_key
            from detection.painting_detector import detect_paintings_iterative
            from extraction.extractor import extract_paintings
        except ImportError as e:
            print("Error: missing modules:", e)
            sys.exit(1)
        api_key = get_openai_api_key()
        if not api_key:
            print("Error: OpenAI API key not configured (secrets.yaml)")
            sys.exit(1)
        sections, _ = detect_paintings_iterative(image_path, api_key)
        if not sections:
            print("No paintings detected.")
            sys.exit(0)
        quads = [s["corners"] for s in sections]
        result = extract_paintings(image_path, quads, extractions_dir)

    n = len(result.get("paintings", []))
    print(f"Extracted {n} painting(s) to {result.get('output_dir', '')}")
    for p in result.get("paintings", []):
        print(f"  {p.get('filename', '')}")


if __name__ == "__main__":
    main()
