#!/usr/bin/env python3
"""
Run painting retrieval on all images in data/inputs/. Writes to data/extractions/<source_id>/.
Requires OpenAI API key in secrets.yaml.
Usage: python scripts/retrieve_paintings_batch.py [--inputs-dir DIR] [--extractions-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.image import IMAGE_EXTENSIONS

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch retrieve paintings from all images in data/inputs/")
    parser.add_argument("--inputs-dir", type=Path, default=None, help="Input directory (default: project/data/inputs)")
    parser.add_argument("--extractions-dir", type=Path, default=None, help="Output root (default: project/data/extractions)")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    inputs_dir = args.inputs_dir or project_root / "data" / "inputs"
    extractions_dir = args.extractions_dir or project_root / "data" / "extractions"

    try:
        from secrets_loader import get_openai_api_key
        from detection.painting_detector import detect_paintings_iterative
        from extraction.extractor import extract_paintings
    except ImportError as e:
        print("Error: missing dependencies or modules:", e)
        sys.exit(1)

    api_key = get_openai_api_key()
    if not api_key:
        print("Error: OpenAI API key not configured. Set openai.api_key in secrets.yaml")
        sys.exit(1)

    inputs_dir.mkdir(parents=True, exist_ok=True)
    extractions_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        p for p in inputs_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {*IMAGE_EXTENSIONS, ".heic"}
    )
    if not paths:
        print(f"No images in {inputs_dir}")
        return

    print(f"Processing {len(paths)} image(s) from {inputs_dir}")
    for path in paths:
        print(f"  {path.name}…", end=" ", flush=True)
        try:
            sections, _ = detect_paintings_iterative(path, api_key)
            if not sections:
                print("no paintings detected")
                continue
            quads = [s["corners"] for s in sections]
            result = extract_paintings(path, quads, extractions_dir)
            n = len(result.get("paintings", []))
            print(f"{n} painting(s) -> {result.get('output_dir', '')}")
        except Exception as e:
            print(f"error: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
