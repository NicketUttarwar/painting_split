#!/usr/bin/env python3
"""
Create a sample secrets YAML file. Copy the output to secrets.yaml and add your Anthropic API key.
secrets.yaml is gitignored and never committed.
"""
from pathlib import Path

SAMPLE = """# Painting Split — secrets (do not commit secrets.yaml)
# Copy this file to secrets.yaml in the project root and fill in your API key.
# Get an API key: https://console.anthropic.com/

anthropic:
  api_key: "your-anthropic-api-key-here"
"""

def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out = root / "secrets.example.yaml"
    out.write_text(SAMPLE.strip() + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print("Next: copy to secrets.yaml and set anthropic.api_key to your key.")
    print("  cp secrets.example.yaml secrets.yaml")

if __name__ == "__main__":
    main()
