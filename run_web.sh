#!/usr/bin/env bash
# Run the Painting Split web app (use the project venv).
set -e
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
  echo "Creating venv and installing deps..."
  python3.13 -m venv .venv 2>/dev/null || python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
.venv/bin/python app.py
