#!/usr/bin/env bash
# Run the Painting Split web app (use the project venv).
set -e
cd "$(dirname "$0")"

create_venv() {
  echo "Creating venv and installing deps..."
  (python3.13 -m venv .venv 2>/dev/null) || (python3 -m venv .venv)
  .venv/bin/pip install -q -r requirements.txt
}

if [[ ! -d .venv ]]; then
  create_venv
else
  # Recreate venv if it's broken (e.g. wrong architecture after switching machines)
  if ! .venv/bin/python -c "import cv2" 2>/dev/null; then
    echo "Venv broken or wrong architecture; recreating..."
    rm -rf .venv
    create_venv
  fi
fi

.venv/bin/python app.py
