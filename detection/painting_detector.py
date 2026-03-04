"""
Painting-only detection using OpenAI vision.
Identifies only painting(s); excludes wall, floor, carpet, drop cloth, supplies.
Returns list of quads (4 corners per panel) for extraction.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from core.image import load_image

DEFAULT_VISION_MODEL = "gpt-4o"
MAX_IMAGE_PX = 2048
MAX_REFINEMENT_ITERATIONS = 5


def _image_to_base64(path: Path, max_size_px: int = MAX_IMAGE_PX) -> tuple[str, int, int]:
    """Load image, optionally downscale; return (base64 png, orig_w, orig_h)."""
    import cv2
    img, _ = load_image(path)
    if img is None or img.size == 0:
        raise ValueError(f"Could not load image: {path}")
    orig_h, orig_w = img.shape[:2]
    sent_w, sent_h = orig_w, orig_h
    if max(orig_h, orig_w) > max_size_px:
        scale = max_size_px / max(orig_h, orig_w)
        sent_w = max(1, int(orig_w * scale))
        sent_h = max(1, int(orig_h * scale))
        img = cv2.resize(img, (sent_w, sent_h), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".png", img)
    data = base64.standard_b64encode(buf.tobytes()).decode("ascii")
    return data, orig_w, orig_h


def _corners_to_normalized(sections: list[dict], image_w: int, image_h: int) -> list[dict]:
    """Convert section corners from pixel coords to 0–1000 normalized."""
    if image_w <= 0 or image_h <= 0:
        return []
    scale_x = 1000.0 / image_w
    scale_y = 1000.0 / image_h
    out = []
    for s in sections:
        corners = s.get("corners")
        if not isinstance(corners, list) or len(corners) != 4:
            continue
        pts = []
        for c in corners:
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                x = float(c[0]) * scale_x
                y = float(c[1]) * scale_y
                pts.append([round(x, 2), round(y, 2)])
            else:
                break
        if len(pts) == 4:
            out.append({"corners": pts})
    return out


# Painting-only: exclude wall, floor, carpet, drop cloth, supplies.
SYSTEM_PROMPT = """You are an expert at analyzing photos that contain paintings.

Goal: Identify ONLY the painting(s) in the image. We need (x, y) coordinates for every corner of every painting or panel so we can extract just the painting images—with no background.

Critical rules:
- Include ONLY the actual painting(s): canvas(es), panels (diptych, triptych, grids, L-shapes). Each panel gets its own 4 corners.
- EXCLUDE everything that is not the painting: wall, floor, carpet, rug, drop cloth, table, furniture, art supplies (palette, brushes, cups, tubes), and any other objects. Do not include any background or non-painting region in your sections.
- The photo may be handheld (perspective warp). Edges of the painting are straight in reality; detect the true canvas edge, not shadows or reflections.
- There may be ONE painting (single canvas) or MULTIPLE panels (connected or with gaps). Return one section per panel, each with 4 corners: top-left, top-right, bottom-right, bottom-left.
- If you see no painting (e.g. only a room or only supplies), return "sections": [].

You must reply with valid JSON only. No other text before or after."""

USER_PROMPT_INITIAL = """Analyze this image and identify ONLY the painting(s). Exclude wall, floor, carpet, drop cloth, furniture, art supplies, and any non-painting object.

We need (x, y) coordinates for every corner of every painting or panel. The image may be from a handheld camera (warped). Edges are straight in reality; detect the true canvas edge.

Use NORMALIZED coordinates: x and y between 0 and 1000. Image width maps to x 0–1000, height to y 0–1000. Top-left is (0,0).

Return a single JSON object with this exact structure:
{
  "sections": [
    {
      "corners": [
        [x_tl, y_tl],
        [x_tr, y_tr],
        [x_br, y_br],
        [x_bl, y_bl]
      ]
    }
  ],
  "self_assessment": {
    "refinement_notes": ""
  }
}

- Corner order for each section: top-left (tl), top-right (tr), bottom-right (br), bottom-left (bl).
- One entry in "sections" per painting panel. If there are no paintings, return "sections": [].
- In self_assessment.refinement_notes describe what you see: how many paintings/panels, what you excluded (e.g. "excluded wooden floor, excluded palette on left"). This is used in the next refinement pass."""

USER_PROMPT_REFINE_TEMPLATE = """This is refinement pass {iteration} of {max_iterations}. We need (x,y) for every corner of every painting only. Use the previous result and the notes below to improve corner positions. Do not include any non-painting region.

Previous detection (normalized 0–1000 coordinates):
{previous_json}

Refinement notes from the previous pass:
{refinement_notes}

Your task: Adjust the corner points to better match the actual painting edges. Keep excluding wall, floor, carpet, and objects. Return the same JSON structure with NORMALIZED coordinates (0–1000). In self_assessment.refinement_notes explain what you changed."""


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract first complete JSON object from model reply."""
    text = (text or "").strip()
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if json_match:
        text = json_match.group(1)
    else:
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        text = text[start : i + 1]
                        break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_sections_json(text: str, image_w: int, image_h: int) -> list[dict[str, Any]]:
    """Parse model reply into list of sections with corners; scale from 0–1000 to image pixels."""
    data = _extract_json(text)
    if not isinstance(data, dict):
        return []
    sections = data.get("sections")
    if not isinstance(sections, list):
        return []
    scale_x = image_w / 1000.0 if image_w > 0 else 1.0
    scale_y = image_h / 1000.0 if image_h > 0 else 1.0
    out: list[dict[str, Any]] = []
    for s in sections:
        corners = s.get("corners") if isinstance(s, dict) else None
        if not isinstance(corners, list) or len(corners) != 4:
            continue
        pts = []
        for c in corners:
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                x = float(c[0]) * scale_x
                y = float(c[1]) * scale_y
                x = max(0, min(image_w, x))
                y = max(0, min(image_h, y))
                pts.append([x, y])
            else:
                break
        if len(pts) == 4:
            out.append({
                "x": min(p[0] for p in pts),
                "y": min(p[1] for p in pts),
                "width": max(p[0] for p in pts) - min(p[0] for p in pts),
                "height": max(p[1] for p in pts) - min(p[1] for p in pts),
                "rotation_degrees": 0,
                "corners": pts,
            })
    return out


def _parse_self_assessment(data: dict[str, Any]) -> str:
    """Extract refinement_notes from response."""
    assessment = data.get("self_assessment") if isinstance(data, dict) else None
    if not isinstance(assessment, dict):
        return ""
    notes = assessment.get("refinement_notes")
    return (notes or "").strip() if isinstance(notes, str) else ""


def _call_vision(
    client: OpenAI,
    b64: str,
    prompt: str,
    model: str = DEFAULT_VISION_MODEL,
) -> str:
    """Single vision API call; returns assistant message content."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ],
        max_tokens=4096,
    )
    return (response.choices[0].message.content or "").strip() if response.choices else ""


def detect_paintings_iterative(
    image_path: Path,
    api_key: str,
    max_iterations: int = MAX_REFINEMENT_ITERATIONS,
    model: str = DEFAULT_VISION_MODEL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Detect painting regions with iterative refinement. Returns list of sections
    (each with "corners": 4 points) and metadata. Excludes non-painting regions.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    b64, orig_w, orig_h = _image_to_base64(path)
    client = OpenAI(api_key=api_key)

    sections: list[dict[str, Any]] = []
    refinement_notes = ""
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        if iteration == 1:
            prompt = USER_PROMPT_INITIAL
        else:
            previous_normalized = _corners_to_normalized(sections, orig_w, orig_h)
            previous_json = json.dumps({"sections": previous_normalized}, indent=2)
            prompt = USER_PROMPT_REFINE_TEMPLATE.format(
                iteration=iteration,
                max_iterations=max_iterations,
                previous_json=previous_json,
                refinement_notes=refinement_notes or "(No notes; please refine corner positions to match painting edges.)",
            )

        text = _call_vision(client, b64, prompt, model=model)
        data = _extract_json(text)
        if data is None:
            break
        sections = _parse_sections_json(text, orig_w, orig_h)
        refinement_notes = _parse_self_assessment(data)

        if not sections:
            break

    metadata: dict[str, Any] = {
        "iterations": iteration,
        "refinement_notes": refinement_notes,
        "refinement_used": iteration > 1,
    }
    return sections, metadata


def detect_paintings(image_path: Path, api_key: str) -> list[dict[str, Any]]:
    """Detect painting quads (one per panel). Returns list of sections with corners."""
    sections, _ = detect_paintings_iterative(image_path, api_key)
    return sections


def detect_paintings_iterative_stream(
    image_path: Path,
    api_key: str,
    max_iterations: int = MAX_REFINEMENT_ITERATIONS,
    model: str = DEFAULT_VISION_MODEL,
):
    """
    Same as detect_paintings_iterative but yields progress events: start,
    iteration_start, iteration_done, done, error.
    """
    path = Path(image_path)
    if not path.is_file():
        yield {"event": "error", "error": f"Image not found: {path}"}
        return
    try:
        b64, orig_w, orig_h = _image_to_base64(path)
    except Exception as e:
        yield {"event": "error", "error": str(e)}
        return
    client = OpenAI(api_key=api_key)

    yield {
        "event": "start",
        "max_iterations": max_iterations,
        "message": f"Starting painting detection ({max_iterations} refinement passes). Identifying only paintings; excluding background and objects.",
    }

    sections: list[dict[str, Any]] = []
    refinement_notes = ""
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        if iteration == 1:
            prompt = USER_PROMPT_INITIAL
            prompt_summary = "Initial pass: find all painting panels and return corners; exclude non-painting regions."
        else:
            previous_normalized = _corners_to_normalized(sections, orig_w, orig_h)
            previous_json = json.dumps({"sections": previous_normalized}, indent=2)
            refinement_notes_filled = refinement_notes or "(No notes; refine corners to match painting edges.)"
            prompt = USER_PROMPT_REFINE_TEMPLATE.format(
                iteration=iteration,
                max_iterations=max_iterations,
                previous_json=previous_json,
                refinement_notes=refinement_notes_filled,
            )
            prompt_summary = f"Refinement pass {iteration}/{max_iterations}: adjust corners."

        yield {
            "event": "iteration_start",
            "iteration": iteration,
            "max_iterations": max_iterations,
            "prompt_summary": prompt_summary,
            "message": f"Pass {iteration} of {max_iterations}: finding painting corners.",
        }
        yield {"event": "api_call", "iteration": iteration, "message": f"Calling OpenAI (pass {iteration})…"}

        try:
            text = _call_vision(client, b64, prompt, model=model)
        except Exception as e:
            yield {"event": "error", "error": str(e)}
            return
        data = _extract_json(text)
        if data is None:
            yield {
                "event": "iteration_done",
                "iteration": iteration,
                "sections": sections,
                "sections_count": len(sections),
                "refinement_notes": refinement_notes or "Could not parse response.",
                "message": f"Pass {iteration}: parse failed.",
            }
            break
        sections = _parse_sections_json(text, orig_w, orig_h)
        refinement_notes = _parse_self_assessment(data)

        yield {
            "event": "iteration_done",
            "iteration": iteration,
            "sections": sections,
            "sections_count": len(sections),
            "refinement_notes": refinement_notes,
            "message": f"Pass {iteration} done: {len(sections)} painting(s).",
        }

        if not sections:
            break

    metadata = {
        "iterations": iteration,
        "refinement_notes": refinement_notes,
        "refinement_used": iteration > 1,
    }
    yield {
        "event": "done",
        "sections": sections,
        "refinement": metadata,
        "message": f"Done. {len(sections)} painting(s) detected.",
    }
