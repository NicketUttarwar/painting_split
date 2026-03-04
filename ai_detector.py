"""
AI-powered canvas detection using OpenAI vision models (GPT-4o).
Detects edges and corners of one or more canvases; handles connected panels and gaps.
Iterative refinement: re-calls the vision model exactly 5 times, each time sending
the previous corners and asking for adjustments. No early exit — always 5 passes.
Returns a list of sections, each with 4 corners (quad) for perspective correction.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from image_processor import load_image
except ImportError:
    load_image = None

DEFAULT_VISION_MODEL = "gpt-4o"
MAX_IMAGE_PX = 2048
MAX_REFINEMENT_ITERATIONS = 5


def _image_to_base64(path: Path, max_size_px: int = MAX_IMAGE_PX) -> tuple[str, int, int]:
    """Load image, optionally downscale; return (base64 png, orig_w, orig_h)."""
    if load_image is None:
        raise RuntimeError("image_processor not available")
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
    """Convert section corners from pixel coords to 0–1000 normalized for sending back to the model."""
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


# --- Handheld / warped context: single perspective, re-adjustment expected ---
# Goal: image separation for each canvas; x,y for every corner of every canvas/sub-canvas; straight edges with shadow.
SYSTEM_PROMPT = """You are an expert at analyzing photos of paintings and multi-panel artwork taken with handheld cameras.

Goal: There is a canvas or a set of canvases put together. We want image separation for each one — so we need (x, y) coordinates for every single corner of every canvas or sub-canvas. Each canvas will be extracted and re-warped to a rectangle using these corners.

Critical context — assume every image is a handheld camera photo:
- The image is from a single perspective and will be warped. Nothing is perfectly rectangular in the photo.
- Edges are physically straight in the real world; in the photo they may appear as lines with perspective. It is safe to assume straight lines for the edges, with some shadow or lighting variation throughout — still detect the true edge, not the shadow.
- You must capture: the outer edges of each canvas, the inner edges between connected panels (diptych, triptych, grids, L-shapes), and every corner and connecting point of every canvas. Accuracy of these points is essential for correct perspective correction and image separation.
- There may be ONE canvas or MULTIPLE canvases. Multiple canvases can be CONNECTED (touching) or SEPARATE (visible gap). When touching, detect the boundary (frame edge, seam, divider) and return each canvas as its own section with its own four corners.
- Each canvas appears as a quadrilateral in the photo; return the 4 corners as seen in the image. Ignore background and non-canvas areas.

You must reply with valid JSON only. No other text before or after."""

PROMPT_SUMMARY_INITIAL = "Initial analysis: find every canvas/sub-canvas and return (x,y) for every corner. Image separation per canvas; assume straight edges with possible shadow."

USER_PROMPT_INITIAL = """Analyze this image. There is a canvas or a set of canvases put together; we want image separation for each one. We need (x, y) coordinates for every single corner of every canvas or sub-canvas. Sometimes the image is warped (handheld); factor this in. Edges are straight in reality; there may be shadow — still detect the true edge.

Pay close attention to outer edges, inner edges between panels, and every connecting point and edge of every canvas.

Image dimensions (for your reference): the image may be resized for analysis. Use the NORMALIZED coordinate system below.

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
- Use NORMALIZED coordinates: x and y between 0 and 1000. Image width maps to x 0–1000, height to y 0–1000. Top-left is (0,0), top-right (1000,0), bottom-right (1000,1000), bottom-left (0,1000).
- self_assessment.refinement_notes: Be verbose. Describe what you see and what you did: how many canvases or panels you found, where the outer and inner edges are, which corners you placed and why (e.g. "section 1 top-left at frame edge; section 2 bottom-right at seam between panels"). Note any warping or shadow you accounted for. This text is shown to the user and used in the next refinement pass.

Return one entry in "sections" per canvas/panel. Ensure corners form a convex quadrilateral in TL, TR, BR, BL order."""

PROMPT_SUMMARY_REFINE = "Refinement pass {iteration}/{max_iterations}: adjust corner (x,y) positions using the previous result and refinement notes. Goal: better match to canvas edges (straight lines, possible shadow)."

USER_PROMPT_REFINE_TEMPLATE = """This is refinement pass {iteration} of up to {max_iterations}. We are doing image separation for each canvas and need (x,y) for every corner. Use the previous detection and the notes below to improve the corner positions. Edges are straight with possible shadow; adjust points to match true edges.

Previous detection (normalized 0–1000 coordinates):
{previous_json}

Refinement notes from the previous pass:
{refinement_notes}

Image dimensions (same as before): width and height map to x,y in 0–1000.

Your task: Adjust the corner points so they better match the actual canvas edges — outer edges, inner edges, and every connecting point. Assume handheld camera warp and single-point perspective; small adjustments are expected.

Return a single JSON object with the same structure:
{
  "sections": [ ... ],
  "self_assessment": {
    "refinement_notes": ""
  }
}

Use NORMALIZED coordinates (0–1000). In self_assessment.refinement_notes be verbose and explain exactly what you changed: for each section and corner you adjusted, name it (e.g. "section 2, top-left") and describe the change (e.g. "moved ~3% left to align with the frame edge", "section 1 bottom-right moved down to meet the seam"). If you kept some corners unchanged, you can say so briefly. This description is shown to the user and drives the next pass."""


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


def _parse_self_assessment(data: dict[str, Any]) -> tuple[int, str]:
    """Extract refinement_notes from response (verbose 'what changed' description). Optionally parses satisfaction_percent if present (not used for control)."""
    assessment = data.get("self_assessment") if isinstance(data, dict) else None
    if not isinstance(assessment, dict):
        return 0, ""
    notes = assessment.get("refinement_notes")
    notes = (notes or "").strip() if isinstance(notes, str) else ""
    return 0, notes


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


def detect_canvas_sections_iterative(
    image_path: Path,
    api_key: str,
    max_iterations: int = MAX_REFINEMENT_ITERATIONS,
    model: str = DEFAULT_VISION_MODEL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Detect canvas sections with iterative refinement. Re-calls the vision model
    exactly max_iterations times (default 5). Each pass sends the previous corners
    and refinement_notes; the model returns updated corners and a verbose description
    of what it changed. No early exit — the loop always runs 5 times (or until
    parse failure / empty sections).
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
                refinement_notes=refinement_notes or "(No specific notes; please refine corner positions for better fit to canvas edges.)",
            )

        text = _call_vision(client, b64, prompt, model=model)
        data = _extract_json(text)
        if data is None:
            break
        sections = _parse_sections_json(text, orig_w, orig_h)
        _, refinement_notes = _parse_self_assessment(data)

        if not sections:
            break

    metadata: dict[str, Any] = {
        "iterations": iteration,
        "refinement_notes": refinement_notes,
        "refinement_used": iteration > 1,
    }
    return sections, metadata


def detect_canvas_sections(image_path: Path, api_key: str) -> list[dict[str, Any]]:
    """
    Use OpenAI vision model with iterative refinement. Runs exactly 5 passes;
    each pass refines corners using the previous result and verbose change notes.
    Returns a list of sections, each with corners (4 points: TL, TR, BR, BL).
    """
    sections, _ = detect_canvas_sections_iterative(image_path, api_key)
    return sections


def detect_canvas_sections_iterative_stream(
    image_path: Path,
    api_key: str,
    max_iterations: int = MAX_REFINEMENT_ITERATIONS,
    model: str = DEFAULT_VISION_MODEL,
):
    """
    Same as detect_canvas_sections_iterative but yields progress events for each
    API call and result. The loop runs exactly max_iterations (5) times, then ends.
    Yields dicts: start, iteration_start, iteration_done, done, or error.
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
        "message": "Starting iterative canvas detection. The loop will run exactly " + str(max_iterations) + " times. Each pass: we send the image and (from pass 2 onward) the previous corner coordinates and the model's description of what it changed; the model returns updated corners and a verbose explanation of what it changed. Sections (the quads on the image) are updated and shown after each pass.",
    }

    sections: list[dict[str, Any]] = []
    refinement_notes = ""
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        if iteration == 1:
            prompt = USER_PROMPT_INITIAL
            prompt_summary = PROMPT_SUMMARY_INITIAL
        else:
            previous_normalized = _corners_to_normalized(sections, orig_w, orig_h)
            previous_json = json.dumps({"sections": previous_normalized}, indent=2)
            refinement_notes_filled = refinement_notes or "(No specific notes; please refine corner positions for better fit to canvas edges.)"
            prompt = USER_PROMPT_REFINE_TEMPLATE.format(
                iteration=iteration,
                max_iterations=max_iterations,
                previous_json=previous_json,
                refinement_notes=refinement_notes_filled,
            )
            prompt_summary = PROMPT_SUMMARY_REFINE.format(
                iteration=iteration, max_iterations=max_iterations
            )

        yield {
            "event": "iteration_start",
            "iteration": iteration,
            "max_iterations": max_iterations,
            "prompt_summary": prompt_summary,
            "prompt_full": prompt,
            "message": f"Loop {iteration} of {max_iterations}: Sending this prompt to OpenAI. It asks the model to " + ("find all canvas corners (first pass)." if iteration == 1 else "refine the corner positions using the previous result and the change notes from the last pass."),
        }
        # Yield a small event so the stream flushes before the long-running API call; keeps UI and connection alive.
        yield {
            "event": "api_call",
            "iteration": iteration,
            "message": f"Calling OpenAI for loop {iteration} (this may take 20–60 seconds)…",
        }

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
                "refinement_notes": refinement_notes or "Could not parse model response.",
                "prompt_used": prompt,
                "prompt_summary": prompt_summary,
                "message": f"Loop {iteration}: Could not parse JSON from model; stopping.",
            }
            break
        sections = _parse_sections_json(text, orig_w, orig_h)
        _, refinement_notes = _parse_self_assessment(data)

        yield {
            "event": "iteration_done",
            "iteration": iteration,
            "sections": sections,
            "sections_count": len(sections),
            "refinement_notes": refinement_notes,
            "prompt_used": prompt,
            "prompt_summary": prompt_summary,
            "message": (
                f"Loop {iteration} done: {len(sections)} section(s) (these are the quads drawn on the image). "
                + "What the model said it changed: " + (refinement_notes[:400] + "…" if len(refinement_notes or "") > 400 else (refinement_notes or "—"))
            ),
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
        "message": f"All {max_iterations} passes complete. Final result: {len(sections)} section(s). The sections shown on the canvas are the chosen quads; you can adjust corners by hand or export.",
    }
