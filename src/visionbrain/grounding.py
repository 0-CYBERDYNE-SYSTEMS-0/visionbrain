"""LFM VLM grounding — a third opinion on where the tracked targets are.

The ``analyze`` pipeline (``visionbrain.cli.cmd_analyze``) can ask the local
LFM2.5-VL to locate every instance of the SAM targets on the same key frames
Falcon refines, parse the boxes it emits from its text reply, and cross-check
them against SAM's detections for a third-engine view of the scene.

Parsing is tolerant of the formats small VLMs actually emit:

1. Canonical (what :func:`build_grounding_prompt` asks for) — one
   ``<box>x1,y1,x2,y2</box> label`` line per instance, integer coordinates
   normalized 0–1000.
2. Bare parenthesized corners — ``(x1,y1),(x2,y2)``.
3. JSON arrays — ``[[x1,y1,x2,y2], ...]`` or a flat ``[x1,y1,x2,y2]``.

Coordinate scale heuristic (per box, after parsing the four values): if all
four values are <= 1.5 the box is treated as 0–1 normalized; elif all four are
<= 100 it is treated as 0–100; otherwise it is treated as 0–1000. Values are
multiplied by the frame width/height in that scale, clamped to the image
bounds, and ordered so x1 < x2 and y1 < y2.

Everything here is importable on any host: stdlib (:mod:`re`, :mod:`json`)
plus the stdlib-only :mod:`visionbrain.crosscheck`. MLX/PIL are touched only
inside the ``__main__`` probe.
"""

from __future__ import annotations

import json
import re

from . import crosscheck

GROUNDING_IOU_THRESHOLD = 0.3

# Canonical form: <box>x1,y1,x2,y2</box> — optional spaces, int or float.
_BOX_TAG_RE = re.compile(
    r"<box>\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*</box>",
    re.IGNORECASE,
)
# Fallback: bare parenthesized corners — (x1,y1),(x2,y2).
_PAREN_PAIR_RE = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,?\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
)
# Fallback: JSON array of boxes — [[x1,y1,x2,y2], ...].
_JSON_NESTED_RE = re.compile(r"\[\s*\[[\d.,\s-]+\](?:\s*,\s*\[[\d.,\s-]+\])*\s*\]")
# Fallback: a single flat JSON box — [x1,y1,x2,y2].
_JSON_FLAT_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def build_grounding_prompt(targets: list[str]) -> str:
    """Build the instruction that makes a VLM emit parseable grounding lines.

    Names every target phrase to look for and pins the reply format: one line
    per instance, ``<box>x1,y1,x2,y2</box> label`` with INTEGER coordinates
    normalized 0–1000, or exactly ``NONE`` when nothing is visible.
    """
    phrases = [str(t).strip() for t in (targets or []) if t and str(t).strip()]
    target_list = ", ".join(phrases) if phrases else "objects"
    return (
        f"Locate every instance of the following target(s) in the image: {target_list}.\n"
        "Find EVERY instance, including small or partially hidden ones.\n"
        "Answer with exactly one line per instance in this format:\n"
        "<box>x1,y1,x2,y2</box> label\n"
        "where x1,y1 is the top-left corner and x2,y2 is the bottom-right corner "
        "of the object, as INTEGER coordinates normalized 0-1000 relative to the "
        "image size (0,0 = top-left, 1000,1000 = bottom-right), followed by the "
        "target phrase the box matches.\n"
        "Example: <box>120,340,480,760</box> boat\n"
        "If none of the targets are visible, answer exactly: NONE"
    )


def _scale_to_pixels(
    vals: list[float], width: float, height: float
) -> list[float]:
    """Convert four raw box values to clamped, ordered pixel xyxy.

    Scale heuristic: all values <= 1.5 -> 0-1 normalized; elif all <= 100 ->
    0-100; else 0-1000. Multiplies by width/height, clamps to the image
    bounds, and orders corners so x1 < x2 and y1 < y2.
    """
    if all(v <= 1.5 for v in vals):
        sx, sy = width, height
    elif all(v <= 100.0 for v in vals):
        sx, sy = width / 100.0, height / 100.0
    else:
        sx, sy = width / 1000.0, height / 1000.0
    x1 = min(max(vals[0] * sx, 0.0), width)
    y1 = min(max(vals[1] * sy, 0.0), height)
    x2 = min(max(vals[2] * sx, 0.0), width)
    y2 = min(max(vals[3] * sy, 0.0), height)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [x1, y1, x2, y2]


def parse_grounding_boxes(text: str, width: int, height: int) -> list[dict]:
    """Extract grounding boxes from a VLM reply as pixel-space detection dicts.

    Tries, in order: canonical ``<box>a,b,c,d</box>`` tags (label = trailing
    text after the tag, stripped, "" allowed), bare parenthesized
    ``(a,b),(c,d)`` corners, then JSON-array forms ``[[a,b,c,d],...]`` and a
    flat ``[a,b,c,d]``. Each box's four values go through the scale heuristic
    documented in the module docstring (all <= 1.5 -> 0-1, elif all <= 100 ->
    0-100, else 0-1000), are multiplied by ``width``/``height``, clamped to the
    image bounds, and ordered x1<x2, y1<y2.

    Returns ``[]`` when the reply is empty, exactly ``NONE``, or contains no
    recognizable box. Coordinates are floats in pixel space.
    """
    cleaned = (text or "").strip()
    if not cleaned or cleaned.upper().rstrip(".!") == "NONE":
        return []

    raw: list[tuple[list[float], str]] = []

    # 1. Canonical <box>...</box> tags, label = trailing text on the same line.
    for m in _BOX_TAG_RE.finditer(cleaned):
        vals = [float(m.group(i)) for i in (1, 2, 3, 4)]
        label = cleaned[m.end():].split("\n", 1)[0].strip()
        raw.append((vals, label))

    # 2. Bare parenthesized corners (no tag -> no label).
    if not raw:
        for m in _PAREN_PAIR_RE.finditer(cleaned):
            raw.append(([float(m.group(i)) for i in (1, 2, 3, 4)], ""))

    # 3. JSON arrays: nested [[...], ...] first, then a flat [...].
    if not raw:
        for m in _JSON_NESTED_RE.finditer(cleaned):
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            if not isinstance(data, list):
                continue
            for item in data:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) >= 4
                    and all(isinstance(v, (int, float)) for v in item[:4])
                ):
                    raw.append(([float(v) for v in item[:4]], ""))
    if not raw:
        for m in _JSON_FLAT_RE.finditer(cleaned):
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(data, list) and len(data) >= 4 and all(
                isinstance(v, (int, float)) for v in data[:4]
            ):
                raw.append(([float(v) for v in data[:4]], ""))

    w, h = float(width), float(height)
    return [
        {"bbox_xyxy": _scale_to_pixels(vals, w, h), "label": label}
        for vals, label in raw
    ]


def grounding_crosscheck(
    sam_dets: list[dict],
    boxes: list[dict],
    *,
    iou_threshold: float = GROUNDING_IOU_THRESHOLD,
) -> "crosscheck.CrosscheckResult":
    """Cross-check SAM detections against parsed LFM grounding boxes.

    Thin wrapper over :func:`visionbrain.crosscheck.crosscheck` with a looser
    default threshold (0.3 vs 0.5): VLM boxes are coarse, so a slightly
    offset box should still count as agreement. The returned result's
    ``frame_index`` is ``-1``; set it when the frame is known.
    """
    return crosscheck.crosscheck(sam_dets, boxes, iou_threshold=iou_threshold)


def _main() -> None:
    """Probe: ground argv targets on an argv image with the local LFM VLM.

    Prints the RAW model reply and the parsed boxes so a human can verify the
    real model's output format against :func:`parse_grounding_boxes`. Imports
    stay lazy — importing this module never touches MLX or PIL.
    """
    import sys

    if len(sys.argv) < 3:
        print(
            "usage: python -m visionbrain.grounding <image> <target> [target ...]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    from .inference_admission import AdmissionError, InferenceAdmission

    try:
        admission = InferenceAdmission()
        handle = admission.try_acquire("visionbrain-grounding")
    except AdmissionError as exc:
        raise SystemExit(f"inference admission unavailable: {exc}") from exc
    if handle is None:
        raise SystemExit(f"inference admission busy: {admission.describe_holder()}")

    with handle:
        from . import vlm_registry
        from PIL import Image

        image_path, targets = sys.argv[1], sys.argv[2:]
        image = Image.open(image_path)
        prompt = build_grounding_prompt(targets)

        vlm_registry.set_model("lfm")
        reply = vlm_registry.ask(prompt, image=image)

        print(f"=== raw LFM reply ({vlm_registry.current_model()}) " + "=" * 20)
        print(reply)
        print("=== parsed boxes " + "=" * 40)
        boxes = parse_grounding_boxes(reply, image.size[0], image.size[1])
        if not boxes:
            print("  [none]")
        for box in boxes:
            print(f"  {box['label']!r}: {box['bbox_xyxy']}")


if __name__ == "__main__":
    _main()
