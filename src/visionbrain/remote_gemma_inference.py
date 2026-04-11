"""Remote Gemma 4 inference — calls the remote server instead of loading locally.

This is the partner to gemma_inference.py for machines where Gemma 4 doesn't fit
in local RAM (e.g. Mac Mini M4 16GB).
Uses the OpenAI-compatible API at the remote endpoint.

Remote server: http://100.72.41.118:8080
Endpoint:     /v1/chat/completions
Model:        mlx-community/gemma-4-26b-a4b-it-4bit
"""

from __future__ import annotations

import time
import urllib.request
import urllib.error
import json
from dataclasses import dataclass
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_ENDPOINT = "http://100.72.41.118:8080/v1/chat/completions"
DEFAULT_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"
SYSTEM_PROMPT = (
    "You are an agricultural intelligence assistant helping farmers and ranchers "
    "analyze drone and camera footage. You have access to structured object detection "
    "data from vision AI models: bounding boxes with confidence scores, pixel-level "
    "segmentation masks with area fractions, object tracks across video frames "
    "(track IDs, centroid positions), and class labels (e.g. 'cow', 'sheep', 'fence', "
    "'crop row'). "
    "Be specific, practical, and actionable. Focus on: animal health and behavior "
    "(injuries, isolation, unusual movement), infrastructure (fence damage, water "
    "trough availability), crop stress indicators, and anomalies requiring human "
    "attention. Keep reports concise but detailed enough to act on in the field."
)


# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GemmaStats:
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    decode_ms: float


@dataclass
class GemmaResponse:
    text: str
    stats: GemmaStats


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _serialize_detections(detections: list[dict]) -> str:
    if not detections:
        return "No detections available."
    lines = []
    for d in detections:
        label = d.get("label", "?")
        score = d.get("score", 0)
        track_id = d.get("track_id", d.get("id", "?"))
        cx = d.get("centroid_norm", {}).get("x", 0)
        cy = d.get("centroid_norm", {}).get("y", 0)
        area = d.get("area_fraction", 0)
        region = d.get("image_region", "unknown")
        lines.append(
            f"  [{track_id}] {label} | conf={score:.2f} | "
            f"centroid=({cx:.2f}, {cy:.2f}) | area={area:.3f} | region={region}"
        )
    return "\n".join(lines)


def _serialize_frame_history(frames: list[dict]) -> str:
    if not frames:
        return "No frame history."
    lines = []
    for frame in frames:
        frame_id = frame.get("frame_index", "?")
        ts = frame.get("timestamp", "?")
        dets = frame.get("detections", [])
        if not dets:
            lines.append(f"Frame {frame_id} (t={ts}s): no detections")
            continue
        obj_lines = []
        for d in dets:
            label = d.get("label", "?")
            track_id = d.get("track_id", d.get("id", "?"))
            cx = d.get("centroid_norm", {}).get("x", 0)
            cy = d.get("centroid_norm", {}).get("y", 0)
            obj_lines.append(f"{track_id}({label}): ({cx:.2f},{cy:.2f})")
        lines.append(f"Frame {frame_id} (t={ts}s): {', '.join(obj_lines)}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def ask(
    question: str,
    *,
    detections: Optional[list[dict]] = None,
    frame_history: Optional[list[dict]] = None,
    image_path: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
) -> GemmaResponse:
    """Ask a question about structured detection data via remote Gemma 4.

    Args:
        question: Natural-language question about the detections or footage.
        detections: List of detection dicts from SAM 3.1 or Falcon Perception.
        frame_history: For tracking — list of frames with detections.
        image_path: Not yet supported remotely — include as text description.
        max_tokens: Max output tokens.
        temperature: Sampling temperature.
        endpoint: OpenAI-compatible API endpoint URL.
        model: Model ID to use.

    Returns:
        GemmaResponse with answer text and timing stats.
    """
    sections = []
    if detections:
        sections.append(f"## Detections\n{_serialize_detections(detections)}")
    if frame_history:
        sections.append(f"## Frame tracking data\n{_serialize_frame_history(frame_history)}")
    if image_path:
        sections.append(f"## Image\n(image at {image_path} — analyze if provided)")
    sections.append(f"## Question\n{question}")
    prompt_text = "\n\n".join(sections)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]

    t0 = time.perf_counter()
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to Gemma 4 server at {endpoint}: {e}") from e

    decode_ms = (time.perf_counter() - t0) * 1000
    usage = result.get("usage", {})
    choice = result.get("choices", [{}])[0]
    message = choice.get("message", {})
    text = message.get("content", "")

    return GemmaResponse(
        text=text,
        stats=GemmaStats(
            prompt_tokens=usage.get("input_tokens", 0),
            generation_tokens=usage.get("output_tokens", 0),
            prompt_tps=usage.get("prompt_tps", 0),
            generation_tps=usage.get("generation_tps", 0),
            decode_ms=round(decode_ms, 1),
        ),
    )


def generate_report(
    summary_text: str,
    *,
    report_type: str = "field",
    max_tokens: int = 768,
    temperature: float = 0.7,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
) -> GemmaResponse:
    """Generate a written field report via remote Gemma 4.

    Args:
        summary_text: Structured or free-text description of analysis results.
        report_type: "field" (detailed), "brief" (one paragraph), "json" (structured).
        max_tokens: Max output tokens.
        temperature: Sampling temperature.
        endpoint: OpenAI-compatible API endpoint URL.
        model: Model ID to use.

    Returns:
        GemmaResponse with report text and timing stats.
    """
    styles = {
        "field": (
            "Write a detailed field report a rancher or farmer can act on. "
            "Include: overview, key findings, animals/areas of concern with severity, "
            "and recommended actions. Be specific about locations, counts, and urgency."
        ),
        "brief": (
            "Write a one-paragraph summary suitable for a text message or phone call "
            "to the farm manager. Include the most critical finding."
        ),
        "json": (
            "Write a structured JSON report with fields: overview (string), "
            "findings (list of {severity: string, description: string, location: string}), "
            "and actions (list of string). Output ONLY the JSON."
        ),
    }
    style = styles.get(report_type, styles["field"])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"## Analysis summary\n{summary_text}\n\n## Task\n{style}"},
    ]

    t0 = time.perf_counter()
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to Gemma 4 server at {endpoint}: {e}") from e

    decode_ms = (time.perf_counter() - t0) * 1000
    usage = result.get("usage", {})
    choice = result.get("choices", [{}])[0]
    message = choice.get("message", {})
    text = message.get("content", "")

    return GemmaResponse(
        text=text,
        stats=GemmaStats(
            prompt_tokens=usage.get("input_tokens", 0),
            generation_tokens=usage.get("output_tokens", 0),
            prompt_tps=usage.get("prompt_tps", 0),
            generation_tps=usage.get("generation_tps", 0),
            decode_ms=round(decode_ms, 1),
        ),
    )


def gemma_available() -> bool:
    """Check if the remote Gemma 4 server is reachable."""
    try:
        req = urllib.request.Request(
            "http://100.72.41.118:8080/v1/models",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("id") for m in data.get("data", [])]
            return any("gemma" in m.lower() for m in models)
    except Exception:
        return False


def test_remote_connection() -> dict:
    """Smoke test the remote Gemma 4 server. Returns status dict."""
    try:
        req = urllib.request.Request(
            "http://100.72.41.118:8080/v1/models",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("id") for m in data.get("data", [])]
            return {"status": "connected", "models": models}
    except Exception as e:
        return {"status": "error", "message": str(e)}
