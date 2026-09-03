"""Gemma 4 inference — consolidated reasoning layer for VisionBrain.

Handles three backend options (auto-selected by availability):
  1. Ollama (gemma4:e2b, 7.2GB) — local, preferred
  2. Remote server (mlx-community/gemma-4-26b-a4b-it-4bit) — fallback
  3. Local MLX (gemma-4-26b-a4b-it-4bit) — last resort, requires ~32GB RAM

Role: reasoning on top of SAM 3.1 + Falcon Perception outputs.
Given structured detections/masks, Gemma 4 answers questions and
generates field reports — the "brain" layer.

Usage:
    from visionbrain.gemma_inference import available_backend, ask, generate_report, gemma_available
    backend = available_backend()  # 'ollama' | 'remote' | 'local' | None
    if gemma_available():
        resp = ask("Which vehicles are parked in restricted areas?", detections=frame_data)
        report = generate_report(summary_text, report_type="field")
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import loader

# ──────────────────────────────────────────────────────────────────────────────
# Configuration per backend
# ──────────────────────────────────────────────────────────────────────────────

OLLAMA_ENDPOINT = "http://localhost:11434/v1/chat/completions"
OLLAMA_MODEL = "gemma4:e2b"

REMOTE_ENDPOINT = "http://100.72.41.118:8080/v1/chat/completions"
REMOTE_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"

LOCAL_HF_REPO = "mlx-community/gemma-4-26b-a4b-it-4bit"

# Stop tokens for Ollama to prevent multi-turn output
OLLAMA_STOP_TOKENS = ["<end_of_turn>", "<eos>"]

# Shared system prompt
SYSTEM_PROMPT = (
    "You are a visual intelligence assistant helping operators analyze drone and camera "
    "footage. You have access to structured object detection "
    "data from vision AI models: bounding boxes with confidence scores, pixel-level "
    "segmentation masks with area fractions, object tracks across video frames "
    "(track IDs, centroid positions), and class labels (e.g. 'person', 'vehicle', 'building', "
    "'animal'). "
    "Be specific, practical, and actionable. Focus on: activity and behavior patterns "
    "(unusual movement, loitering, isolation), site and infrastructure conditions "
    "(damage, obstruction, wear), terrain and environmental indicators, and anomalies requiring human "
    "attention. Keep reports concise but detailed enough to act on."
)


# ──────────────────────────────────────────────────────────────────────────────
# Backend detection
# ──────────────────────────────────────────────────────────────────────────────

def available_backend() -> Optional[str]:
    """Check which Gemma backend is available, in priority order.

    Returns:
        'ollama'  — Ollama server with gemma4:e2b
        'remote'  — Remote Gemma 4 server reachable
        'local'   — Local MLX weights cached
        None     — No backend available
    """
    # Priority 1: Ollama
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            names = [m.get("name", "") for m in data.get("models", [])]
            if OLLAMA_MODEL in names or "gemma4:latest" in names:
                return "ollama"
    except Exception:
        pass

    # Priority 2: Remote server
    try:
        req = urllib.request.Request(
            f"{REMOTE_ENDPOINT.rsplit('/v1', 1)[0]}/v1/models",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("id") for m in data.get("data", [])]
            if any("gemma" in m.lower() for m in models if m):
                return "remote"
    except Exception:
        pass

    # Priority 3: Local MLX weights
    if loader.gemma4_record().is_cached:
        return "local"

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Availability
# ──────────────────────────────────────────────────────────────────────────────

def gemma_available() -> bool:
    """True if at least one Gemma 4 backend is available."""
    return available_backend() is not None


def test_connection() -> dict:
    """Smoke test the active backend. Returns status dict."""
    backend = available_backend()
    if backend == "ollama":
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/tags",
                headers={"Content-Type": "application/json"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                names = [m.get("name", "") for m in data.get("models", [])]
                return {
                    "backend": "ollama",
                    "status": "connected",
                    "ollama_models": names,
                    "gemma_ready": OLLAMA_MODEL in names or "gemma4:latest" in names,
                }
        except Exception as e:
            return {"backend": "ollama", "status": "error", "message": str(e)}

    elif backend == "remote":
        try:
            req = urllib.request.Request(
                f"{REMOTE_ENDPOINT.rsplit('/v1', 1)[0]}/v1/models",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("id") for m in data.get("data", [])]
                return {"backend": "remote", "status": "connected", "models": models}
        except Exception as e:
            return {"backend": "remote", "status": "error", "message": str(e)}

    elif backend == "local":
        return {"backend": "local", "status": "connected", "note": "local MLX weights"}

    return {"backend": None, "status": "unavailable"}


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
# Model cache (for local MLX backend)
# ──────────────────────────────────────────────────────────────────────────────

_gemma_cache: dict = {}


def _ensure_local_gemma(kv_bits: float = 3.5, kv_quant_scheme: str = "turboquant") -> dict:
    """Load local Gemma 4 + processor once. Returns cache dict."""
    if "model" not in _gemma_cache:
        from mlx_vlm.utils import load as vlm_load

        # mlx_vlm 0.4.4 cannot load gemma-4's quantized per_layer_model_projection
        # (ScaledLinear lacks to_quantized) — see visionbrain.mlx_compat.
        from .mlx_compat import apply_all

        apply_all()
        print(f"Loading Gemma 4 26B ({LOCAL_HF_REPO}) via MLX...")
        t0 = time.perf_counter()
        model, processor = vlm_load(LOCAL_HF_REPO)
        print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

        _gemma_cache["model"] = model
        _gemma_cache["processor"] = processor
        _gemma_cache["kv_config"] = {
            "kv_bits": kv_bits,
            "kv_quant_scheme": kv_quant_scheme,
        }

    return _gemma_cache


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _serialize_detections(detections: list[dict], compact: bool = True) -> str:
    """Serialize detection data for Gemma.

    Args:
        detections: list of detection dicts from SAM 3.1 or Falcon Perception.
        compact: if True, use slash-delimited format (gemma4:e2b compatible).
                 if False, use pipe-delimited format (local/remote compatible).
    """
    if not detections:
        return "No detections available."

    if compact:
        # Compact one-line format — avoids pipe character issue with gemma4:e2b
        parts = []
        for d in detections:
            label = d.get("label", "?")
            score = d.get("score", 0)
            track_id = d.get("track_id", d.get("id", "?"))
            cx = d.get("centroid_norm", {}).get("x", 0)
            cy = d.get("centroid_norm", {}).get("y", 0)
            parts.append(f"{track_id}/{label}/{score:.2f}/({cx:.2f},{cy:.2f})")
        return "[" + "], [".join(parts) + "]"
    else:
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


def _serialize_frame_history(frames: list[dict], compact: bool = True) -> str:
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
        if compact:
            obj_parts = []
            for d in dets:
                label = d.get("label", "?")
                track_id = d.get("track_id", d.get("id", "?"))
                cx = d.get("centroid_norm", {}).get("x", 0)
                cy = d.get("centroid_norm", {}).get("y", 0)
                obj_parts.append(f"{track_id}/{label}/({cx:.2f},{cy:.2f})")
            lines.append(f"Frame {frame_id} (t={ts}s): [{', '.join(obj_parts)}]")
        else:
            obj_lines = []
            for d in dets:
                label = d.get("label", "unknown")
                track_id = d.get("track_id", d.get("id", "?"))
                cx = d.get("centroid_norm", {}).get("x", 0)
                cy = d.get("centroid_norm", {}).get("y", 0)
                obj_lines.append(f"{track_id}({label}): ({cx:.2f},{cy:.2f})")
            lines.append(f"Frame {frame_id} (t={ts}s): {', '.join(obj_lines)}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Ollama backend
# ──────────────────────────────────────────────────────────────────────────────

def _ollama_ask(
    question: str,
    *,
    detections: Optional[list[dict]] = None,
    frame_history: Optional[list[dict]] = None,
    image_path: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> GemmaResponse:
    """Call Ollama gemma4:e2b."""
    sections = []
    if detections:
        sections.append(f"## Detections\n{_serialize_detections(detections, compact=True)}")
    if frame_history:
        sections.append(f"## Frame tracking data\n{_serialize_frame_history(frame_history, compact=True)}")
    if image_path:
        sections.append(f"## Image\n(image at {image_path} — describe if useful)")
    sections.append(f"## Question\n{question}")
    prompt_text = "\n\n".join(sections)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]

    t0 = time.perf_counter()
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": OLLAMA_STOP_TOKENS,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to Ollama at {OLLAMA_ENDPOINT}: {e}") from e

    decode_ms = (time.perf_counter() - t0) * 1000
    usage = result.get("usage", {})
    choice = result.get("choices", [{}])[0]
    message = choice.get("message", {})
    text = message.get("content", "")

    return GemmaResponse(
        text=text,
        stats=GemmaStats(
            prompt_tokens=usage.get("prompt_tokens", 0),
            generation_tokens=usage.get("completion_tokens", 0),
            prompt_tps=round(usage.get("prompt_tokens", 0) / (decode_ms / 1000), 1)
                       if decode_ms > 0 else 0.0,
            generation_tps=round(usage.get("completion_tokens", 0) / (decode_ms / 1000), 1)
                          if decode_ms > 0 else 0.0,
            decode_ms=round(decode_ms, 1),
        ),
    )


def _ollama_generate_report(
    summary_text: str,
    *,
    report_type: str = "field",
    max_tokens: int = 768,
    temperature: float = 0.7,
) -> GemmaResponse:
    """Generate field report via Ollama."""
    styles = {
        "field": (
            "Write a detailed field report an operator can act on. "
            "Include: overview, key findings, objects/areas of concern with severity, "
            "and recommended actions. Be specific about locations, counts, and urgency."
        ),
        "brief": (
            "Write a one-paragraph summary suitable for a text message or phone call "
            "to the site manager. Include the most critical finding."
        ),
        "json": (
            "Write a structured JSON report with fields: overview (string), "
            "findings (list of {severity: string, description: string, location: string}), "
            "and actions (list of string). Output ONLY the JSON, no markdown."
        ),
    }
    style = styles.get(report_type, styles["field"])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"## Analysis summary\n{summary_text}\n\n## Task\n{style}"},
    ]

    t0 = time.perf_counter()
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": OLLAMA_STOP_TOKENS,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to Ollama at {OLLAMA_ENDPOINT}: {e}") from e

    decode_ms = (time.perf_counter() - t0) * 1000
    usage = result.get("usage", {})
    choice = result.get("choices", [{}])[0]
    message = choice.get("message", {})
    text = message.get("content", "")

    return GemmaResponse(
        text=text,
        stats=GemmaStats(
            prompt_tokens=usage.get("prompt_tokens", 0),
            generation_tokens=usage.get("completion_tokens", 0),
            prompt_tps=round(usage.get("prompt_tokens", 0) / (decode_ms / 1000), 1)
                       if decode_ms > 0 else 0.0,
            generation_tps=round(usage.get("completion_tokens", 0) / (decode_ms / 1000), 1)
                          if decode_ms > 0 else 0.0,
            decode_ms=round(decode_ms, 1),
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Remote backend
# ──────────────────────────────────────────────────────────────────────────────

def _remote_ask(
    question: str,
    *,
    detections: Optional[list[dict]] = None,
    frame_history: Optional[list[dict]] = None,
    image_path: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> GemmaResponse:
    """Call remote Gemma 4 server."""
    sections = []
    if detections:
        sections.append(f"## Detections\n{_serialize_detections(detections, compact=False)}")
    if frame_history:
        sections.append(f"## Frame tracking data\n{_serialize_frame_history(frame_history, compact=False)}")
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
        "model": REMOTE_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        REMOTE_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to Gemma 4 server at {REMOTE_ENDPOINT}: {e}") from e

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


def _remote_generate_report(
    summary_text: str,
    *,
    report_type: str = "field",
    max_tokens: int = 768,
    temperature: float = 0.7,
) -> GemmaResponse:
    """Generate field report via remote Gemma 4."""
    styles = {
        "field": (
            "Write a detailed field report an operator can act on. "
            "Include: overview, key findings, objects/areas of concern with severity, "
            "and recommended actions. Be specific about locations, counts, and urgency."
        ),
        "brief": (
            "Write a one-paragraph summary suitable for a text message or phone call "
            "to the site manager. Include the most critical finding."
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
        "model": REMOTE_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        REMOTE_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to Gemma 4 server at {REMOTE_ENDPOINT}: {e}") from e

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


# ──────────────────────────────────────────────────────────────────────────────
# Local MLX backend
# ──────────────────────────────────────────────────────────────────────────────

def _local_ask(
    question: str,
    *,
    detections: Optional[list[dict]] = None,
    frame_history: Optional[list[dict]] = None,
    image_path: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    kv_bits: float = 3.5,
    kv_quant_scheme: str = "turboquant",
) -> GemmaResponse:
    """Ask via local MLX Gemma 4."""
    from mlx_vlm.generate import generate

    cache = _ensure_local_gemma(kv_bits=kv_bits, kv_quant_scheme=kv_quant_scheme)
    model = cache["model"]
    processor = cache["processor"]

    sections = []
    if detections:
        sections.append(f"## Detections\n{_serialize_detections(detections, compact=False)}")
    if frame_history:
        sections.append(f"## Frame tracking data\n{_serialize_frame_history(frame_history, compact=False)}")
    sections.append(f"## Question\n{question}")
    prompt_text = "\n\n".join(sections)

    image_paths = [image_path] if image_path and Path(image_path).exists() else None

    t0 = time.perf_counter()
    result = generate(
        model,
        processor,
        prompt_text,
        image=image_paths,
        max_tokens=max_tokens,
        temperature=temperature,
        **cache["kv_config"],
    )
    decode_ms = (time.perf_counter() - t0) * 1000

    return GemmaResponse(
        text=result.text,
        stats=GemmaStats(
            prompt_tokens=result.prompt_tokens,
            generation_tokens=result.generation_tokens,
            prompt_tps=round(result.prompt_tps, 1),
            generation_tps=round(result.generation_tps, 1),
            decode_ms=round(decode_ms, 1),
        ),
    )


def _local_generate_report(
    summary_text: str,
    *,
    report_type: str = "field",
    max_tokens: int = 768,
    temperature: float = 0.7,
    kv_bits: float = 3.5,
    kv_quant_scheme: str = "turboquant",
) -> GemmaResponse:
    """Generate report via local MLX Gemma 4."""
    from mlx_vlm.generate import generate

    cache = _ensure_local_gemma(kv_bits=kv_bits, kv_quant_scheme=kv_quant_scheme)
    model = cache["model"]
    processor = cache["processor"]

    styles = {
        "field": (
            "Write a detailed field report an operator can act on. "
            "Include: overview, key findings, objects/areas of concern with severity, "
            "and recommended actions. Be specific about locations, counts, and urgency."
        ),
        "brief": (
            "Write a one-paragraph summary suitable for a text message or phone call "
            "to the site manager. Include the most critical finding."
        ),
        "json": (
            "Write a structured JSON report with fields: overview (string), "
            "findings (list of {severity: string, description: string, location: string}), "
            "and actions (list of string). Output ONLY the JSON, no markdown."
        ),
    }
    style = styles.get(report_type, styles["field"])

    prompt = f"## Analysis summary\n{summary_text}\n\n## Task\n{style}"

    t0 = time.perf_counter()
    result = generate(
        model,
        processor,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        **cache["kv_config"],
    )
    decode_ms = (time.perf_counter() - t0) * 1000

    return GemmaResponse(
        text=result.text,
        stats=GemmaStats(
            prompt_tokens=result.prompt_tokens,
            generation_tokens=result.generation_tokens,
            prompt_tps=round(result.prompt_tps, 1),
            generation_tps=round(result.generation_tps, 1),
            decode_ms=round(decode_ms, 1),
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API — auto-selects backend
# ──────────────────────────────────────────────────────────────────────────────

def ask(
    question: str,
    *,
    detections: Optional[list[dict]] = None,
    frame_history: Optional[list[dict]] = None,
    image_path: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    kv_bits: float = 3.5,
    kv_quant_scheme: str = "turboquant",
) -> GemmaResponse:
    """Ask a question about structured detection data or an image.

    Auto-selects the best available backend (Ollama → remote → local MLX).

    Args:
        question: Natural-language question about the detections or image.
        detections: List of detection dicts from SAM 3.1 or Falcon Perception.
        frame_history: For tracking queries — list of frames with detections.
        image_path: Optional image for multimodal reasoning (local MLX only).
        max_tokens: Max output tokens. Must be >= 200 for structured responses (Ollama).
        temperature: Sampling temperature (0 = deterministic).
        kv_bits: KV cache quantization bits (local MLX only).
        kv_quant_scheme: KV quantization scheme (local MLX only).

    Returns:
        GemmaResponse with answer text and timing stats.
    """
    backend = available_backend()

    if backend == "ollama":
        return _ollama_ask(
            question,
            detections=detections,
            frame_history=frame_history,
            image_path=image_path,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "remote":
        return _remote_ask(
            question,
            detections=detections,
            frame_history=frame_history,
            image_path=image_path,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "local":
        return _local_ask(
            question,
            detections=detections,
            frame_history=frame_history,
            image_path=image_path,
            max_tokens=max_tokens,
            temperature=temperature,
            kv_bits=kv_bits,
            kv_quant_scheme=kv_quant_scheme,
        )
    else:
        raise RuntimeError(
            "No Gemma backend available. Install Ollama, or ensure the remote server "
            "is reachable, or cache local Gemma 4 weights."
        )


def generate_report(
    summary_text: str,
    *,
    report_type: str = "field",
    max_tokens: int = 768,
    temperature: float = 0.7,
    kv_bits: float = 3.5,
    kv_quant_scheme: str = "turboquant",
) -> GemmaResponse:
    """Generate a written field report from analysis summary data.

    Auto-selects the best available backend (Ollama → remote → local MLX).

    Args:
        summary_text: Structured or free-text description of analysis results.
        report_type: "field" (detailed actionable report), "brief" (one paragraph),
                     or "json" (structured JSON).
        max_tokens: Max output tokens.
        temperature: Sampling temperature.
        kv_bits: KV cache quantization bits (local MLX only).
        kv_quant_scheme: KV quantization scheme (local MLX only).

    Returns:
        GemmaResponse with report text and timing stats.
    """
    backend = available_backend()

    if backend == "ollama":
        return _ollama_generate_report(
            summary_text,
            report_type=report_type,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "remote":
        return _remote_generate_report(
            summary_text,
            report_type=report_type,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "local":
        return _local_generate_report(
            summary_text,
            report_type=report_type,
            max_tokens=max_tokens,
            temperature=temperature,
            kv_bits=kv_bits,
            kv_quant_scheme=kv_quant_scheme,
        )
    else:
        raise RuntimeError(
            "No Gemma backend available. Install Ollama, or ensure the remote server "
            "is reachable, or cache local Gemma 4 weights."
        )


def unload_gemma() -> None:
    """Release Gemma 4 from model cache to free RAM.

    Only affects local MLX backend — Ollama and remote backends have no local
    model to unload.
    """
    global _gemma_cache
    _gemma_cache.clear()
    try:
        import mlx.core as mx
        mx.metal.reset()
    except ImportError:
        pass
    print("Gemma 4 unloaded, memory freed.")