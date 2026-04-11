"""Gemma 4 26B inference — reasoning layer for VisionBrain.

Model:  mlx-community/gemma-4-26b-a4b-it-4bit
       Google's 26B MoE (~3.8B active params/token), 4-bit quantized.
       Native MLX — no PyTorch, no CUDA.

Role: reasoning on top of SAM 3.1 + Falcon Perception outputs.
Given structured detections/masks, Gemma 4 answers questions and
generates field reports — the "brain" layer.

Usage:
    from visionbrain.gemma_inference import ask, generate_report, gemma_available
    resp = ask("Which cattle are isolated from the herd?", detections=frame_data)
    report = generate_report(summary_text, report_type="field")
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

GEMMA4_HF_REPO = "mlx-community/gemma-4-26b-a4b-it-4bit"

# ──────────────────────────────────────────────────────────────────────────────
# Availability
# ──────────────────────────────────────────────────────────────────────────────

def gemma_available() -> bool:
    """True if Gemma 4 weights are cached locally."""
    cached = Path.home() / ".cache" / "huggingface" / "hub" / "models--mlx-community--gemma-4-26b-a4b-it-4bit"
    if not cached.exists():
        return False
    size_gb = sum(f.stat().st_size for f in cached.rglob("*") if f.is_file()) / (1024 ** 3)
    return size_gb > 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Model cache (loaded once per process)
# ──────────────────────────────────────────────────────────────────────────────

_gemma_cache: dict = {}


def _ensure_gemma(kv_bits: float = 3.5, kv_quant_scheme: str = "turboquant") -> dict:
    """Load Gemma 4 + processor once. Returns cache dict."""
    if "model" not in _gemma_cache:
        from mlx_vlm.utils import load as vlm_load

        print(f"Loading Gemma 4 26B ({GEMMA4_HF_REPO})...")
        t0 = time.perf_counter()
        model, processor = vlm_load(GEMMA4_HF_REPO)
        print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

        _gemma_cache["model"] = model
        _gemma_cache["processor"] = processor
        _gemma_cache["kv_config"] = {
            "kv_bits": kv_bits,
            "kv_quant_scheme": kv_quant_scheme,
        }

    return _gemma_cache


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

def _build_system_prompt() -> str:
    return (
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


def _serialize_detections(detections: list[dict]) -> str:
    if not detections:
        return "No detections available."
    lines = []
    for d in detections:
        label = d.get("label", d.get("class", "unknown"))
        score = d.get("score", d.get("confidence", 0))
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
        return "No frame history available."
    lines = []
    for frame in frames:
        frame_id = frame.get("frame_index", "?")
        dets = frame.get("detections", [])
        if not dets:
            lines.append(f"Frame {frame_id}: no detections")
            continue
        obj_lines = []
        for d in dets:
            label = d.get("label", "unknown")
            track_id = d.get("track_id", d.get("id", "?"))
            cx = d.get("centroid_norm", {}).get("x", 0)
            cy = d.get("centroid_norm", {}).get("y", 0)
            obj_lines.append(f"{track_id}({label}): ({cx:.2f},{cy:.2f})")
        lines.append(f"Frame {frame_id}: {', '.join(obj_lines)}")
    return "\n".join(lines)


def _detections_to_text(detections: list[dict], preamble: str = "") -> str:
    """Compact one-line-per-detection format for longer histories."""
    parts = [preamble] if preamble else []
    for d in detections:
        label = d.get("label", "?")
        track_id = d.get("track_id", d.get("id", "?"))
        cx = d.get("centroid_norm", {}).get("x", 0)
        cy = d.get("centroid_norm", {}).get("y", 0)
        score = d.get("score", 0)
        parts.append(f"[{track_id}]{label}({cx:.2f},{cy:.2f})@{score:.2f}")
    return " ".join(parts)


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
    kv_bits: float = 3.5,
    kv_quant_scheme: str = "turboquant",
) -> GemmaResponse:
    """Ask a question about structured detection data or an image.

    Args:
        question: Natural-language question about the detections or image.
        detections: List of detection dicts from SAM 3.1 or Falcon Perception.
                    Expected fields: label, score, centroid_norm {x,y}, bbox_norm,
                    area_fraction, track_id/id.
        frame_history: For tracking queries — list of frames with detections,
                       each having frame_index, timestamp, detections.
        image_path: Optional image for multimodal reasoning.
        max_tokens: Max output tokens.
        temperature: Sampling temperature (0 = deterministic).
        kv_bits: KV cache quantization bits (3.5 = turboquant sweet spot).
        kv_quant_scheme: KV quantization scheme.

    Returns:
        GemmaResponse with answer text and timing stats.
    """
    from mlx_vlm.generate import generate

    cache = _ensure_gemma(kv_bits=kv_bits, kv_quant_scheme=kv_quant_scheme)
    model = cache["model"]
    processor = cache["processor"]

    # Build text prompt
    sections = []
    if detections:
        sections.append(f"## Detections\n{_serialize_detections(detections)}")
    if frame_history:
        sections.append(f"## Frame tracking data\n{_serialize_frame_history(frame_history)}")
    sections.append(f"## Question\n{question}")
    prompt_text = "\n\n".join(sections)

    # Image handling — pass to generate if provided
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

    Args:
        summary_text: Structured or free-text description of analysis results.
        report_type: "field" (detailed actionable report), "brief" (one paragraph),
                     or "json" (structured JSON).
        max_tokens: Max output tokens.
        temperature: Sampling temperature.

    Returns:
        GemmaResponse with report text and timing stats.
    """
    from mlx_vlm.generate import generate

    cache = _ensure_gemma(kv_bits=kv_bits, kv_quant_scheme=kv_quant_scheme)
    model = cache["model"]
    processor = cache["processor"]

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


def unload_gemma() -> None:
    """Release Gemma 4 from model cache to free ~15.6GB RAM."""
    _gemma_cache.clear()
    try:
        import mlx.core as mx
        mx.metal.reset()
    except ImportError:
        pass
    print("Gemma 4 unloaded, memory freed.")
