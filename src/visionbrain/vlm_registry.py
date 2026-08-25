"""Hot-swappable local vision-language models for ask/report — one registry.

The single place that names which multimodal checkpoints answer questions and
write reports. Any frontend (Ground Control console, bridge cockpit, Scout)
selects by key; the checkpoint swap takes effect on the next generate.

Keys:
  gemma  — mlx-community/gemma-4-e2b-it-4bit — careful reader (~3GB, ~2.5s/ask)
  lfm    — LiquidAI/LFM2.5-VL-450M-MLX-4bit — the cheap one (~0.6GB, ~0.4s/ask);
           identical checkpoint to the LFM grounding engine, so when both are
           active they share ONE resident copy via model_host.
  lfm3b  — LiquidAI/LFM2.5-VL-3B-MLX-4bit — mid-tier: noticeably sharper
           grounding/reasoning than 450M at still-frugal cost (~2.2GB).

Checkpoints load through ``model_host.HOST`` so co-resident callers never
duplicate weights. Requires mlx_vlm >= 0.6.1 for Gemma 4 KV-sharing weights,
and < 0.6.4 which drops SAM 3.1 support.

Also hosts the shared detection-line formatter used inside ask/report prompts:
one plain-language rendering of detections for humans and models alike.
"""

from __future__ import annotations

import logging
import os
import threading

from .model_host import HOST

log = logging.getLogger("visionbrain.vlm_registry")

MODELS = {
    "gemma": "mlx-community/gemma-4-e2b-it-4bit",
    "lfm": "LiquidAI/LFM2.5-VL-450M-MLX-4bit",
    "lfm3b": "LiquidAI/LFM2.5-VL-3B-MLX-4bit",
}
DEFAULT_KEY = "gemma"

# VB_VLM_MODEL pins a raw checkpoint id outside MODELS — escape hatch, not
# the main path; a pin cannot be switched away from.
_pinned = os.environ.get("VB_VLM_MODEL")
DEFAULT_MODEL = _pinned or MODELS[DEFAULT_KEY]

ASK_MAX_TOKENS = 220
REPORT_MAX_TOKENS = 448
TEMPERATURE = 0.3
# LFM2.5-VL's card recommends min_p=0.15 + repetition_penalty=1.05; Gemma 4
# E2B tolerates them and gains the same resistance to repeated tokens.
MIN_P = 0.15
REPETITION_PENALTY = 1.05

SYSTEM_PROMPT = (
    "You are the AI copilot of a drone camera system. You are looking at the "
    "live camera frame. Describe only what is actually visible. Do not invent "
    "objects, people, weather, or places you cannot see.\n"
    "Detector output, when provided, covers ONLY the labels it was asked to "
    "find — it is never a full inventory of the scene. Prefer its counts when "
    "answering about those specific labels, but keep describing anything else "
    "you can see, and never claim the scene contains nothing beyond them.\n"
    "Never expand a detector count into a list of individual objects. If it "
    "reports 2x person, you may say there are two people; you may NOT invent "
    "a third, nor assign positions, numbers or descriptions to each one unless "
    "the detector listed them itself."
)

_lock = threading.Lock()
_model = None
_processor = None
_config = None
_loaded_id: str | None = None
_wanted_id: str = DEFAULT_MODEL


def available() -> bool:
    """True when the local MLX VLM stack is importable."""
    try:
        import mlx_vlm  # noqa: F401

        return True
    except Exception:
        return False


def current_model() -> str:
    """The checkpoint the next ask/report will use — not necessarily loaded yet."""
    return _wanted_id


def current_key() -> str:
    """MODELS key for the wanted checkpoint, or "" for a raw pin."""
    for key, model_id in MODELS.items():
        if model_id == _wanted_id:
            return key
    return ""


def set_model(key: str) -> str:
    """Select a VLM by MODELS key. Cheap: the swap happens on next generate."""
    global _wanted_id
    model_id = MODELS.get(key)
    if model_id is None:
        raise ValueError(f"unknown VLM {key!r}; expected one of {sorted(MODELS)}")
    _wanted_id = model_id
    return model_id


def _load_checkpoint(target: str):
    from mlx_vlm.utils import load, load_config

    model, processor = load(target)
    config = load_config(target)
    return model, processor, config


def _ensure_loaded():
    global _model, _processor, _config, _loaded_id
    with _lock:
        if _loaded_id is not None and _loaded_id != _wanted_id:
            log.info("switching VLM %s -> %s", _loaded_id, _wanted_id)
            # Frees only once every holder released — an engine holding the
            # same checkpoint keeps it resident across the switch.
            HOST.release(_loaded_id)
            _model = _processor = _config = None
            _loaded_id = None
        if _model is None:
            target = _wanted_id
            _model, _processor, _config = HOST.acquire(
                target, lambda: _load_checkpoint(target)
            )
            _loaded_id = target
        return _model, _processor, _config


def _generate(user_text: str, image, max_tokens: int) -> str:
    model, processor, config = _ensure_loaded()
    from mlx_vlm.generate import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    images = [image] if image is not None else []
    prompt = apply_chat_template(
        processor,
        config,
        f"{SYSTEM_PROMPT}\n\n{user_text}",
        num_images=len(images),
    )
    kwargs = {
        "max_tokens": max_tokens,
        "verbose": False,
        "temperature": TEMPERATURE,
        "min_p": MIN_P,
        "repetition_penalty": REPETITION_PENALTY,
    }
    result = generate(model, processor, prompt, image=images or None, **kwargs)
    return str(getattr(result, "text", result)).strip()


def position_label(x: float, y: float) -> str:
    """Normalized centroid -> plain-language position ("lower left of frame")."""
    col = "left" if x < 0.4 else "right" if x > 0.6 else ""
    row = "upper" if y < 0.4 else "lower" if y > 0.6 else ""
    if not col and not row:
        return "center of frame"
    if not row:
        return f"{col} side of frame"
    if not col:
        return f"{row} frame"
    return f"{row} {col} of frame"


def format_detection_lines(detections: list[dict]) -> str:
    """One "- label, NN% confidence, position [source]" line per detection."""
    lines = []
    for det in detections:
        c = det.get("centroid_norm", {})
        pct = round(det.get("score", 0.0) * 100)
        pos = position_label(c.get("x", 0.5), c.get("y", 0.5))
        src = det.get("source")
        tag = f" [{src}]" if src else ""
        lines.append(f"- {det.get('label', 'object')}, {pct}% confidence, {pos}{tag}")
    return "\n".join(lines)


def ask(
    question: str,
    detections: list[dict] | None = None,
    prompts: list[str] | None = None,
    image=None,
) -> str:
    """Answer a question about the current frame (+ optional detector output)."""
    if image is None:
        return (
            "No live frame available to look at. "
            "Check that a source is streaming before asking about the view."
        )

    sections = []
    dets = detections or []
    if dets:
        counts: dict[str, int] = {}
        for det in dets:
            label = det.get("label", "object")
            counts[label] = counts.get(label, 0) + 1
        totals = ", ".join(f"{n}x {label}" for label, n in counts.items())
        sections.append(f"## Detector counts for armed labels only\n{totals}")
        sections.append(f"## Detector output\n{format_detection_lines(dets)}")
    armed = [p for p in (prompts or []) if p and str(p).strip()]
    if armed:
        sections.append(f"## Armed prompts\n{', '.join(armed)}")
    sections.append(f"## Question\n{question}\n\nAnswer in at most 4 short sentences.")
    return _generate("\n\n".join(sections), image, ASK_MAX_TOKENS)


def generate_report(summary_text: str, report_type: str = "field", image=None) -> str:
    """Write a field/brief report on the current frame + detector summary."""
    summary = (summary_text or "").strip()
    # An unframed summary reads as a prose hint, and both Gemma and LFM answer
    # it by inventing per-object breakdowns. Stating it is verbatim and
    # exhaustive stops the elaboration.
    if summary:
        detector_block = (
            "## Detector summary (verbatim — authoritative)\n"
            f"{summary}\n\n"
            "Those counts are exact and complete for the labels the detector was "
            "given. Repeat them as written. Do not break a count down into "
            "individual objects, and do not add labels the detector did not report."
        )
    else:
        detector_block = (
            "## Detector summary\n"
            "There is no detector output for this frame. Do not mention a "
            "detector, a field report, or any measured count — none exists. "
            "Describe only what you can see. If you give a number, say plainly "
            "that it is your own estimate from the image."
        )
    user_text = (
        f"Write a {report_type} report on what this drone is currently looking at. "
        f"Be factual and brief; describe the scene as well as the detector output.\n\n"
        f"{detector_block}"
    )
    return _generate(user_text, image, REPORT_MAX_TOKENS)
