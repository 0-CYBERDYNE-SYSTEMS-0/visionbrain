"""Falcon Perception inference — MLX path (no torch required).

This module wraps the Falcon-Perception MLX batch_inference pipeline that
already works on this machine. It is READ-ONLY with respect to the
Falcon-Perception repo — it imports and calls, never modifies.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from .loader import fp_module_path, falcon_perception_record

# ── Add Falcon-Perception to the Python path (read-only) ──────────────────────
_FALCON_ROOT = fp_module_path().parent
if str(_FALCON_ROOT) not in sys.path:
    sys.path.insert(0, str(_FALCON_ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MaskResult:
    """A single segmentation mask with metadata."""
    mask_id: int
    centroid_x: float   # normalized 0-1
    centroid_y: float   # normalized 0-1
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    area_fraction: float
    image_region: str   # e.g. "bottom-left", "center"
    rle: dict           # COCO RLE, original resolution (needed for relations)

    def to_dict(self) -> dict:
        return {
            "id": self.mask_id,
            "centroid_norm": {"x": self.centroid_x, "y": self.centroid_y},
            "bbox_norm": {
                "x1": self.bbox_x1, "y1": self.bbox_y1,
                "x2": self.bbox_x2, "y2": self.bbox_y2,
            },
            "area_fraction": self.area_fraction,
            "image_region": self.image_region,
        }


@dataclass
class DetectionResult:
    """A single bounding box detection."""
    label: str
    score: float
    cx: float   # normalized center x
    cy: float   # normalized center y
    h: float    # normalized height
    w: float    # normalized width

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": self.score,
            "cx": self.cx, "cy": self.cy,
            "h": self.h, "w": self.w,
        }


@dataclass
class InferenceStats:
    """Timing and throughput statistics for a single inference run."""
    preprocess_ms: float
    generation_ms: float
    total_ms: float
    prefill_tokens: int
    decoded_tokens: int
    tokens_per_sec: float
    n_masks: int
    n_detections: int


# ──────────────────────────────────────────────────────────────────────────────
# Internal RLE helpers (mirrored from fp_tools.py, no torch needed)
# ──────────────────────────────────────────────────────────────────────────────

def _to_bytes_rle(rle: dict) -> dict:
    out = rle.copy()
    if isinstance(out.get("counts"), str):
        out["counts"] = out["counts"].encode("utf-8")
    return out


def _resize_rle(rle: dict, target_h: int, target_w: int) -> dict:
    from pycocotools import mask as mask_utils
    from PIL import Image
    import numpy as np

    cur_h, cur_w = rle["size"]
    if cur_h == target_h and cur_w == target_w:
        return rle
    binary = mask_utils.decode(_to_bytes_rle(rle))
    resized = np.asfortranarray(
        np.array(
            Image.fromarray(binary).resize((target_w, target_h), Image.NEAREST)
        ).astype(np.uint8)
    )
    new_rle = mask_utils.encode(resized)
    if isinstance(new_rle.get("counts"), bytes):
        new_rle["counts"] = new_rle["counts"].decode("utf-8")
    return new_rle


def _image_region(cx_norm: float, cy_norm: float) -> str:
    h = "left" if cx_norm < 0.33 else ("center" if cx_norm < 0.67 else "right")
    v = "top" if cy_norm < 0.33 else ("middle" if cy_norm < 0.67 else "bottom")
    if v == "middle" and h == "center":
        return "center"
    if v == "middle":
        return h
    return f"{v}-{h}"


# ──────────────────────────────────────────────────────────────────────────────
# Core inference
# ──────────────────────────────────────────────────────────────────────────────

# Cached model + tokenizer (loaded once per process)
_model_cache: dict = {}


def _ensure_model() -> tuple:
    """Load Falcon Perception model + tokenizer once, cache in process memory."""
    if "model" not in _model_cache:
        rec = falcon_perception_record()
        if not rec.can_load:
            raise RuntimeError(
                f"Falcon Perception cannot load: {rec.note}\n"
                f"Cache: {rec.cache_dir} ({rec.disk_gb} GB)"
            )
        from falcon_perception import (
            PERCEPTION_MODEL_ID, load_and_prepare_model
        )
        from falcon_perception.mlx.batch_inference import BatchInferenceEngine

        print(f"Loading Falcon Perception ({PERCEPTION_MODEL_ID})...")
        t0 = time.perf_counter()
        model, tokenizer, model_args = load_and_prepare_model(
            hf_model_id=PERCEPTION_MODEL_ID,
            dtype="float16",
            backend="mlx",
        )
        engine = BatchInferenceEngine(model, tokenizer)
        print(f"  Loaded in {time.perf_counter()-t0:.2f}s")
        _model_cache["model"] = model
        _model_cache["tokenizer"] = tokenizer
        _model_cache["model_args"] = model_args
        _model_cache["engine"] = engine

    return (
        _model_cache["engine"],
        _model_cache["tokenizer"],
        _model_cache["model_args"],
    )


def _postprocess_masks(
    aux_outputs: Any,
    orig_w: int,
    orig_h: int,
    min_dim: int = 256,
    max_dim: int = 1024,
) -> list[MaskResult]:
    """Convert raw aux_outputs from BatchInferenceEngine to MaskResult list."""
    from falcon_perception.data import load_image
    from pycocotools import mask as mask_utils

    masks_rle = []
    if hasattr(aux_outputs, "masks_rle") and aux_outputs.masks_rle:
        masks_rle = list(aux_outputs.masks_rle)
    elif hasattr(aux_outputs, "bboxes_raw"):
        # Detection-only output — no RLE masks
        masks_rle = []

    results: list[MaskResult] = []
    assigned_id = 1

    for raw_rle in masks_rle:
        if not isinstance(raw_rle, dict) or "counts" not in raw_rle:
            continue
        # Resize from inference resolution → original image resolution
        rle_orig = _resize_rle(raw_rle, orig_h, orig_w)
        binary = mask_utils.decode(_to_bytes_rle(rle_orig))
        if binary is None or not binary.any():
            continue

        area = int(binary.sum())
        area_fraction = round(area / (orig_h * orig_w), 4)

        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        if not rows.any() or not cols.any():
            continue

        rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

        cx = round((cmin + cmax + 1) / 2.0 / orig_w, 4)
        cy = round((rmin + rmax + 1) / 2.0 / orig_h, 4)

        results.append(MaskResult(
            mask_id=assigned_id,
            centroid_x=cx,
            centroid_y=cy,
            bbox_x1=round(cmin / orig_w, 4),
            bbox_y1=round(rmin / orig_h, 4),
            bbox_x2=round((cmax + 1) / orig_w, 4),
            bbox_y2=round((rmax + 1) / orig_h, 4),
            area_fraction=area_fraction,
            image_region=_image_region(cx, cy),
            rle=rle_orig,
        ))
        assigned_id += 1

    return results


def _postprocess_detections(
    aux_outputs: Any,
) -> list[DetectionResult]:
    """Extract bounding-box detections from aux_outputs."""
    bboxes = []
    scores = []
    cur = {}

    raw_bboxes = []
    if hasattr(aux_outputs, "bboxes_raw"):
        raw_bboxes = list(aux_outputs.bboxes_raw) if aux_outputs.bboxes_raw else []

    for entry in raw_bboxes:
        if not isinstance(entry, dict):
            continue
        cur.update(entry)
        if all(k in cur for k in ("x", "y", "h", "w")):
            bboxes.append(dict(cur))
            cur = {}

    raw_scores = []
    if hasattr(aux_outputs, "scores") and aux_outputs.scores:
        raw_scores = list(aux_outputs.scores)
    elif hasattr(aux_outputs, "logits") and aux_outputs.logits:
        raw_scores = list(aux_outputs.logits)

    results = []
    for i, b in enumerate(bboxes):
        score = raw_scores[i] if i < len(raw_scores) else 1.0
        results.append(DetectionResult(
            label="detection",
            score=float(score),
            cx=float(b["x"]),
            cy=float(b["y"]),
            h=float(b["h"]),
            w=float(b["w"]),
        ))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def segment(
    image: Image.Image,
    expression: str,
    *,
    max_new_tokens: int = 2048,
    min_dimension: int = 256,
    max_dimension: int = 1024,
) -> tuple[list[MaskResult], InferenceStats]:
    """Segment objects in *image* matching *expression*.

    Args:
        image: PIL Image
        expression: natural-language expression e.g. "cow", "lame sheep"
        max_new_tokens: token budget for generation (more = slower but thorough)
        min_dimension: shortest side of image fed to model
        max_dimension: longest side of model input

    Returns:
        (list of MaskResult, InferenceStats)
    """
    from falcon_perception import build_prompt_for_task
    from falcon_perception.data import load_image
    from falcon_perception.mlx.batch_inference import process_batch_and_generate

    engine, tokenizer, model_args = _ensure_model()

    pil_img = image.convert("RGB")
    orig_w, orig_h = pil_img.size
    prompt = build_prompt_for_task(expression, "segmentation")

    t1 = time.perf_counter()
    batch = process_batch_and_generate(
        tokenizer,
        [(pil_img, prompt)],
        max_length=model_args.max_seq_len,
        min_dimension=min_dimension,
        max_dimension=max_dimension,
    )
    preprocess_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    output_tokens, aux_outputs = engine.generate(
        tokens=batch["tokens"],
        pos_t=batch["pos_t"],
        pos_hw=batch["pos_hw"],
        pixel_values=batch["pixel_values"],
        pixel_mask=batch["pixel_mask"],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        task="segmentation",
    )
    generation_ms = (time.perf_counter() - t2) * 1000

    aux = aux_outputs[0]
    masks = _postprocess_masks(aux, orig_w, orig_h, min_dimension, max_dimension)

    eos_toks = np.array(output_tokens[0]).flatten()
    pad = tokenizer.pad_token_id
    eos = tokenizer.eos_token_id
    prefill_n = batch["tokens"].shape[1]
    dec_toks = eos_toks[prefill_n:]
    eos_pos = np.where((dec_toks == eos) | (dec_toks == pad))[0]
    n_dec = int(eos_pos[0] + 1) if len(eos_pos) > 0 else len(dec_toks)
    tok_s = n_dec / (generation_ms / 1000) if generation_ms > 0 else 0

    stats = InferenceStats(
        preprocess_ms=round(preprocess_ms, 1),
        generation_ms=round(generation_ms, 1),
        total_ms=round(preprocess_ms + generation_ms, 1),
        prefill_tokens=prefill_n,
        decoded_tokens=n_dec,
        tokens_per_sec=round(tok_s, 1),
        n_masks=len(masks),
        n_detections=0,
    )
    return masks, stats


def detect(
    image: Image.Image,
    expression: str,
    *,
    max_new_tokens: int = 200,
    min_dimension: int = 256,
    max_dimension: int = 1024,
) -> tuple[list[DetectionResult], InferenceStats]:
    """Detect bounding boxes for objects matching *expression*.

    Faster than segment() — no mask generation overhead.
    """
    from falcon_perception import build_prompt_for_task
    from falcon_perception.data import load_image
    from falcon_perception.mlx.batch_inference import process_batch_and_generate

    engine, tokenizer, model_args = _ensure_model()

    pil_img = image.convert("RGB")
    orig_w, orig_h = pil_img.size
    prompt = build_prompt_for_task(expression, "detection")

    t1 = time.perf_counter()
    batch = process_batch_and_generate(
        tokenizer,
        [(pil_img, prompt)],
        max_length=model_args.max_seq_len,
        min_dimension=min_dimension,
        max_dimension=max_dimension,
    )
    preprocess_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    output_tokens, aux_outputs = engine.generate(
        tokens=batch["tokens"],
        pos_t=batch["pos_t"],
        pos_hw=batch["pos_hw"],
        pixel_values=batch["pixel_values"],
        pixel_mask=batch["pixel_mask"],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        task="detection",
    )
    generation_ms = (time.perf_counter() - t2) * 1000

    aux = aux_outputs[0]
    detections = _postprocess_detections(aux)
    detections_labeled = [
        DetectionResult(
            label=expression,
            score=d.score,
            cx=d.cx, cy=d.cy, h=d.h, w=d.w,
        )
        for d in detections
    ]

    eos_toks = np.array(output_tokens[0]).flatten()
    pad = tokenizer.pad_token_id
    eos = tokenizer.eos_token_id
    prefill_n = batch["tokens"].shape[1]
    dec_toks = eos_toks[prefill_n:]
    eos_pos = np.where((dec_toks == eos) | (dec_toks == pad))[0]
    n_dec = int(eos_pos[0] + 1) if len(eos_pos) > 0 else len(dec_toks)
    tok_s = n_dec / (generation_ms / 1000) if generation_ms > 0 else 0

    stats = InferenceStats(
        preprocess_ms=round(preprocess_ms, 1),
        generation_ms=round(generation_ms, 1),
        total_ms=round(preprocess_ms + generation_ms, 1),
        prefill_tokens=prefill_n,
        decoded_tokens=n_dec,
        tokens_per_sec=round(tok_s, 1),
        n_masks=0,
        n_detections=len(detections_labeled),
    )
    return detections_labeled, stats


def ocr(
    image: Image.Image,
    question: str = "read all text in the image",
    *,
    max_new_tokens: int = 500,
    min_dimension: int = 256,
    max_dimension: int = 1024,
) -> tuple[list[DetectionResult], str, InferenceStats]:
    """Read text from an image (ear tags, brand markings, signage).

    Returns (detections, extracted_text, stats).
    The detections point to text regions; extracted_text is the full decoded string.
    """
    from falcon_perception import build_prompt_for_task
    from falcon_perception.data import load_image
    from falcon_perception.mlx.batch_inference import process_batch_and_generate

    engine, tokenizer, model_args = _ensure_model()

    pil_img = image.convert("RGB")
    orig_w, orig_h = pil_img.size
    prompt = build_prompt_for_task(question, "ocr")

    t1 = time.perf_counter()
    batch = process_batch_and_generate(
        tokenizer,
        [(pil_img, prompt)],
        max_length=model_args.max_seq_len,
        min_dimension=min_dimension,
        max_dimension=max_dimension,
    )
    preprocess_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    output_tokens, aux_outputs = engine.generate(
        tokens=batch["tokens"],
        pos_t=batch["pos_t"],
        pos_hw=batch["pos_hw"],
        pixel_values=batch["pixel_values"],
        pixel_mask=batch["pixel_mask"],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        task="ocr",
    )
    generation_ms = (time.perf_counter() - t2) * 1000

    # Decode the generated text
    eos_toks = np.array(output_tokens[0]).flatten()
    pad = tokenizer.pad_token_id
    eos = tokenizer.eos_token_id
    prefill_n = batch["tokens"].shape[1]
    dec_toks = eos_toks[prefill_n:]
    eos_pos = np.where((dec_toks == eos) | (dec_toks == pad))[0]
    n_dec = int(eos_pos[0] + 1) if len(eos_pos) > 0 else len(dec_toks)
    tok_s = n_dec / (generation_ms / 1000) if generation_ms > 0 else 0

    # Decode text tokens — OCR outputs special markup tokens so we keep them
    # and parse coord/size/seg structure
    text_tokens = eos_toks[prefill_n:prefill_n + n_dec].tolist()
    extracted_text = tokenizer.decode(text_tokens, skip_special_tokens=False).strip()

    # Also get detections if any bounding boxes were found
    aux = aux_outputs[0]
    detections = _postprocess_detections(aux)

    stats = InferenceStats(
        preprocess_ms=round(preprocess_ms, 1),
        generation_ms=round(generation_ms, 1),
        total_ms=round(preprocess_ms + generation_ms, 1),
        prefill_tokens=prefill_n,
        decoded_tokens=n_dec,
        tokens_per_sec=round(tok_s, 1),
        n_masks=0,
        n_detections=len(detections),
    )
    return detections, extracted_text.strip(), stats
