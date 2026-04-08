"""Agent tools — ground_expression, compute_relations, masks_to_vlm_json.

This module provides the SAME public API as Falcon-Perception/demo/agent/fp_tools.py
but uses the MLX batch_inference path (no torch). The original fp_tools.py is
never modified.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from .loader import fp_module_path

# ── Add Falcon-Perception to path (read-only) ──────────────────────────────────
_ROOT = fp_module_path().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .fp_inference import segment, MaskResult


# ──────────────────────────────────────────────────────────────────────────────
# RLE helpers (same as fp_tools.py — no torch needed)
# ──────────────────────────────────────────────────────────────────────────────

def _to_bytes_rle(rle: dict) -> dict:
    out = rle.copy()
    if isinstance(out.get("counts"), str):
        out["counts"] = out["counts"].encode("utf-8")
    return out


def _resize_rle(rle: dict, target_h: int, target_w: int) -> dict:
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


# ──────────────────────────────────────────────────────────────────────────────
# Mask metadata computation
# ──────────────────────────────────────────────────────────────────────────────

def _image_region_label(cx_norm: float, cy_norm: float) -> str:
    h = "left" if cx_norm < 0.33 else ("center" if cx_norm < 0.67 else "right")
    v = "top" if cy_norm < 0.33 else ("middle" if cy_norm < 0.67 else "bottom")
    if v == "middle" and h == "center":
        return "center"
    if v == "middle":
        return h
    return f"{v}-{h}"


def _compute_mask_metadata(
    rle: dict,
    img_w: int,
    img_h: int,
    mask_id: int,
) -> dict | None:
    """Compute spatial metadata from an RLE mask.

    Returns dict with: id, area_fraction, centroid_norm, bbox_norm,
    image_region, rle
    """
    binary = mask_utils.decode(_to_bytes_rle(rle))
    if binary is None or not binary.any():
        return None

    area = int(binary.sum())
    area_fraction = round(area / (img_h * img_w), 4)

    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any() or not cols.any():
        return None

    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

    cx_norm = round((cmin + cmax + 1) / 2.0 / img_w, 4)
    cy_norm = round((rmin + rmax + 1) / 2.0 / img_h, 4)

    return {
        "id": mask_id,
        "area_fraction": area_fraction,
        "centroid_norm": {"x": cx_norm, "y": cy_norm},
        "bbox_norm": {
            "x1": round(cmin / img_w, 4),
            "y1": round(rmin / img_h, 4),
            "x2": round((cmax + 1) / img_w, 4),
            "y2": round((rmax + 1) / img_h, 4),
        },
        "image_region": _image_region_label(cx_norm, cy_norm),
        "rle": rle,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core tool: ground_expression
# ──────────────────────────────────────────────────────────────────────────────

def run_ground_expression(
    image: Image.Image,
    expression: str,
    *,
    max_new_tokens: int = 2048,
    min_dimension: int = 256,
    max_dimension: int = 1024,
) -> dict[int, dict]:
    """Run Falcon Perception segmentation and return per-mask metadata.

    Returns dict mapping 1-indexed mask IDs to metadata dicts:
      {id, area_fraction, centroid_norm, bbox_norm, image_region, rle}

    This is the drop-in replacement for fp_tools.run_ground_expression()
    that uses the MLX path (no torch dependency).
    """
    pil_image = image.convert("RGB")
    orig_w, orig_h = pil_image.size

    masks, _ = segment(
        pil_image,
        expression,
        max_new_tokens=max_new_tokens,
        min_dimension=min_dimension,
        max_dimension=max_dimension,
    )

    result: dict[int, dict] = {}
    assigned_id = 1
    for m in masks:
        # Resize RLE from inference resolution → original image resolution
        rle_orig = _resize_rle(m.rle, orig_h, orig_w)
        meta = _compute_mask_metadata(rle_orig, orig_w, orig_h, mask_id=assigned_id)
        if meta is not None:
            result[assigned_id] = meta
            assigned_id += 1

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Core tool: compute_relations
# ──────────────────────────────────────────────────────────────────────────────

def compute_relations(
    masks: dict[int, dict],
    mask_ids: list[int],
) -> dict:
    """Compute pairwise spatial relationships between the given mask IDs.

    Uses pycocotools IoU and centroid arithmetic.

    Returns:
        {"pairs": {"1_vs_2": {"iou": 0.02, "1_left_of_2": true, ...}, ...}}
    """
    valid_ids = [mid for mid in mask_ids if mid in masks]
    if len(valid_ids) < 2:
        return {
            "note": (
                f"Need at least 2 valid mask IDs. "
                f"Requested: {mask_ids}, available: {sorted(masks.keys())}"
            )
        }

    prepped: dict[int, dict] = {}
    for mid in valid_ids:
        prepped[mid] = _to_bytes_rle(masks[mid]["rle"])

    pairs: dict[str, dict] = {}
    for i in range(len(valid_ids)):
        for j in range(i + 1, len(valid_ids)):
            a_id = valid_ids[i]
            b_id = valid_ids[j]

            iou_mat = np.asarray(
                mask_utils.iou([prepped[a_id]], [prepped[b_id]], [False])
            )
            iou = round(float(iou_mat[0][0]), 4)

            a = masks[a_id]
            b = masks[b_id]
            cx_a, cy_a = a["centroid_norm"]["x"], a["centroid_norm"]["y"]
            cx_b, cy_b = b["centroid_norm"]["x"], b["centroid_norm"]["y"]
            dist = round(((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5, 4)

            area_a = a["area_fraction"]
            area_b = b["area_fraction"]
            size_ratio = round(area_a / area_b, 3) if area_b > 0 else None

            key = f"{a_id}_vs_{b_id}"
            pairs[key] = {
                "iou": iou,
                f"{a_id}_left_of_{b_id}": cx_a < cx_b,
                f"{a_id}_above_{b_id}": cy_a < cy_b,
                f"{a_id}_larger_than_{b_id}": area_a > area_b,
                f"size_ratio_{a_id}_over_{b_id}": size_ratio,
                "centroid_distance_norm": dist,
            }

    return {"pairs": pairs}


# ──────────────────────────────────────────────────────────────────────────────
# Serialization helper
# ──────────────────────────────────────────────────────────────────────────────

def masks_to_vlm_json(masks: dict[int, dict]) -> list[dict]:
    """Return a JSON-serialisable list of mask metadata, omitting the RLE field."""
    out = []
    for mask_id in sorted(masks.keys()):
        m = masks[mask_id]
        out.append({
            "id": m["id"],
            "area_fraction": m["area_fraction"],
            "centroid_norm": m["centroid_norm"],
            "bbox_norm": m["bbox_norm"],
            "image_region": m["image_region"],
        })
    return out
