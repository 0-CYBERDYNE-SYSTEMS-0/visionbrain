"""Visualization helpers — Set-of-Marks rendering, crop extraction, annotations."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .fp_inference import MaskResult, DetectionResult


# ──────────────────────────────────────────────────────────────────────────────
# Colors (farmer-friendly, high contrast)
# ──────────────────────────────────────────────────────────────────────────────

MASK_COLORS = [
    (255, 80, 80),   # red
    (80, 200, 120),  # green
    (80, 120, 255),  # blue
    (255, 200, 80),  # yellow
    (200, 80, 255),  # purple
    (80, 220, 220),  # cyan
    (255, 140, 80),  # orange
    (200, 200, 80),  # lime
]


# ──────────────────────────────────────────────────────────────────────────────
# Set-of-Marks image
# ──────────────────────────────────────────────────────────────────────────────

def render_som(
    image: Image.Image,
    masks: list[MaskResult],
    max_size: int = 1200,
    show_labels: bool = True,
) -> Image.Image:
    """Render Set-of-Marks overlay on the image.

    Overlays colored masks with numbered labels for each mask.
    """
    img = image.convert("RGB")
    w, h = img.size

    # Downscale for display if very large
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        scale = new_w / w
    else:
        scale = 1.0

    # Convert to numpy for fast drawing
    arr = np.array(img)
    overlay = arr.copy()

    for i, mask in enumerate(masks):
        color = MASK_COLORS[i % len(MASK_COLORS)]
        r, g, b = color

        # Decode RLE mask to pixel array
        binary = _rle_to_binary(mask.rle, img.height, img.width)
        mask_arr = (binary * 255).astype(np.uint8) if binary.dtype != np.uint8 else binary

        # Resize mask to match image
        if mask_arr.shape != (img.height, img.width):
            mask_pil = Image.fromarray(mask_arr, mode="L").resize((img.width, img.height), Image.NEAREST)
            mask_arr = np.array(mask_pil)

        # Apply colored overlay
        for c_idx, c_val in enumerate([r, g, b]):
            channel = overlay[:, :, c_idx]
            channel = np.where(mask_arr > 127, (c_val * 0.4 + channel * 0.6).astype(np.uint8), channel)
            overlay[:, :, c_idx] = channel

    # Blend
    blended = Image.fromarray((overlay * 0.7 + arr * 0.3).astype(np.uint8))
    draw = ImageDraw.Draw(blended)

    # Draw labels
    for i, mask in enumerate(masks):
        color = MASK_COLORS[i % len(MASK_COLORS)]
        cx = int((mask.bbox_x1 + mask.bbox_x2) / 2 * img.width * scale)
        cy = int((mask.bbox_y1 + mask.bbox_y2) / 2 * img.height * scale)

        # Circle
        r = 14
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(255, 255, 255), width=2)
        # Number
        label = str(mask.mask_id)
        # Simple font fallback
        try:
            fnt = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            fnt = ImageFont.load_default()
        bbox = draw.textbbox((cx, cy), label, font=fnt)
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        draw.rectangle([cx - bw // 2 - 2, cy - bh // 2 - 2, cx + bw // 2 + 2, cy + bh // 2 + 2], fill=(0, 0, 0))
        draw.text((cx - bw // 2, cy - bh // 2), label, fill=(255, 255, 255), font=fnt)

    return blended


def render_detections(
    image: Image.Image,
    detections: list[DetectionResult],
    max_size: int = 1200,
    show_scores: bool = True,
) -> Image.Image:
    """Render bounding boxes on the image."""
    img = image.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        scale = ratio
    else:
        scale = 1.0

    draw = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        fnt = ImageFont.load_default()

    for i, det in enumerate(detections):
        color = MASK_COLORS[i % len(MASK_COLORS)]
        cx, cy, bh, bw = det.cx, det.cy, det.h, det.w

        # Convert normalized cxcywh to pixel xyxy
        x1 = int((cx - bw / 2) * img.width)
        y1 = int((cy - bh / 2) * img.height)
        x2 = int((cx + bw / 2) * img.width)
        y2 = int((cy + bh / 2) * img.height)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = det.label
        if show_scores:
            label += f" {det.score:.2f}"
        draw.text((x1 + 4, y1 + 4), label, fill=color, font=fnt)

    return img


# ──────────────────────────────────────────────────────────────────────────────
# Crop extraction
# ──────────────────────────────────────────────────────────────────────────────

def get_crop(image: Image.Image, mask: MaskResult, pad: float = 0.05) -> Image.Image:
    """Extract a tight crop of the image around a mask's bounding box.

    Args:
        image: PIL Image
        mask: MaskResult
        pad: fractional padding around the crop (default 5%)
    """
    from pycocotools import mask as mask_utils

    w, h = image.size
    x1 = max(0, int(mask.bbox_x1 * w - pad * w))
    y1 = max(0, int(mask.bbox_y1 * h - pad * h))
    x2 = min(w, int(mask.bbox_x2 * w + pad * w))
    y2 = min(h, int(mask.bbox_y2 * h + pad * h))

    return image.crop((x1, y1, x2, y2))


# ──────────────────────────────────────────────────────────────────────────────
# Spatial relations
# ──────────────────────────────────────────────────────────────────────────────

def compute_relations(masks: list[MaskResult]) -> dict:
    """Compute pairwise spatial relations between masks.

    Returns IoU, left/right, above/below, size ratio, centroid distance.
    """
    from pycocotools import mask as mask_utils

    n = len(masks)
    if n < 2:
        return {"note": "Need at least 2 masks for pairwise relations"}

    def to_bytes_rle(rle):
        out = dict(rle)
        if isinstance(out.get("counts"), str):
            out["counts"] = out["counts"].encode("utf-8")
        return out

    pairs = {}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = masks[i], masks[j]
            ia = to_bytes_rle(a.rle)
            ib = to_bytes_rle(b.rle)

            iou_mat = np.asarray(mask_utils.iou([ia], [ib], [False]))
            iou = round(float(iou_mat[0][0]), 4)

            dist = round(
                ((a.centroid_x - b.centroid_x) ** 2 +
                 (a.centroid_y - b.centroid_y) ** 2) ** 0.5, 4
            )
            size_ratio = round(a.area_fraction / b.area_fraction, 3) if b.area_fraction > 0 else None

            key = f"{a.mask_id}_vs_{b.mask_id}"
            pairs[key] = {
                "iou": iou,
                f"{a.mask_id}_left_of_{b.mask_id}": a.centroid_x < b.centroid_x,
                f"{a.mask_id}_above_{b.mask_id}": a.centroid_y < b.centroid_y,
                f"{a.mask_id}_larger_than_{b.mask_id}": a.area_fraction > b.area_fraction,
                f"size_ratio_{a.mask_id}_over_{b.mask_id}": size_ratio,
                "centroid_distance_norm": dist,
            }

    return {"pairs": pairs}


# ──────────────────────────────────────────────────────────────────────────────
# Internal
# ──────────────────────────────────────────────────────────────────────────────

def _rle_to_binary(rle: dict, h: int, w: int) -> np.ndarray:
    from pycocotools import mask as mask_utils

    def to_bytes_rle(r):
        out = dict(r)
        if isinstance(out.get("counts"), str):
            out["counts"] = out["counts"].encode("utf-8")
        return out

    try:
        decoded = mask_utils.decode(to_bytes_rle(rle))
    except Exception:
        return np.zeros((h, w), dtype=np.uint8)

    if decoded.shape != (h, w):
        pil = Image.fromarray(decoded.astype(np.uint8), mode="L").resize((w, h), Image.NEAREST)
        return np.array(pil)
    return decoded.astype(np.uint8)
