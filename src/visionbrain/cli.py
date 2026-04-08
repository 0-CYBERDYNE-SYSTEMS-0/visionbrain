#!/usr/bin/env python3
"""VisionBrain CLI — farmer-friendly interface to agricultural vision AI.

Usage:
    visionbrain detect  --image <path> --query <expression>
    visionbrain segment --image <path> --query <expression>
    visionbrain ocr     --image <path>
    visionbrain track   --video <path> --query <expr> [--output <path>]
    visionbrain agent   --image <path> --question <question>
    visionbrain status
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

from . import __version__
from .loader import print_status, falcon_perception_record, sam31_record
from .fp_inference import segment, detect, ocr
from .sam3_inference import (
    track_video,
    track_realtime,
    detect_multi,
    sam31_available,
)
from .viz import render_som, render_detections


# ──────────────────────────────────────────────────────────────────────────────
# detect
# ──────────────────────────────────────────────────────────────────────────────

def cmd_detect(args: argparse.Namespace) -> None:
    """Count and localize objects via bounding boxes (fast)."""
    img = Image.open(args.image)
    rec = falcon_perception_record()
    if not rec.can_load:
        print(f"ERROR: Falcon Perception not ready — {rec.note}", file=sys.stderr)
        sys.exit(1)

    results, stats = detect(
        img,
        args.query,
        max_new_tokens=args.max_tokens,
    )

    print(f"\nDetected {len(results)} '{args.query}' objects")
    print(f"  Preprocess: {stats.preprocess_ms:.0f}ms | Generation: {stats.generation_ms:.0f}ms | Total: {stats.total_ms:.0f}ms")
    print(f"  Prefill tokens: {stats.prefill_tokens} | Decoded: {stats.decoded_tokens} | Speed: {stats.tokens_per_sec:.1f} tok/s")

    for i, r in enumerate(results[:20]):
        print(f"  [{i+1}] score={r.score:.3f}  cx={r.cx:.3f} cy={r.cy:.3f}  h={r.h:.3f} w={r.w:.3f}")
    if len(results) > 20:
        print(f"  ... and {len(results) - 20} more")

    if args.output:
        out = render_detections(img, results)
        out.save(args.output)
        print(f"\nAnnotated image saved to {args.output}")


# ──────────────────────────────────────────────────────────────────────────────
# segment
# ──────────────────────────────────────────────────────────────────────────────

def cmd_segment(args: argparse.Namespace) -> None:
    """Segment objects with pixel-accurate masks (slower than detect)."""
    img = Image.open(args.image)
    rec = falcon_perception_record()
    if not rec.can_load:
        print(f"ERROR: Falcon Perception not ready — {rec.note}", file=sys.stderr)
        sys.exit(1)

    results, stats = segment(
        img,
        args.query,
        max_new_tokens=args.max_tokens,
    )

    print(f"\nSegmented {len(results)} '{args.query}' objects")
    print(f"  Preprocess: {stats.preprocess_ms:.0f}ms | Generation: {stats.generation_ms:.0f}ms | Total: {stats.total_ms:.0f}ms")

    for r in results[:20]:
        print(f"  [{r.mask_id}] region={r.image_region:12s} area={r.area_fraction:.4f}  cx={r.centroid_x:.3f} cy={r.centroid_y:.3f}  bbox=({r.bbox_x1:.3f},{r.bbox_y1:.3f})→({r.bbox_x2:.3f},{r.bbox_y2:.3f})")
    if len(results) > 20:
        print(f"  ... and {len(results) - 20} more")

    if args.output:
        out = render_som(img, results)
        out.save(args.output)
        print(f"\nSet-of-Marks image saved to {args.output}")


# ──────────────────────────────────────────────────────────────────────────────
# ocr
# ──────────────────────────────────────────────────────────────────────────────

def cmd_ocr(args: argparse.Namespace) -> None:
    """Read text from an image (ear tags, brand markings, signage)."""
    img = Image.open(args.image)
    rec = falcon_perception_record()
    if not rec.can_load:
        print(f"ERROR: Falcon Perception not ready — {rec.note}", file=sys.stderr)
        sys.exit(1)

    question = args.question or "read all text in the image"
    results, text, stats = ocr(img, question, max_new_tokens=args.max_tokens)

    print(f"\nOCR results ({len(results)} text regions)")
    print(f"  Total time: {stats.total_ms:.0f}ms")
    if text:
        # OCR outputs special markup tokens — clean them up for display
        clean = (text
            .replace("<|coord|>", " ")
            .replace("<|size|>", " ")
            .replace("<|seg|>", " ")
            .replace("<|end_of_query|>", " ")
            .replace("<|end_of_text|>", " ")
            .replace("<|pad|>", "")
            .replace("<", "<").strip())
        # Collapse whitespace
        import re
        clean = re.sub(r"\s+", " ", clean).strip()
        if clean:
            print(f"\n  Extracted markup:\n    {clean}")
    for i, r in enumerate(results[:20]):
        print(f"  Region [{i+1}] score={r.score:.3f}  cx={r.cx:.3f} cy={r.cy:.3f}  h={r.h:.3f} w={r.w:.3f}")
    if len(results) > 20:
        print(f"  ... and {len(results) - 20} more")


# ──────────────────────────────────────────────────────────────────────────────
# sam3 detect-multi
# ──────────────────────────────────────────────────────────────────────────────

def cmd_sam3_detect(args: argparse.Namespace) -> None:
    """SAM 3.1 multi-prompt detection/segmentation on still images."""
    if not sam31_available():
        rec = sam31_record()
        print(f"ERROR: SAM 3.1 not ready — {rec.note}", file=sys.stderr)
        print("Run: huggingface-cli download facebook/sam3.1", file=sys.stderr)
        sys.exit(1)

    img = Image.open(args.image)
    results = detect_multi(
        img,
        args.prompts,
        threshold=args.threshold,
        resolution=args.resolution,
        task=args.task,
    )

    print(f"\nSAM 3.1 found {len(results)} objects")
    for i, r in enumerate(results[:20]):
        mask_str = " [masked]" if r.mask is not None else ""
        print(f"  [{i+1}] '{r.label}' score={r.score:.3f}  bbox=({r.bbox_xyxy[0]:.3f},{r.bbox_xyxy[1]:.3f})→({r.bbox_xyxy[2]:.3f},{r.bbox_xyxy[3]:.3f}){mask_str}")
    if len(results) > 20:
        print(f"  ... and {len(results) - 20} more")

    if args.output:
        # Render SAM detections
        from .sam3_inference import Sam31Detection
        det_results = [
            type('D', (), {
                'label': r.label,
                'score': r.score,
                'cx': (r.bbox_xyxy[0] + r.bbox_xyxy[2]) / 2 / img.width,
                'cy': (r.bbox_xyxy[1] + r.bbox_xyxy[3]) / 2 / img.height,
                'h': (r.bbox_xyxy[3] - r.bbox_xyxy[1]) / img.height,
                'w': (r.bbox_xyxy[2] - r.bbox_xyxy[0]) / img.width,
            })()
            for r in results
        ]
        out = render_detections(img, det_results)
        out.save(args.output)
        print(f"\nAnnotated image saved to {args.output}")


# ──────────────────────────────────────────────────────────────────────────────
# track
# ──────────────────────────────────────────────────────────────────────────────

def cmd_track(args: argparse.Namespace) -> None:
    """Track objects in a video file using SAM 3.1."""
    if not sam31_available():
        rec = sam31_record()
        print(f"ERROR: SAM 3.1 not ready — {rec.note}", file=sys.stderr)
        print("Run: huggingface-cli download facebook/sam3.1", file=sys.stderr)
        sys.exit(1)

    print(f"Tracking {args.prompts} in {args.video}...")
    stats = track_video(
        args.video,
        args.prompts,
        output_path=args.output,
        threshold=args.threshold,
        every_n_frames=args.every,
        backbone_every=args.backbone_every,
        resolution=args.resolution,
        opacity=args.opacity,
    )

    print(f"\nDone. Output: {stats.output_path}")
    print(f"  {stats.processed_frames}/{stats.total_frames} frames processed")
    print(f"  {stats.unique_objects} object types tracked")


# ──────────────────────────────────────────────────────────────────────────────
# agent
# ──────────────────────────────────────────────────────────────────────────────

def cmd_agent(args: argparse.Namespace) -> None:
    """Interactive VLM-powered agent on an image."""
    img = Image.open(args.image)
    rec = falcon_perception_record()
    if not rec.can_load:
        print(f"ERROR: Falcon Perception not ready — {rec.note}", file=sys.stderr)
        sys.exit(1)

    if not args.api_key:
        # Try environment variable
        import os
        args.api_key = os.environ.get("OPENAI_API_KEY", "")

    if not args.api_key:
        print("ERROR: --api-key required (or set OPENAI_API_KEY env var)", file=sys.stderr)
        sys.exit(1)

    from .agent_loop import run_agent, VLMClient

    client = VLMClient(
        api_key=args.api_key,
        model=args.model or "gpt-4o",
        base_url=args.base_url,
    )

    result = run_agent(
        img,
        args.question,
        client,
        verbose=args.verbose,
    )

    print(f"\n{'='*60}")
    print(f"Answer: {result.answer}")
    print(f"Supporting masks: {result.supporting_mask_ids}")
    print(f"FP calls: {result.n_fp_calls} | VLM calls: {result.n_vlm_calls}")
    print(f"{'='*60}")

    if result.final_image and args.output:
        result.final_image.save(args.output)
        print(f"Annotated image saved to {args.output}")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VisionBrain — Agricultural vision AI on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--version", action="version", version=f"VisionBrain {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # status
    sub.add_parser("status", help="Print model cache status and exit")

    # detect
    p = sub.add_parser("detect", help="Detect objects with bounding boxes (fast)")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--query", required=True, help="Natural-language expression, e.g. 'cow'")
    p.add_argument("--max-tokens", type=int, default=200, help="Token budget (default 200)")
    p.add_argument("--output", help="Save annotated image to this path")

    # segment
    p = sub.add_parser("segment", help="Segment objects with pixel-accurate masks")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--query", required=True, help="Natural-language expression")
    p.add_argument("--max-tokens", type=int, default=2048, help="Token budget (default 2048)")
    p.add_argument("--output", help="Save SoM image to this path")

    # ocr
    p = sub.add_parser("ocr", help="Read text from an image")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--question", default=None, help="Custom question (default: read all text)")
    p.add_argument("--max-tokens", type=int, default=500, help="Token budget")

    # sam3
    p = sub.add_parser("sam3", help="SAM 3.1 multi-prompt detection (requires SAM 3.1 weights)")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--prompts", required=True, nargs="+", help="Text prompts, e.g. cow sheep fence")
    p.add_argument("--task", choices=["detect", "segment"], default="detect")
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--resolution", type=int, default=1008)
    p.add_argument("--output", help="Save annotated image")

    # track
    p = sub.add_parser("track", help="Track objects in video (requires SAM 3.1 weights)")
    p.add_argument("--video", required=True, help="Video file path")
    p.add_argument("--prompts", required=True, nargs="+", help="Text prompts to track")
    p.add_argument("--output", help="Output video path")
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--every", type=int, default=2, help="Run detection every N frames")
    p.add_argument("--backbone-every", type=int, default=1, help="Re-run ViT every N detections")
    p.add_argument("--resolution", type=int, default=1008)
    p.add_argument("--opacity", type=float, default=0.6)

    # agent
    p = sub.add_parser("agent", help="VLM-powered visual reasoning agent")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--question", required=True, help="Question about the image")
    p.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--model", default=None, help="VLM model (default: gpt-4o)")
    p.add_argument("--base-url", default=None, help="API base URL for proxies")
    p.add_argument("--output", help="Save annotated image")
    p.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.command == "status":
        print_status()
        return

    dispatch = {
        "detect": cmd_detect,
        "segment": cmd_segment,
        "ocr": cmd_ocr,
        "sam3": cmd_sam3_detect,
        "track": cmd_track,
        "agent": cmd_agent,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
