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
import sys
from pathlib import Path

from PIL import Image

from . import __version__


# ──────────────────────────────────────────────────────────────────────────────
# detect
# ──────────────────────────────────────────────────────────────────────────────

def cmd_detect(args: argparse.Namespace) -> None:
    """Count and localize objects via bounding boxes (fast)."""
    from .loader import falcon_perception_record
    from .fp_inference import detect
    from .viz import render_detections

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
    from .loader import falcon_perception_record
    from .fp_inference import segment
    from .viz import render_som

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
    from .loader import falcon_perception_record
    from .fp_inference import ocr

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
    from .loader import sam31_record
    from .sam3_inference import detect_multi, sam31_available
    from .viz import render_detections

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
    from .loader import sam31_record
    from .sam3_inference import track_video, sam31_available

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
# analyze — full pipeline: SAM 3.1 video → structured JSON → Gemma 4 reasoning
# ──────────────────────────────────────────────────────────────────────────────

def _extract_key_frames(video_path: str, frame_data: list[dict], n: int = 8) -> list[tuple[int, float, "Image.Image"]]:
    """Extract the N frames with the most detections as PIL Images.

    Returns list of (frame_index, timestamp, pil_image) tuples.
    """
    import cv2

    # Sort by detection count descending, take top N
    ranked = sorted(frame_data, key=lambda f: f["n_detections"], reverse=True)[:n]
    ranked_by_frame = sorted(ranked, key=lambda f: f["frame_index"])

    cap = cv2.VideoCapture(video_path)
    results = []
    for frame_info in ranked_by_frame:
        fi = frame_info["frame_index"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame_bgr = cap.read()
        if ret:
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            results.append((fi, frame_info["timestamp"], pil))
    cap.release()
    return results


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run the full VisionBrain pipeline on a video:

    1. SAM 3.1 — track objects frame-by-frame, annotated video + structured JSON
    2. Falcon Perception (optional) — semantic deep-dive on key frames
    3. Gemma 4 (remote) — reason about the detections, generate field report

    Output: annotated video + per-frame JSON detections + natural-language report.
    """
    import json
    from pathlib import Path

    from .loader import sam31_record, falcon_perception_record
    from .sam3_inference import track_video_with_json, sam31_available
    from .fp_inference import detect
    from .remote_gemma_inference import generate_report as gemma_report, ask as gemma_ask, gemma_available as gemma_remote_available

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if not sam31_available():
        rec = sam31_record()
        print(f"ERROR: SAM 3.1 not ready — {rec.note}", file=sys.stderr)
        print("Run: huggingface-cli download mlx-community/sam3.1-bf16", file=sys.stderr)
        sys.exit(1)

    stem = video_path.stem
    out_video = args.output or str(video_path.parent / f"{stem}_analyzed{video_path.suffix}")
    out_json = args.json_output or str(video_path.parent / f"{stem}_detections.json")
    out_report = args.report_output or str(video_path.parent / f"{stem}_report.txt")

    print(f"\n=== VisionBrain Pipeline ===")
    print(f"Video: {video_path}")
    print(f"Query: {args.query}")
    print(f"Remote Gemma: http://100.72.41.118:8080")
    print()

    # Step 1: SAM 3.1 tracking with structured JSON export
    total_steps = 4 if getattr(args, "falcon_refine", False) else 3
    prompts = args.prompts if args.prompts else [args.query]
    print(f"[1/{total_steps}] SAM 3.1 — tracking '{' '.join(prompts)}' in {video_path.name}...")
    print(f"       (every {args.every} frames, threshold {args.threshold}, resolution {args.resolution})")
    stats, frame_data = track_video_with_json(
        str(video_path),
        prompts,
        output_path=out_video,
        json_path=out_json,
        threshold=args.threshold,
        every_n_frames=args.every,
        backbone_every=args.backbone_every,
        resolution=args.resolution,
        opacity=args.opacity,
        contour_thickness=2,
    )
    print(f"  → {stats.processed_frames}/{stats.total_frames} frames processed")
    print(f"  → {stats.unique_objects} unique objects tracked")
    print(f"  → Annotated video: {out_video}")
    print(f"  → JSON detections: {out_json}")
    print()

    # Step 2 (optional): Falcon Perception key-frame refinement
    falcon_summary_parts = []
    if getattr(args, "falcon_refine", False):
        rec = falcon_perception_record()
        if not rec.can_load:
            print(f"  WARNING: Falcon Perception not ready — skipping refine step ({rec.note})")
        else:
            frames_with_dets = [f for f in frame_data if f["n_detections"] > 0]
            if frames_with_dets:
                n_refine = min(getattr(args, "falcon_frames", 6), len(frames_with_dets))
                print(f"[2/{total_steps}] Falcon Perception — semantic analysis on top {n_refine} frames...")
                key_frames = _extract_key_frames(str(video_path), frames_with_dets, n=n_refine)
                for fi, ts, pil_frame in key_frames:
                    fp_results, fp_stats = detect(
                        pil_frame,
                        args.query,
                        max_new_tokens=200,
                    )
                    det_strs = [
                        f"  [{i+1}] score={r.score:.2f} cx={r.cx:.2f} cy={r.cy:.2f} h={r.h:.2f} w={r.w:.2f}"
                        for i, r in enumerate(fp_results[:10])
                    ]
                    falcon_summary_parts.append(
                        f"Frame {fi} (t={ts:.2f}s) — Falcon '{args.query}' "
                        f"({fp_stats.total_ms:.0f}ms, {len(fp_results)} results):\n"
                        + ("\n".join(det_strs) if det_strs else "  [no detections]")
                    )
                    print(f"  Frame {fi} (t={ts:.1f}s): {len(fp_results)} Falcon detections — {fp_stats.total_ms:.0f}ms")
                print()
            else:
                print(f"[2/{total_steps}] Falcon Perception — no frames with SAM detections to refine; skipping.")
                print()
        step_gemma = f"[3/{total_steps}]"
    else:
        step_gemma = f"[2/{total_steps}]"

    # Step 3: Remote Gemma 4 reasoning
    if not gemma_remote_available():
        print("WARNING: Gemma 4 server unreachable — detections saved but report not generated.")
        print(f"  Check: curl http://100.72.41.118:8080/v1/models")
        print(f"\n=== Pipeline complete (partial) ===")
        return

    print(f"{step_gemma} Gemma 4 26B — reasoning about {video_path.name}...")
    print(f"       Sending {len(frame_data)} frames of structured detections...")

    # Build a compact summary for Gemma
    total_dets = sum(f["n_detections"] for f in frame_data)
    all_labels: dict[str, int] = {}
    all_track_ids: set[int] = set()
    for frame in frame_data:
        for det in frame["detections"]:
            lbl = det["label"]
            all_labels[lbl] = all_labels.get(lbl, 0) + 1
            all_track_ids.add(det["track_id"])

    # Build the structured summary Gemma will reason over
    summary_parts = [
        f"Video: {video_path.name}",
        f"Duration: {stats.total_frames/stats.fps:.1f}s ({stats.total_frames} frames at {stats.fps:.1f} fps)",
        f"Resolution: {frame_data[0]['detections'][0]['bbox_xyxy'][2]:.0f}x? (from first detection)" if frame_data and frame_data[0]['detections'] else f"Resolution: {args.resolution}",
        f"Tracked object types: {list(all_labels.keys())}",
        f"Total detections across {len(frame_data)} processed frames: {total_dets}",
        f"Unique tracked objects: {len(all_track_ids)}",
        f"User query: {args.query}",
        "",
        "Per-frame detection data (centroid x/y normalized 0-1, area as fraction of frame):",
    ]
    for frame in frame_data:
        if frame["detections"]:
            dets_str = ", ".join(
                f"id{det['track_id']}({det['label']}:{det['score']:.2f}@({det['centroid_norm']['x']:.2f},{det['centroid_norm']['y']:.2f}))"
                for det in frame["detections"]
            )
            summary_parts.append(f"  Frame {frame['frame_index']} (t={frame['timestamp']:.2f}s): {dets_str}")

    if falcon_summary_parts:
        summary_parts.append("")
        summary_parts.append("Falcon Perception key-frame semantic analysis:")
        summary_parts.extend(falcon_summary_parts)

    summary_text = "\n".join(summary_parts)

    if args.report:
        report_resp = gemma_report(
            summary_text,
            report_type=args.report_type,
            max_tokens=args.max_tokens,
            temperature=0.7,
        )
        print(f"  → {report_resp.stats.generation_tokens} tokens in {report_resp.stats.decode_ms/1000:.1f}s")
        print(f"  → Speed: {report_resp.stats.generation_tps:.1f} tok/s")
        print()
        print(f"  {'='*60}")
        print(f"  FIELD REPORT")
        print(f"  {'='*60}")
        print(f"  {report_resp.text}")
        print(f"  {'='*60}")
        with open(out_report, "w") as f:
            f.write(report_resp.text)
        print(f"\n  Report saved to: {out_report}")
    else:
        question = args.question or (
            f"What are the key findings from this drone footage? "
            f"Focus on: {args.query}. Identify individual objects, their movement patterns, "
            f"anomalies, and anything a farmer or rancher should act on."
        )
        resp = gemma_ask(question, detections=[], frame_history=frame_data, max_tokens=args.max_tokens)
        print(f"  → Answer ({resp.stats.generation_tokens} tokens, {resp.stats.decode_ms/1000:.1f}s):")
        print(f"  {resp.text}")
        print(f"  Speed: {resp.stats.generation_tps:.1f} tok/s")

    stages = ["SAM 3.1 tracking"]
    if getattr(args, "falcon_refine", False):
        stages.append("Falcon Perception key-frames")
    stages.append("Gemma 4 reasoning")
    print(f"\n=== Pipeline complete ({' → '.join(stages)}) ===")
    print(f"  Annotated video : {out_video}")
    print(f"  Detection JSON  : {out_json}")
    if args.report:
        print(f"  Field report    : {out_report}")

def cmd_agent(args: argparse.Namespace) -> None:
    """Interactive VLM-powered agent on an image."""
    from .loader import falcon_perception_record
    from .agent_loop import run_agent, VLMClient

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

    # analyze
    p = sub.add_parser("analyze", help="Full pipeline: SAM 3.1 track → Gemma 4 reasoning → report")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--query", required=True, help="Natural-language query (e.g. 'cattle in the pasture')")
    p.add_argument("--prompts", nargs="+", help="SAM 3.1 text prompts to track (default: use --query)")
    p.add_argument("--output", help="Output video path")
    p.add_argument("--json-output", help="Output JSON path for per-frame detections")
    p.add_argument("--report-output", help="Output text report path")
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--every", type=int, default=2, help="Run SAM detection every N frames")
    p.add_argument("--backbone-every", type=int, default=1, help="Re-run ViT backbone every N detections")
    p.add_argument("--resolution", type=int, default=1008, help="SAM input resolution (1008 = native)")
    p.add_argument("--opacity", type=float, default=0.6, help="Mask overlay opacity")
    p.add_argument("--sample-frames", type=int, default=10, help="Number of frames to sample for Gemma reasoning")
    p.add_argument("--report", action="store_true", help="Generate a written field report via Gemma 4")
    p.add_argument("--report-type", choices=["field", "brief", "json"], default="field",
                   help="Report format (default: field)")
    p.add_argument("--question", help="Custom question for Gemma (overrides default reasoning)")
    p.add_argument("--max-tokens", type=int, default=512, help="Max output tokens for Gemma")
    p.add_argument("--falcon-refine", action="store_true",
                   help="Run Falcon Perception on key frames for semantic deep-dive")
    p.add_argument("--falcon-frames", type=int, default=6,
                   help="Number of key frames to analyze with Falcon (default 6)")

    # ui
    p = sub.add_parser("ui", help="Launch the web Ground Control UI (opens browser)")
    p.add_argument("--port", type=int, default=7860, help="Port (default 7860)")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    p.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")

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
        from .loader import print_status
        print_status()
        return

    if args.command == "ui":
        try:
            from .web_app import run as ui_run
        except ImportError:
            print("ERROR: fastapi and uvicorn required. Install with:", file=sys.stderr)
            print("  pip install fastapi 'uvicorn[standard]' python-multipart", file=sys.stderr)
            sys.exit(1)
        print(f"Starting VisionBrain Ground Control at http://{args.host}:{args.port}")
        ui_run(host=args.host, port=args.port, open_browser=not args.no_browser)
        return

    dispatch = {
        "detect": cmd_detect,
        "segment": cmd_segment,
        "ocr": cmd_ocr,
        "sam3": cmd_sam3_detect,
        "track": cmd_track,
        "analyze": cmd_analyze,
        "agent": cmd_agent,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
