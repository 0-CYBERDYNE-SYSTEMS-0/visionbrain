#!/usr/bin/env python3
"""VisionBrain CLI — operator-friendly interface to vision AI on Apple Silicon.

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
    """Read text from an image (signage, labels, markings)."""
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
    from .sam3_inference import track_video, track_video_with_json, sam31_available

    if not sam31_available():
        rec = sam31_record()
        print(f"ERROR: SAM 3.1 not ready — {rec.note}", file=sys.stderr)
        print("Run: huggingface-cli download facebook/sam3.1", file=sys.stderr)
        sys.exit(1)

    if args.json_output:
        print(f"Tracking {args.prompts} in {args.video}...")
        stats, _ = track_video_with_json(
            args.video,
            args.prompts,
            output_path=args.output,
            json_path=args.json_output,
            threshold=args.threshold,
            every_n_frames=args.every,
            backbone_every=args.backbone_every,
            resolution=args.resolution,
            opacity=args.opacity,
            use_supervision=args.supervision,
            track_persistent_ids=args.persistent_ids,
            adaptive_motion=args.adaptive_motion,
            motion_threshold=args.motion_threshold,
            propagate_frames=args.propagate,
        )

        print(f"\nDone. JSON detections: {args.json_output}")
        print(f"  {stats.processed_frames}/{stats.total_frames} frames processed")
        print(f"  {stats.unique_objects} object types tracked")
        if stats.output_path:
            print(f"  Output: {stats.output_path}")
        return

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

    1. (Optional) Fast-path Falcon scan — quick answer in seconds
    2. SAM 3.1 — track objects frame-by-frame, annotated video + structured JSON
    3. Falcon Perception (optional) — semantic deep-dive on key frames, parallel
    4. Gemma 4 (remote) — reason about the detections, generate field report

    Output: annotated video + per-frame JSON detections + natural-language report.
    """
    import cv2
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    from .loader import sam31_record, falcon_perception_record
    from .sam3_inference import render_review_outputs, track_video_with_json, sam31_available
    from .fp_inference import detect
    from .gemma_inference import (
        generate_report as gemma_report,
        ask as gemma_ask,
        gemma_available,
        available_backend,
    )
    from .prompt_router import route, route_fallback
    from .crosscheck import crosscheck, falcon_to_dets, summarize

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
    out_json = args.json_output or str(video_path.parent / f"{stem}_detections.json")
    out_report = args.report_output or str(video_path.parent / f"{stem}_report.txt")
    out_review_reel = getattr(args, "review_reel_output", None)
    include_video = getattr(args, "include_video", False)
    out_video = args.output if include_video else None
    out_still_dir = str(video_path.parent / f"{stem}_stills/")

    # ── Prompt routing — split query into SAM targets and semantic question ────
    routed = route(args.query)
    if routed.segment_targets:
        sam_targets = routed.segment_targets
    else:
        sam_targets = route_fallback(args.query)
    semantic_question = routed.semantic_query or args.query

    print(f"\n=== VisionBrain Pipeline ===")
    print(f"Video: {video_path}")
    print(f"Query: {args.query}")
    print(f"Routing: SAM targets={sam_targets}")
    print(f"         Semantic question: {semantic_question}")
    backend = available_backend()
    if backend:
        print(f"Gemma: {backend} backend active")
    else:
        print("Gemma: not available (detections will still be saved)")
    print()

    # ── Step 0: Fast-path Falcon scan ─────────────────────────────────────────
    fast_result = None
    relevance_scores: dict[int, float] = {}

    if getattr(args, "fast", False):
        from .frame_selector import score_frames

        print(f"\n=== Fast-Path Scan ===")
        print(f"  Scoring frames for '{args.query}'...")

        fast_result = score_frames(
            str(video_path),
            args.query,
            sample_every_n_seconds=5.0,
            max_frames=60,
            resolution=360,
            min_relevance=getattr(args, "min_relevance", 0.2),
        )

        print(f"\n  QUICK ANSWER: {fast_result.quick_answer}")
        print(f"  {fast_result.frames_scored} frames scored in under 60s")

        if fast_result.regions:
            print(f"  Regions of interest:")
            for r in fast_result.regions:
                print(f"    [{r.start_time:.1f}s – {r.end_time:.1f}s] {r.label} (relevance: {r.avg_relevance:.2f})")

        # Build relevance lookup: frame_index → relevance score
        for fs in fast_result.frame_scores:
            relevance_scores[fs.frame_index] = fs.relevance_score

        if args.fast_output:
            Path(args.fast_output).write_text(json.dumps(fast_result.to_dict(), indent=2))
            print(f"\n  Fast-scan JSON saved to: {args.fast_output}")

        # If --fast only (no --report, no --falcon-refine), exit after fast scan
        if not getattr(args, "report", False) and not getattr(args, "falcon_refine", False):
            print(f"\n=== Fast scan complete (no full analysis requested) ===")
            return

        print(f"\n  Continuing with full pipeline...")

    print(f"\n=== VisionBrain Pipeline ===")
    print(f"Video: {video_path}")
    print(f"Query: {args.query}")
    print(f"Remote Gemma: http://100.72.41.118:8080")
    print()

    # ── Step 1: SAM 3.1 tracking ─────────────────────────────────────────────
    total_steps = 4 if getattr(args, "falcon_refine", False) else 3

    adaptive = getattr(args, "adaptive", False)
    motion_threshold = getattr(args, "motion_threshold", 0.03)
    propagate_frames = getattr(args, "propagate", 0)
    relevance_filter = getattr(args, "relevance_filter", False) and bool(relevance_scores)

    adaptive_strs = []
    if adaptive:
        adaptive_strs.append("motion-skip")
    if propagate_frames > 0:
        adaptive_strs.append(f"propagate-{propagate_frames}")
    if relevance_filter:
        adaptive_strs.append("relevance-filter")

    print(f"[1/{total_steps}] SAM 3.1 — tracking {sam_targets} in {video_path.name}...")
    print(f"       every={args.every} frames, backbone-every={args.backbone_every}, "
          f"resolution={args.resolution}, threshold={args.threshold}"
          + (f" | ADAPTIVE: {', '.join(adaptive_strs)}" if adaptive_strs else ""))

    # ── Auto-detect chunking for large videos ──────────────────────────────
    chunk_duration = getattr(args, "chunk_duration", 0)
    chunk_overlap = getattr(args, "chunk_overlap", 3)
    use_chunking = chunk_duration != 0  # 0 = disabled / auto

    # Auto-enable chunking for videos > 60 seconds
    cap = cv2.VideoCapture(str(video_path))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_duration = total_video_frames / video_fps
    cap.release()

    if use_chunking and chunk_duration < 0:
        chunk_duration = 30  # negative means "use default"
    if not use_chunking and video_duration > 60:
        # Auto-enable: videos > 60s likely to OOM
        chunk_duration = 30
        use_chunking = True
        print(f"       AUTO-CHUNKING: video is {video_duration:.0f}s — using {chunk_duration}s chunks")
    if chunk_duration > 0 and video_duration <= chunk_duration:
        # Video short enough — skip chunking
        use_chunking = False
        print(f"       Video is {video_duration:.0f}s (≤ {chunk_duration}s chunk size) — chunking not needed")

    if use_chunking:
        from .sam3_inference import track_video_chunked

        chunk_kwargs = dict(
            chunk_duration=chunk_duration if chunk_duration > 0 else 30,
            overlap_duration=chunk_overlap,
            relevance_regions=fast_result.regions if fast_result and fast_result.regions else None,
        )
        stats, frame_data = track_video_chunked(
            str(video_path),
            sam_targets,
            output_path=out_video,
            json_path=out_json,
            threshold=args.threshold,
            every_n_frames=args.every,
            backbone_every=args.backbone_every,
            resolution=args.resolution,
            opacity=args.opacity,
            contour_thickness=2,
            adaptive_motion=adaptive,
            motion_threshold=motion_threshold,
            propagate_frames=propagate_frames,
            relevance_scores=relevance_scores if relevance_filter else None,
            relevance_threshold=getattr(args, "min_relevance", 0.2),
            **chunk_kwargs,
        )
    else:
        stats, frame_data = track_video_with_json(
            str(video_path),
            sam_targets,
            output_path=out_video,
            json_path=out_json,
            threshold=args.threshold,
            every_n_frames=args.every,
            backbone_every=args.backbone_every,
            resolution=args.resolution,
            opacity=args.opacity,
            contour_thickness=2,
            adaptive_motion=adaptive,
            motion_threshold=motion_threshold,
            propagate_frames=propagate_frames,
            relevance_scores=relevance_scores if relevance_filter else None,
            relevance_threshold=getattr(args, "min_relevance", 0.2),
    )
    print(f"  → {stats.processed_frames}/{stats.total_frames} frames processed")
    print(f"  → {stats.unique_objects} unique objects tracked")
    print(f"  → JSON detections: {out_json}")
    if include_video:
        print(f"  → Annotated video: {out_video}")
    review_outputs = render_review_outputs(
        str(video_path),
        frame_data,
        review_reel_path=out_review_reel,
        hold_seconds=getattr(args, "hold_seconds", 2.0),
        still_dir=out_still_dir,
        prompts=sam_targets,
    )
    if review_outputs.get("review_reel"):
        print(f"  → Review reel: {review_outputs['review_reel']}")
    if review_outputs.get("still_dir"):
        print(f"  → Annotated stills: {review_outputs['still_dir']}")
    print()

    # ── Step 2 (optional): Falcon Perception key-frame refinement ───────────────
    falcon_summary_parts = []
    crosscheck_results: list = []
    key_frames: list = []
    sam_by_frame: dict = {}
    key_frame_dims: dict = {}
    do_crosscheck = not getattr(args, "no_crosscheck", False)
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

                # Cross-engine inputs: SAM detections per frame index + key-frame
                # pixel dims (for Falcon normalized → pixel box conversion).
                sam_by_frame = {f["frame_index"]: f.get("detections", []) for f in frame_data}
                key_frame_dims = {fi: pf.size for fi, _ts, pf in key_frames}

                def _crosscheck_frame(fi: int, fp_results: list) -> None:
                    """Validate one frame's Falcon boxes against SAM (never fails the pipeline)."""
                    if not do_crosscheck:
                        return
                    try:
                        w, h = key_frame_dims.get(fi, (0, 0))
                        falcon_dets = falcon_to_dets(fp_results, w, h)
                        cc = crosscheck(sam_by_frame.get(fi, []), falcon_dets)
                        cc.frame_index = fi
                        crosscheck_results.append(cc)
                    except Exception as exc:
                        print(f"    WARNING: SAM-vs-Falcon cross-check failed on frame {fi}: {exc}")

                if getattr(args, "parallel_falcon", True):
                    # ── Parallel Falcon via ThreadPoolExecutor ───────────────
                    def _falcon_one(args_tuple):
                        fi, ts, pil_frame, query_str = args_tuple
                        results, st = detect(pil_frame, query_str, max_new_tokens=200)
                        return fi, ts, results, st

                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = {
                            executor.submit(_falcon_one, (fi, ts, pf, args.query)): (fi, ts)
                            for fi, ts, pf in key_frames
                        }
                        for future in as_completed(futures):
                            fi, ts, fp_results, fp_stats = future.result()
                            det_strs = [
                                f"  [{i+1}] score={r.score:.2f} cx={r.cx:.2f} cy={r.cy:.2f} h={r.h:.2f} w={r.w:.2f}"
                                for i, r in enumerate(fp_results[:10])
                            ]
                            falcon_summary_parts.append(
                                f"Frame {fi} (t={ts:.2f}s) — Falcon '{args.query}' "
                                f"({fp_stats.total_ms:.0f}ms, {len(fp_results)} results):\n"
                                + ("\n".join(det_strs) if det_strs else "  [no detections]")
                            )
                            print(f"  Frame {fi} (t={ts:.1f}s): {len(fp_results)} Falcon detections — {fp_stats.total_ms:.0f}ms [parallel]")
                            _crosscheck_frame(fi, fp_results)
                    print()
                else:
                    # ── Sequential Falcon (fallback) ─────────────────────────
                    for fi, ts, pil_frame in key_frames:
                        fp_results, fp_stats = detect(pil_frame, args.query, max_new_tokens=200)
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
                        _crosscheck_frame(fi, fp_results)
                    print()
            else:
                print(f"[2/{total_steps}] Falcon Perception — no frames with SAM detections to refine; skipping.")
                print()
        if crosscheck_results:
            print(f"  Cross-check (SAM vs Falcon, IoU ≥ 0.5):")
            for cc in sorted(crosscheck_results, key=lambda r: r.frame_index):
                print(f"    frame {cc.frame_index}: matched {cc.matched} · sam-only {cc.sam_only} · "
                      f"falcon-only {cc.falcon_only} (agreement {cc.agreement:.2f})")
            agg = summarize(crosscheck_results)
            print(f"  → Cross-check aggregate: {agg['frames']} frames · agreement {agg['agreement']*100:.0f}% · "
                  f"{agg['matched']} matched · {agg['sam_only']} SAM-only · {agg['falcon_only']} falcon-only")
            print()
        step_gemma = f"[3/{total_steps}]"
    else:
        step_gemma = f"[2/{total_steps}]"

    # ── Step 2b (optional): LFM grounding third opinion on the same key frames ──
    # Asks the local LFM2.5-VL where the SAM targets are on each key frame,
    # parses the boxes from its text reply, and cross-checks them against
    # SAM's detections at the same frame index. Opt-in via --lfm-ground; a
    # missing model never fails the pipeline (one warning line, then on).
    lfm_grounding_results: list = []
    if getattr(args, "lfm_ground", False):
        if not key_frames:
            print("  LFM grounding: no key frames available (needs --falcon-refine); skipping.")
        else:
            try:
                from . import grounding
                from . import vlm_registry

                vlm_registry.set_model("lfm")
                grounding_prompt = grounding.build_grounding_prompt(sam_targets)
                print("  LFM grounding check (SAM vs LFM, IoU ≥ 0.3):")
                for fi, _ts, pil_frame in key_frames:
                    reply = vlm_registry.ask(grounding_prompt, image=pil_frame)
                    lfm_boxes = grounding.parse_grounding_boxes(
                        reply, pil_frame.size[0], pil_frame.size[1]
                    )
                    gc = grounding.grounding_crosscheck(sam_by_frame.get(fi, []), lfm_boxes)
                    gc.frame_index = fi
                    lfm_grounding_results.append(gc)
                    print(f"    frame {fi}: {len(lfm_boxes)} LFM box(es) · matched {gc.matched} · "
                          f"sam-only {gc.sam_only} · lfm-only {gc.falcon_only} (agreement {gc.agreement:.2f})")
                agg_lfm = summarize(lfm_grounding_results)
                print(f"  → LFM grounding aggregate: {agg_lfm['frames']} frames · "
                      f"agreement {agg_lfm['agreement']*100:.0f}% · {agg_lfm['matched']} matched · "
                      f"{agg_lfm['sam_only']} SAM-only · {agg_lfm['falcon_only']} lfm-only")
                print()
            except Exception as exc:
                print(f"  LFM grounding unavailable: {exc}")

    # Step 3: Remote Gemma 4 reasoning
    if not gemma_available():
        print("WARNING: Gemma 4 server unreachable — detections saved but report not generated.")
        print(f"  Check: curl http://100.72.41.118:8080/v1/models")
        print(f"\n=== Pipeline complete (partial) ===")
        return

    print(f"{step_gemma} Gemma 4 e2b (Ollama) — reasoning about {video_path.name}...")
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
        f"User query: {semantic_question}",
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

    if crosscheck_results:
        try:
            agg = summarize(crosscheck_results)
            summary_parts.append(
                f"Cross-engine check (SAM vs Falcon) on {agg['frames']} key frames: "
                f"agreement {agg['agreement']*100:.0f}%, {agg['matched']} matched, "
                f"{agg['sam_only']} SAM-only, {agg['falcon_only']} Falcon-only."
            )
        except Exception as exc:
            print(f"  WARNING: cross-check summary for Gemma failed: {exc}")

    if lfm_grounding_results:
        try:
            agg_lfm = summarize(lfm_grounding_results)
            summary_parts.append(
                f"LFM grounding check on {agg_lfm['frames']} key frames: "
                f"agreement {agg_lfm['agreement']*100:.0f}%, {agg_lfm['matched']} matched, "
                f"{agg_lfm['sam_only']} SAM-only, {agg_lfm['falcon_only']} lfm-only."
            )
        except Exception as exc:
            print(f"  WARNING: LFM grounding summary for Gemma failed: {exc}")

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
            f"Focus on: {semantic_question}. Identify individual objects, their movement patterns, "
            f"anomalies, and anything the operator should act on."
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
    print(f"  Detection JSON  : {out_json}")
    if include_video and out_video:
        print(f"  Annotated video : {out_video}")
    if out_review_reel:
        print(f"  Review reel    : {out_review_reel}")
    print(f"  Annotated stills: {out_still_dir}")
    if args.report:
        print(f"  Field report   : {out_report}")

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
# cmd_fastscan
# ──────────────────────────────────────────────────────────────────────────────
def cmd_fastscan(args: argparse.Namespace) -> None:
    """Fast Falcon-only scan — relevance answer in seconds."""
    import json
    from pathlib import Path

    from .frame_selector import score_frames

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== FastScan ===")
    print(f"Video: {video_path}")
    print(f"Query: '{args.query}'")
    print(f"Scoring up to {args.max_frames} frames at {args.resolution}p...")
    print()

    result = score_frames(
        str(video_path),
        args.query,
        sample_every_n_seconds=args.every,
        max_frames=args.max_frames,
        resolution=args.resolution,
        min_relevance=args.min_relevance,
    )

    print(f"  QUICK ANSWER: {result.quick_answer}")
    print(f"  {result.frames_scored} frames scored in {result.duration_s:.1f}s total video")
    print(f"  Relevant: {'YES' if result.is_relevant else 'NO'}")
    if result.regions:
        print(f"  Regions of interest:")
        for r in result.regions:
            print(f"    [{r.start_time:.1f}s – {r.end_time:.1f}s] {r.label} "
                  f"(relevance: {r.avg_relevance:.2f})")

    if args.output:
        Path(args.output).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\n  JSON saved to: {args.output}")

    print(f"\n=== FastScan complete ===")


# ──────────────────────────────────────────────────────────────────────────────
# cmd_pilot_eval
# ──────────────────────────────────────────────────────────────────────────────
def cmd_pilot_eval(args: argparse.Namespace) -> None:
    """Replay-based pilot evaluation against a ground-truth event label file."""
    import json
    from pathlib import Path

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"ERROR: ground-truth file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    try:
        ground_truth = json.loads(gt_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: ground-truth file is not valid JSON: {gt_path} ({exc})",
              file=sys.stderr)
        sys.exit(1)

    from .pilot_eval import run_pilot_eval

    try:
        report = run_pilot_eval(
            str(video_path),
            ground_truth,
            tolerance_s=args.tolerance_s,
            sample_every_n_seconds=args.sample_every,
            max_frames=args.max_frames,
            resolution=args.resolution,
            min_relevance=args.min_relevance,
            evidence_dir=args.evidence_dir,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    for line in report.summary():
        print(line)

    if args.report:
        report.save(args.report)
        print(f"\n  Report JSON saved to: {args.report}")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VisionBrain — Vision AI on Apple Silicon",
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
    p.add_argument("--query", required=True, help="Natural-language expression, e.g. 'person'")
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
    p.add_argument("--prompts", required=True, nargs="+", help="Text prompts, e.g. person car bicycle")
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
    p.add_argument("--json-output", help="Also write per-frame detections JSON at this path")
    p.add_argument("--supervision", action="store_true",
                   help="Render with supervision annotators (mask/box/label)")
    p.add_argument("--persistent-ids", action="store_true",
                   help="ByteTrack persistent tracker IDs across occlusions")
    p.add_argument("--adaptive-motion", action="store_true",
                   help="Skip detection on low-motion frames")
    p.add_argument("--motion-threshold", type=float, default=0.03,
                   help="Grey-delta threshold for adaptive motion skip")
    p.add_argument("--propagate", type=int, default=0,
                   help="Propagate last detection forward N frames after each detect")

    # analyze
    p = sub.add_parser("analyze", help="Full pipeline: SAM 3.1 track → Gemma 4 reasoning → report")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--query", required=True, help="Natural-language query (e.g. 'vehicles near the gate')")
    p.add_argument("--prompts", nargs="+", help="SAM 3.1 text prompts to track (default: use --query)")
    p.add_argument("--output", help="Output video path (requires --include-video, defaults to disabled)")
    p.add_argument("--include-video", action="store_true",
                   help="Generate annotated MP4 video (disabled by default; use to re-enable)")
    p.add_argument("--json-output", help="Output JSON path for per-frame detections")
    p.add_argument("--report-output", help="Output text report path")
    p.add_argument("--review-reel-output", help="Output keyframe review reel path")
    p.add_argument("--hold-seconds", type=float, default=2.0,
                   help="Seconds to hold each analyzed frame in the review reel (default 2.0)")
    p.add_argument("--still-dir", help="Directory for exact annotated analyzed frames")
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
    p.add_argument("--no-crosscheck", action="store_true",
                   help="Skip SAM-vs-Falcon cross-validation on key frames")
    p.add_argument("--lfm-ground", action="store_true",
                   help="Ask the local LFM VLM to ground targets on key frames and cross-check against SAM")
    # ── Fast-path + adaptive ─────────────────────────────────────
    p.add_argument("--fast", action="store_true",
                   help="Run fast-path Falcon scan first, return quick answer immediately")
    p.add_argument("--fast-output", help="Write fast-scan JSON result to this path")
    p.add_argument("--adaptive", action="store_true",
                   help="Enable adaptive SAM: motion-guided skip + relevance filter")
    p.add_argument("--motion-threshold", type=float, default=0.03,
                   help="Frame delta threshold for motion skip (default 0.03, lower = more sensitive)")
    p.add_argument("--propagate", type=int, default=0,
                   help="Propagate masks forward N frames after each detect (default 0=off)")
    p.add_argument("--relevance-filter", action="store_true",
                   help="Skip frames where Falcon fast-scan scored relevance below threshold")
    p.add_argument("--parallel-falcon", action="store_true", default=True,
                   help="Process Falcon key-frames in parallel with ThreadPoolExecutor (default: True)")
    p.add_argument("--sequential-falcon", dest="parallel_falcon", action="store_false",
                   help="Disable parallel Falcon processing")
    # ── Chunking (large video support) ──────────────────────────
    p.add_argument("--chunk-duration", type=int, default=0,
                   help="Process video in chunks of N seconds to avoid OOM (0=auto, default: 0 for auto-detect)")
    p.add_argument("--chunk-overlap", type=int, default=3,
                   help="Seconds of overlap between chunks for tracker ID stitching (default 3)")

    # fastscan
    p = sub.add_parser("fastscan", help="Fast Falcon-only scan: quick relevance answer in seconds")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--query", required=True, help="Natural-language query (e.g. 'person')")
    p.add_argument("--every", type=float, default=5.0,
                   help="Sample one frame every N seconds (default 5)")
    p.add_argument("--max-frames", type=int, default=60,
                   help="Maximum frames to score (default 60)")
    p.add_argument("--resolution", type=int, default=360,
                   help="Falcon resolution (default 360 — low-res for speed)")
    p.add_argument("--min-relevance", type=float, default=0.2,
                   help="Minimum relevance to count as a region (default 0.2)")
    p.add_argument("--output", help="Write structured JSON result to this path")

    # pilot-eval
    p = sub.add_parser("pilot-eval",
                       help="Replay-based evaluation: missed events, false alerts, "
                            "latency and coverage against a labeled video")
    p.add_argument("--video", required=True, help="Recorded video path to replay")
    p.add_argument("--ground-truth", required=True,
                   help="Ground-truth event label JSON path")
    p.add_argument("--report", help="Write the evaluation report JSON to this path")
    p.add_argument("--evidence-dir",
                   help="Directory to save supporting frames as JPEG evidence")
    p.add_argument("--tolerance-s", type=float, default=3.0,
                   help="Matching tolerance around each event window, seconds (default 3.0)")
    p.add_argument("--sample-every", type=float, default=5.0,
                   help="Sample one frame every N seconds (default 5)")
    p.add_argument("--max-frames", type=int, default=60,
                   help="Maximum frames to score (default 60)")
    p.add_argument("--min-relevance", type=float, default=0.2,
                   help="Minimum relevance for a frame to count as a detection (default 0.2)")
    p.add_argument("--resolution", type=int, default=360,
                   help="Falcon resolution (default 360 — low-res for speed)")

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
        "fastscan": cmd_fastscan,
        "pilot-eval": cmd_pilot_eval,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
