# roboflow/supervision — deep recon

*Recon date: 2026-09-08. Investigated against primary sources (GitHub repo/releases/docs/PyPI) plus a
verified audit of VisionBrain's own usage. Companion notes to `SPEC.md`; usage facts below were checked
against the local tree and the `.venv` install (supervision 0.28.0, pin `supervision>=0.28,<0.30` at
`pyproject.toml:22`).*

## TL;DR

Supervision is Roboflow's MIT-licensed "reusable computer vision tools" library — a model-agnostic
`sv.Detections` container with converters, ByteTrack tracking, 24 annotators, zone counters, dataset
I/O, and metrics. It is still actively maintained (three patch releases in Aug–Sep 2026; latest
**0.30.2**, 2026-09-04), but Roboflow's center of gravity has shifted: docs now recommend **RF-DETR**
as the paired model, and **tracking has been spun out into a new `trackers` package** — `sv.ByteTrack`
is deprecated since 0.28.0 and **scheduled for removal in 0.31.0**, which is already in development
(`0.31.0.dev0`).

For VisionBrain specifically: the `supervision>=0.28,<0.30` pin is well-aimed — 0.30.0 is the
disruptive release (OpenCV dropped, new required `av` dep, Python ≥3.10, several behavior changes).
But the sharper cliff is **0.31.0, which deletes the in-repo ByteTrack** that
`supervision_bridge.py:19` deep-imports. The live field-hub path deliberately doesn't use supervision
tracking at all, and the bridge's `ByteTrack` wrapper is exactly the seam where the `trackers` package
would slot in. Also found: one likely-unnecessary hack (`to_compact`'s id-keyed side-cache — upstream
`Detections.mask` accepts a `CompactMask` directly since 0.28.0).

## 1. What it is, and where it stands

**Purpose.** "We write your reusable computer vision tools" — a glue layer between any
detector/segmenter and the plumbing every vision app needs: a unified detections container, tracking,
annotation, zone analytics, slicing, video I/O, dataset formats, and benchmark metrics.
([repo](https://github.com/roboflow/supervision), [PyPI](https://pypi.org/project/supervision/))

**Health.** ~49.9k stars, MIT, not archived, default branch `develop`, last push 2026-09-08. Piotr
Skalski remains maintainer of record. No official maintenance-mode or feature-freeze announcement was
found, but the strategic drift is unmistakable: 0.30.1 rewrote all docs/examples from Ultralytics YOLO
to RF-DETR, and the docs homepage states RF-DETR is the recommended model.
([docs](https://supervision.roboflow.com/latest/), [0.30.1 release](https://github.com/roboflow/supervision/releases/tag/0.30.1))

**Release timeline** ([Releases](https://github.com/roboflow/supervision/releases), PyPI):

| Version | Date | Headline |
|---|---|---|
| 0.26.0 | 2025-07-16 | py3.8 dropped; mAP aligned to pycocotools; `LMM`→`VLM` |
| 0.26.1 | 2025-07-23 | mAP size-evaluation / ID=0 fixes |
| 0.27.0 | 2025-11-16 | `xyxy_to_mask`, Qwen3-VL parsing; InferenceSlicer overlap reworked to pixels; `keypoint`→`key_points` deprecation |
| 0.27.0.post2 | 2026-03-14 | follow-up fix |
| 0.28.0 | 2026-04-30 | **`CompactMask`**, `from_sam3`, **`ByteTrack` deprecated**, `VideoInfo.fps` → float |
| 0.29.0 | 2026-06-15 | Oriented bounding boxes everywhere; keypoint ellipse annotators |
| 0.29.1 | 2026-06-23 | `KeyPoints.with_nms()`; metric FP-counting fixes |
| 0.30.0 | 2026-08-04 | **OpenCV no longer required** (private NumPy/Pillow `_cv2` backend); `av>=14.2` now required; Python ≥3.10; Soft-NMS |
| 0.30.1 | 2026-08-24 | Numeric-precision fixes; lazy PyAV import (macOS libavdevice crash); RF-DETR docs |
| **0.30.2** | **2026-09-04** | Integer-overflow fixes (`box_area` → float64); deterministic InferenceSlicer merge order |
| 0.31.0.dev0 | in develop | **Where `sv.ByteTrack` removal lands** |

**Dependencies** (current): `av>=14.2` (new in 0.30), `defusedxml`, `matplotlib`, `numpy`, `pillow`,
`pydeprecate`, `pyyaml`, `requests`, `scipy`, `tqdm`. **OpenCV is gone** as of 0.30.0 — replaced by a
private `_cv2/` facade reimplementing every OpenCV call in NumPy/Pillow, with PyAV for video.
([pyproject.toml](https://github.com/roboflow/supervision/blob/develop/pyproject.toml),
[0.30.0 release](https://github.com/roboflow/supervision/releases/tag/0.30.0))

**The `trackers` spin-out.** Tracking investment now lives in
[`pip install trackers`](https://pypi.org/project/trackers/) (Apache-2.0, v2.6.0, 2026-08-06,
maintained by Skalski): SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU, McByte — all
"supervision.Detections native" with a unified `update(detections, frame=None)` API (note the rename
from `update_with_detections`).
([deprecated.md](https://github.com/roboflow/supervision/blob/develop/docs/deprecated.md))

## 2. Code architecture

Package moved to a `src/` layout on `develop`. Top-level modules: `detection/` (core `Detections`,
`CompactMask`, `LineZone`, VLM parsing, tools, utils), `tracker/` (deprecated ByteTrack internals),
`annotators/`, `key_points/` (plus a deprecated `keypoint/` shim), `geometry/` (Point/Rect/Position),
`draw/` (Color/ColorPalette), `metrics/`, `dataset/` (YOLO/COCO/VOC/LabelMe/CreateML),
`classification/`, `assets/`, `utils/` (video/conversion/image), and the private `_cv2/` backend.
([source tree](https://github.com/roboflow/supervision/tree/develop/src/supervision))

**`Detections` core** ([detection/core.py](https://github.com/roboflow/supervision/blob/develop/src/supervision/detection/core.py)):

- Fields: `xyxy` (n×4), `mask` (dense `(n,H,W)` bool **or** `CompactMask` — the union is in the
  dataclass signature since 0.28), `confidence`, `class_id`, `tracker_id`, `data` (per-detection
  extras), `metadata` (collection-level).
- `CompactMask` (0.28+): RLE-encoded tight bbox crops, ~10–240× memory reduction;
  `from_dense`/`to_dense`/`crop`/`merge`/`resize`/`__array__`.
  ([compact_mask.py](https://github.com/roboflow/supervision/blob/develop/src/supervision/detection/compact_mask.py))
- ~17 `from_*` constructors (ultralytics, transformers, inference, sam, **sam3**, detectron2,
  mmdetection, easyocr, VLM text parsing, …). There is **no `from_xyxy`** — custom detections go
  through the plain constructor, which is exactly what VisionBrain's bridge does.
- Set ops: `with_nms`, `with_nmm`, `with_soft_nms` (0.30), `merge`, `select`,
  `get_anchors_coordinates`, `area`/`box_area`.
- Polygons are utilities, not a field: `mask_to_polygons`/`polygon_to_mask` in
  `detection/utils/polygons.py`.

**The rest in one pass:** `ByteTrack` (deprecated; params `track_activation_threshold=0.25`,
`lost_track_buffer=30`, `minimum_matching_threshold=0.8`, `frame_rate`, `minimum_consecutive_frames`);
24 annotators (Box/Mask/Label/Trace/Halo/Blur/HeatMap/RichLabel/…);
([annotators/core.py](https://github.com/roboflow/supervision/blob/develop/src/supervision/annotators/core.py));
`LineZone` (in/out counts, class-identity preserved across recycled IDs since 0.28) and `PolygonZone`
(`triggering_anchors`, `require_all_anchors` in 0.30); `InferenceSlicer` (pixel `overlap_wh`,
`batch_size`, GeoTIFF raster support in 0.30)
([inference_slicer.py](https://github.com/roboflow/supervision/blob/develop/src/supervision/detection/tools/inference_slicer.py));
video (`VideoInfo` with float fps, threaded `process_video`, `CSVSink`/`JSONSink`,
`DetectionsSmoother`); metrics (`MeanAveragePrecision`, `ConfusionMatrix`, Precision/Recall/F1 with
size buckets, OBB targets since 0.29).

## 3. Version deltas — and why VisionBrain's pin exists

The **0.30.0 "big five" breaking changes**
([release](https://github.com/roboflow/supervision/releases/tag/0.30.0),
[migration guide](https://github.com/roboflow/supervision/blob/develop/docs/how_to/opencv_migration.md)):

1. OpenCV optional — new hard dep `av>=14.2`, `sv.ImageWindow` replaces `cv2.imshow`, no more opencv
   extra.
2. Python ≥3.10 (3.9 dropped).
3. `JSONSink` emits native JSON types (was strings); `CSVSink` slicing aligned.
4. `mask_non_max_merge` now computes exact overlap (was downscaled approximation) — thresholds need
   re-tuning; positional args deprecated.
5. `Detections.merge()` on mixed dense+CompactMask now **returns a `CompactMask`** (breaks
   `isinstance(..., np.ndarray)` checks).

Then 0.30.1/0.30.2 quietly changed numeric behavior (`box_area`/`area` → float64 on overflow,
`xcycwh_to_xyxy`/`denormalize_boxes` no longer truncate ints, `box_iou` raises on complex dtypes).
Meanwhile the 0.28 deprecations — **ByteTrack removal**, `keypoint`→`key_points`, RLE shim moves —
were deferred from 0.30.0 to **0.31.0**. So `<0.30` buys: ByteTrack still present, OpenCV still
installed, and none of the 0.30.x dtype/behavior churn. Official deprecation catalog:
[docs/deprecated.md](https://github.com/roboflow/supervision/blob/develop/docs/deprecated.md).

## 4. How VisionBrain consumes it (verified locally)

Installed: **supervision 0.28.0** in `.venv` — the oldest end of the allowed range. Touchpoints:

- **`src/visionbrain/supervision_bridge.py`** — the chokepoint. Deep import
  `from supervision.tracker.byte_tracker.core import ByteTrack as _RealByteTrack` (line 19) — a
  private module path, the single most upstream-coupled line in the repo. A local `ByteTrack` wrapper
  (lines 29–59) suppresses the FutureWarning and re-exposes `update_with_detections`/`reset`. Three
  converters build `sv.Detections` from VisionBrain dataclasses (`detections_from_sam31`,
  `detections_from_falcon_masks` via pycocotools RLE, `detections_from_falcon_boxes`), all using the
  plain constructor with `xyxy/mask/confidence/class_id`. `to_compact`/`from_compact` (lines 227–298)
  swap dense masks for `CompactMask` using an id-keyed module-level cache with `weakref.finalize`
  eviction — the cache exists because the comment claims CompactMask "can't go in `data`".
  `to_legacy_dict` (line 305) has **no caller anywhere in the repo** — dead code or external-only.
- **`src/visionbrain/zones.py`** — `sv.LineZone`/`sv.Point` and `sv.PolygonZone` for the line/polygon
  counters; `viz.render_zone_overlay` reads `zone.start/.end` and `zone.polygon` attributes directly.
- **`src/visionbrain/viz.py`** — `sv.ColorPalette([sv.Color(...)])`, `MaskAnnotator`, `BoxAnnotator`,
  `LabelAnnotator`, `TraceAnnotator`.
- **`src/visionbrain/sam3_inference.py`** (`track_video_realtime`) — the full annotator stack + bridge
  ByteTrack for the CLI/realtime rendering path.
- **`src/visionbrain/live_engine.py`** (`_evaluate_triggers`) — lazily builds
  `sv.Detections(xyxy, tracker_id)` from live items **only** to feed `LineZoneCounter`; events go out
  on the wire as `line_cross`.
- **`src/visionbrain/web_app.py` / `cli.py`** — `--supervision` / `--persistent-ids` flags pass
  through to the track job.

Critically, **the live field-hub path does not use supervision for identity at all**: `LiveSamTracker`
uses mlx_vlm's `SimpleTracker` and `detection_core.PersistentTrackManager` is the canonical tracker
(per SPEC.md) — supervision ByteTrack only serves the CLI `track` path. `SIMPLIFICATION_SPEC.md`
freezes that canonical tracker and keeps the compact-mask cache cleanup; it plans no supervision
changes.

**Test coverage**: `TestSupervisionBridge` (4 tests) covers only the CompactMask cache
round-trip/eviction. The converters, the ByteTrack wrapper, and both zone counters are untested —
precisely the code that would break on an upgrade.

## 5. Cross-analysis — risks and openings

1. **The pin is correct for today, but the deadline is 0.31, not 0.30.** The wrapper's docstring says
   it "future-proofs against 0.30+" — it doesn't quite: the import at line 19 still hard-pins the
   private module path. Removal lands in 0.31.0, already in dev. The migration when it comes is small
   and localized: `pip install trackers`, swap the wrapper's internals to `ByteTrackTracker`, rename
   `update_with_detections` → `update`. The wrapper was designed as exactly this seam.
2. **The compact-mask side-cache is probably deletable.** Upstream's `Detections` dataclass accepts
   `mask: NDArray | CompactMask | None` since 0.28.0 — meaning `to_compact` could construct
   `sv.Detections(..., mask=compact)` directly and drop the entire `id()`-keyed cache +
   `weakref.finalize` machinery (and `from_compact` could call `mask.to_dense()` off the field). The
   local comment worries about `data`-dict validation, but the mask field is the sanctioned home.
   Worth a quick spike; 0.30's mixed-`merge()` behavior is unrelated to this usage. (Note:
   `SIMPLIFICATION_SPEC.md` currently says to *keep* the cache cleanup logic — revisiting that line
   would be part of the spike.)
3. **Nothing in 0.29.x tempts VisionBrain** (OBB, keypoint ellipses) — staying on 0.28.0 costs
   nothing for now; CI compatibility holds since supervision is a declared dep and the bridge tests
   construct real `sv.Detections`.
4. **`from_sam3` exists upstream but doesn't fit** — it parses official SAM3 output shapes (PCS/PVS),
   not the mlx-community MLX-layout outputs `sam3_inference` produces. The hand-rolled converters
   stay.
5. **Duplication note**: upstream `mask_to_polygons` overlaps VisionBrain's
   `detection_core.mask_to_polygon` — but keeping the latter dependency-free is deliberate
   (field-bridge importability), so no action.
6. **Zone analytics stay in supervision** (not moved to `trackers`), so `zones.py` and the live-engine
   trigger path are insulated from the tracking spin-out.

## 6. Docs-site deep dive — full capability & workflow catalog

*Second pass: the documentation site (https://supervision.roboflow.com/latest/) read top to bottom,
nav verified against `mkdocs.yml` on `develop` (2026-09-08).*

**How the docs are organized (2026 restructure).** The old notebook-style how-to hub is gone. Current
layout: a **Learn** tab of 10 how-to guides (`docs/how_to/*.md`), an auto-generated **Reference**
(mkdocstrings per-class pages), a **Cookbooks** gallery (15 notebooks under `docs/notebooks/`),
FAQ, a dedicated **Deprecated** page, and a changelog. The old "cheat sheet" page was removed — the
de-facto cheat sheets are now `docs/llms.txt` / `llms.full.txt` (AI-facing doc feeds; robots.txt
explicitly allows GPTBot/ClaudeBot/etc.), plus a v0.24-era task-based cheat sheet still hosted at
[roboflow.github.io/cheatsheet-supervision](https://roboflow.github.io/cheatsheet-supervision/).

### The 10 Learn guides (the workflow catalog)

| Guide | Workflow taught | Key APIs |
|---|---|---|
| Detect and Annotate | inference (RF-DETR / Inference / Ultralytics / Transformers tabs) → convert → annotate | `from_inference`, `from_ultralytics`, `from_transformers`, `BoxAnnotator`, `LabelAnnotator`, `MaskAnnotator`, `from_vlm` |
| Save Detections | frame generator → per-frame inference → sink | `get_video_frames_generator`, `CSVSink`, `JSONSink` (`append(dets, custom_data)`) |
| Filter Detections | NumPy boolean-indexing recipes on Detections | `dets[dets.class_id == 0]`, `.area` / `.box_area` semantics, `with_nms`, `PolygonZone.trigger` as a filter mask |
| Detect Small Objects (SAHI) | slice → per-slice callback → merge | `InferenceSlicer(callback=...)`, pixel `overlap_wh`, `batch_size`, deterministic merge order |
| Track Objects on Video | inference → tracker → labels → annotate → `process_video` | `sv.ByteTrack.update_with_detections` (deprecated banner → `trackers.ByteTrackTracker.update`), `TraceAnnotator`, `DetectionsSmoother`, `KeyPoints.as_detections()` |
| Process Datasets | load → split/merge → iterate → export | `DetectionDataset.from_coco/from_yolo/from_pascal_voc/from_labelme/from_createml`, `as_*`, `.split()`, iteration is memory-safe; augmentation delegated to Albumentations (no `sv.Augmenter`) |
| Benchmark a Model | test set → per-class remap → metrics | `supervision.metrics.MeanAveragePrecision(metric_target=MetricTarget.MASKS).update().compute()`, `F1Score`, size buckets, `ConfusionMatrix.benchmark(save_directory_path=...)` |
| Count in Zone | pick polygons → per-zone trigger → filter → annotate | `PolygonZone` (+`PolygonZoneAnnotator`, `require_all_anchors`), FAQ covers `LineZone` (requires `tracker_id`) and enter-once counting via a seen-ids set |
| Use Compact Masks (new) | RLE ingest → compact ops → annotate without materializing | `CompactMask.from_coco_rle`, `from_inference(compact_masks=True)`, `Detections.to_compact_masks()`, `annotator.requires_mask`, mixed-merge returns CompactMask |
| OpenCV Migration (new) | packaging/backend changes in 0.30 | `_cv2.BACKEND_NAME` check, one-wheel-family rule, `ImageWindow`, app-owned capture |

**The canonical pipeline the docs promote:** load media → model (RF-DETR first in every guide — its
`predict()` returns `sv.Detections` natively, "no conversion step") → filter → track (trackers
package) → annotate → zone/line counting → save (CSV/JSON) → benchmark.

**Cookbooks gallery** (15 notebooks): quickstart, annotate-video-with-detections,
count-objects-crossing-the-line, object-tracking, zero-shot detection with YOLO-World (×2),
blurring_faces, occupancy_analytics, serialise-to-csv/json, small-object-detection-with-sahi,
oriented-bounding-boxes, **compact-mask-sam3** (pairs `from_sam3` with `CompactMask`; documents that
`from_sam3` parses PCS+PVS formats with `class_id` = prompt index), download-supervision-assets,
diffusion-alignment eval. Repo `examples/` scripts: traffic_analysis, speed_estimation,
count_people_in_zone, time_in_zone, heatmap_and_track, compact_mask, tracking.

**New/unusual capabilities a downstream toolkit likely doesn't know:**
- `sv.Detections.to_compact_masks()` is a **built-in** converter, and `CompactMask.from_coco_rle(rles, xyxy, image_shape)` ingests COCO RLE **without ever allocating** the dense `(N,H,W)` array; `BaseAnnotator.requires_mask` tells you which annotators need masks materialized (only Mask/Polygon/Halo).
- Soft-NMS: `sv.Detections.with_soft_nms(...)` (0.30.0).
- Geospatial: `sv.WindowedRasterDataset` + InferenceSlicer for tiled GeoTIFF inference (`supervision[geotiff]`).
- Video helpers: `get_video_frames_generator(..., prefetch=N)` background decode, `process_video(preserve_audio=True)`, `load_image_from_url` with disk cache, `sv.ImageAssets`.
- `ConfusionMatrix.benchmark()` writes per-image TP/FP/FN/GT mosaics for error analysis.
- Stream-reuse `reset()` methods on TraceAnnotator/HeatMapAnnotator/DetectionsSmoother.
- FAQ is explicit that there is no `sv.Augmenter` and no webcam/camera-capture support (frames are caller-owned — which fits a pipeline that already feeds frames directly).

## 7. Ecosystem map

- **`trackers`** ([docs](https://trackers.roboflow.com/latest/), [repo](https://github.com/roboflow/trackers), PyPI v2.6.0 Aug 2026, beta): six clean-room algorithms — SORT (HOTA 58.4), ByteTrack (60.1), OC-SORT (61.9), C-BIoU (63.0), BoT-SORT (63.7), McByte (64.1, mask-conditioned) — all sharing `update(detections, frame=None)` and "speaking `sv.Detections` natively". Ships a CLI (`trackers track/eval/download` with CLEAR/HOTA/Identity metrics), Optuna tuning, HF Space playground. This is where supervision's tracking future lives.
- **RF-DETR** ([repo](https://github.com/roboflow/rf-detr), ICLR 2026): DINOv2-backbone detection/segmentation; the reason every docs example leads with it is `predict()` returns `sv.Detections` natively. Apache-2.0 base; Plus (XL/2XL) under a separate license.
- **`inference`**: Roboflow's self-hostable server/SDK (Apache-2.0 core) with Workflows; bridges in via `sv.Detections.from_inference`.
- **Maestro** and the docs.roboflow.com platform round out the About page's sibling list; `roboflow.com/supervision` now 301-redirects to GitHub, so the repo README is the marketing page.
- **Versioning posture**: Python ≥3.10 (classifiers 3.10–3.14); 2026 release cadence re-accelerated (0.28 Apr → 0.29 Jun → 0.30 Aug); deprecations now promise a **minimum 3-minor-release window** (policy set in 0.29.0), and the Deprecated page names exact removal versions — it is the changelog-of-record for planning upgrades.

## 8. Docs-pass implications for VisionBrain (adds to §5)

7. **The CompactMask cleanup has an even simpler shape than §5.2 suggested.** Upstream ships `Detections.to_compact_masks()` as a built-in, and `CompactMask.from_coco_rle(rles, xyxy, image_shape)` converts Falcon's COCO-RLE masks straight to compact form without materializing dense `(N,H,W)` arrays — the `detections_from_falcon_masks` path could skip pycocotools decode + dense stack entirely for memory-constrained use. The id-keyed side-cache hack has a first-party replacement (`mask=` field + `to_compact_masks()`), making the `SIMPLIFICATION_SPEC.md` "keep the cache cleanup" line worth revisiting after a spike.
8. **The ByteTrack exit unlocks algorithm choice.** Swapping the bridge wrapper's internals to `trackers.ByteTrackTracker` is a near drop-in (`update()` rename) and makes OC-SORT/BoT-SORT/C-BIoU available behind the same `sv.Detections` contract — relevant to the CLI `track --persistent-ids` path quality, not just deprecation hygiene.
9. **No Python obstacle to a future 0.30+ move**: the `.venv` is Python 3.14, inside supervision 0.30.x's supported range; the only 0.30 blockers for VisionBrain remain behavioral (OpenCV-optional packaging is irrelevant since VisionBrain renders via numpy/PIL and its own canvas pipeline).
10. **Live-path fit**: supervision 0.30+ is explicitly frame-caller-owned (no capture), matching VisionBrain's architecture; and `annotator.requires_mask` mirrors the dashboard's existing paint-polygons-else-boxes fallback logic.

## Sources

[repo](https://github.com/roboflow/supervision) ·
[docs (latest)](https://supervision.roboflow.com/latest/) ·
[mkdocs.yml nav](https://raw.githubusercontent.com/roboflow/supervision/develop/mkdocs.yml) ·
[how-to guides](https://supervision.roboflow.com/latest/how_to/detect_and_annotate/) ·
[cookbooks](https://supervision.roboflow.com/latest/cookbooks/) ·
[llms.txt](https://supervision.roboflow.com/llms.txt) ·
[FAQ](https://supervision.roboflow.com/latest/faq/) ·
[trackers docs](https://trackers.roboflow.com/latest/) ·
[trackers repo](https://github.com/roboflow/trackers) ·
[RF-DETR repo](https://github.com/roboflow/rf-detr) ·
[releases](https://github.com/roboflow/supervision/releases) ·
[PyPI supervision](https://pypi.org/project/supervision/) ·
[PyPI trackers](https://pypi.org/project/trackers/) ·
[docs](https://supervision.roboflow.com/latest/) ·
[deprecated.md](https://github.com/roboflow/supervision/blob/develop/docs/deprecated.md) ·
[detection/core.py](https://github.com/roboflow/supervision/blob/develop/src/supervision/detection/core.py) ·
[compact_mask.py](https://github.com/roboflow/supervision/blob/develop/src/supervision/detection/compact_mask.py) ·
[annotators/core.py](https://github.com/roboflow/supervision/blob/develop/src/supervision/annotators/core.py) ·
[inference_slicer.py](https://github.com/roboflow/supervision/blob/develop/src/supervision/detection/tools/inference_slicer.py) ·
[0.30.0 release](https://github.com/roboflow/supervision/releases/tag/0.30.0) ·
[opencv_migration.md](https://github.com/roboflow/supervision/blob/develop/docs/how_to/opencv_migration.md)
