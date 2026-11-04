# VisionBridge landing page — demo site

The marketing landing page for VisionBridge: a roof / wall / perimeter intrusion-watch system
that reads existing CCTV and drone feeds, detects a person on a surface, draws a pixel-accurate
mask, and pushes the annotated frame to field phones.

This directory was committed as a **frozen version** on 18 Sep 2026. `MANIFEST.sha256` lists every
file's hash; the same manifest is kept at `~/visionbridge-plates/_versions/v3-20260918-MANIFEST.sha256`.

## Which file is which

| file | what it is |
|---|---|
| `site/index.html` | v1 — the first build (ruf preserved, superseded) |
| `site/index-v2.html` | v2 — the version live before this commit |
| **`site/index-v3.html`** | **v3 — the approved version. This is the one to look at.** |

v3 keeps v1 and v2 byte-for-byte; nothing was overwritten. The site's own server (`tools/serve.py`)
maps `/` → `index-v2.html` and serves the rest by filename, so v2 remains the default until that
one line is changed.

## Look at it

```bash
cd site && python3 ../tools/serve.py      # serves on 0.0.0.0:8899
# then open http://127.0.0.1:8899/index-v3.html
```

## What v3 changed

- **Four inline SVG plates redrawn** to a written specification (`plates/SPEC.md`): a pipeline
  plate, a drone-fleet plan view, a four-mode deployment plate, and a surfaces elevation.
- **Two real defects in the inherited plate 2** fixed: two coverage circles ran outside the frame
  (silently clipped, which made a false claim about coverage extent), and two captions collided.
- **The overlap lenses in plate 2 were degenerate** — both arcs selected the same circle centre, so
  the path enclosed 0 u² and rendered as a bare arc line. Fixed to a measured 1822 u² / 1388 u².
- **The numbered card row** became a real ordered list with tabular `01–04` ordinals, sharing a
  flat floor.
- **The readout band** now separates what was *measured on this machine* (Plex Mono) from what is
  *supported in the architecture* (Plex Sans) — and moved "under 2 s after mask" out of measured
  type, because no push-path latency exists in `docs/falcon_results.json`.
- **Plate legibility at phone widths**: a 1120-unit plate at a 390px viewport rendered 12.5px mono
  at 3.8px. Below 760px the plate now keeps a legible size and `.diagram` scrolls, mirroring the
  page's existing treatment of its data table.

## Evidence, and its limits

`plates/VERIFICATION.md` and `plates/VERIFICATION-PASS2.md` record what was measured and how,
including the defects that were found by re-measuring work that had already been declared verified.
`docs/falcon_results.json` is the source of every latency figure on the page.

**Nobody has looked at these plates.** `vision_analyze` returns HTTP 404 on the build machine and
the only local vision model is a 3B that contradicted itself on size comparisons. Every claim in
the verification documents is geometry, bounding boxes, contrast ratios, DOM state or gate
counters. Whether the drawings are *beautiful* is unverified.

## Rebuilding

```bash
python3 plates/build-v3.py                 # reassemble site/index-v3.html from plates/v3/
plates/harness/runall.sh plates/v3 <out> 1440   # per-plate gate (also 768 / 390)
node plates/tools/pagecheck.mjs <url>      # whole-page: overflow, collisions, images, ids
```

## Deliberately not committed

`shots/` (31 MB) and `scenes/` (5.1 MB) — QA screenshots and source scene renders. The three final
composited images the page uses are in `site/assets/`.
