# VERIFICATION — VisionBridge plate system v3

**Date:** 18 Sep 2026 · **Orchestrator:** Hermes (`@hermes`) · **Artifact:** `~/.hermes/workspace/visionbridge-demo/site/index-v3.html`

Scope of this job: validate, then professionally re-finish the SVG "info card" plates and the two
adjacent HTML components on the VisionBridge roof/wall/perimeter intrusion-watch landing page.
v2 (`index-v2.html`) and v1 (`index.html`) were **not modified** — v3 is a separate file.

---

## 1. What was established about the baseline before any work

The page's own v2 design notes claimed a clean verification pass. Re-running the numbers found
**that was not true**. Measured on v2 at 1440:

| plate | PASS | texts | collisions | outOfFrame |
|---|---|---|---|---|
| plate1-pipeline | true | 20 | 0 | 0 |
| **plate2-fleet** | **FALSE** | 7 | **1** | **2** |
| plate3-modes | true | 15 | 0 | 0 |
| plate4-elevation | true | 9 | 0 | 0 |

Two coverage circles in plate 2 ran past the top of the frame and were silently clipped by the
browser — a false claim about how much ground is watched, on a page whose entire credibility
argument is that it does not overclaim. The caption collision the v2 notes claimed to have fixed
was still present (103.7 × 0.1 units).

---

## 2. Final measured state — all four plates, three widths

Every figure below was measured by the orchestrator from the files on disk, not taken from a
specialist's report.

| Plate | viewBox | labels | red | 1440 | 768 | 390 |
|---|---|---|---|---|---|---|
| 1 — pipeline / how it works | 350 → **472** | 20 → **24** | 3 | PASS | PASS | PASS |
| 2 — drone fleet plan view | 400 → **420** | 7 → **18** | 2 | PASS | PASS | PASS |
| 3 — the four deployment modes | 250 → **272** | 15 → **12** | 0 | PASS | PASS | PASS |
| 4 — surfaces elevation | 360 (kept) | 9 → **11** | 4 | PASS | PASS | PASS |

All eight hard counters are **0** on every plate at every width: `textCollisions`, `outOfFrame`,
`contrastFails`, `missingRoleOrAria`, `externalRefs`, `classesNotInPage`, `dupIds`,
`missingMarkers`. Contrast minimum 4.85 on all plates.

Red (`#c8361a`) placement is semantically correct throughout: plate 1 = 3 marks (detection box,
mask glyph, person glyph in the phone screen); plate 2 = 2 (detection dot fill + pulse stroke, one
detection); **plate 3 = 0** (a hosting-topology plate depicts no detection, so any red would be
decoration, which the contract forbids); plate 4 = exactly 4, one per detection.

## 3. Page-level verification (the assembled file, not the fragments)

| Check | 1440 | 768 | 390 |
|---|---|---|---|
| horizontal overflow | 0 px | 0 px | 0 px |
| text-vs-text collisions | 0 | 0 | 0 |
| leaf text boxes tested | 212 | 212 | 212 |
| images loaded | 3/3 | 3/3 | 3/3 |
| `svg[role=img]` | 4/4, no missing `aria-label` | same | same |
| duplicate DOM ids | none | none | none |

- `slopscan.mjs index-v3.html` → **0 fails, 0 warns, 0 suppressed**
- Only console error is the browser's automatic `favicon.ico` 404 at first load — pre-existing,
  present in v2 as well.

## 4. Component-level changes worth recording

**Card row (`.beats`)**
- `ONE/TWO/THREE/FOUR` → `01/02/03/04`: IBM Plex Mono digits are tabular (identical advance
  width); the words (3/3/5/4 chars) can never be. This was the root cause of the eyebrow misalignment.
- `<div>` children → `<ol class="beats" role="list">` + `<li>`, so the page's one numbered sequence
  is a real list for assistive tech.
- Card floor: v2 measured ragged by **23.19 px** at 768. v3 is flat at 1440 (0 px) and internally
  flat per band at 768. Residual: the two 768px bands differ by 22.25 px.
- Broken border rules at the 2-up breakpoint: v2 measured `rules 0/1` (the `!important` hack);
  v3 measures `1/1`.

**Readout band (`.readout`)**
- v2 measured `0 measured / 6 architecture` rows, all six in identical mono type. v3: **2 measured
  (Plex Mono) / 4 architecture (Plex Sans)**, with the two categories named in the band itself.
- Verified against source: `falcon_lowres_detect.warm_ms = 1281`, `sam31.segment_warm_ms = 1807`,
  and **no push/alert/delivery latency key exists anywhere in `out/falcon_results.json`**. So
  "under 2 s after mask" is a stage budget, not a benchmark, and was correctly moved out of
  measured type.
- Band alignment: v2 sat **130 px** off its own section at 1440 (24 px at 768). v3 measures 0.
- Signal-red properties in the components: **40 → 0**. Red now means detection and nothing else.

## 5. Corrections and errors owned

- **A false defect the orchestrator invented.** A cloud glyph in plate 3 was briefed to a
  sub-agent as "measured … 52 wide × 52 tall … 35% of the mass". That figure came from reading the
  path's **arc junction coordinates**; arcs bulge far past their junctions. Measured properly, the
  old cloud was already **96.06 × 79.99** — parity already existed. The specialist caught the
  discrepancy independently and said so in its report. The change is retained only because it has
  genuine robustness value: the old cloud's arc chords were 51.857 against a 52-unit maximum —
  within **0.3%** of the threshold at which SVG silently rescales the radius (new margin 3–4%).
- **Tooling bugs found and fixed in the orchestrator's own harness** (fixed the tools, never the
  deliverables):
  - assembler nested the card row inside itself (fragment is an `<ol>`, not a `<div>`);
  - integrity checker counted an `id="how"` inside a documentation comment as a duplicate DOM id;
  - page checker reported 11 phantom collisions by comparing inline runs on the *same line*, whose
    bounding rects necessarily overlap by one line-height. Now only compares text in different
    block containers.

## 6. Self-declared residual weaknesses (not fixed)

- **plate1:** its specialist **timed out before filing a report**, so it has the weakest provenance
  of the four — no account of its reasoning or its own known weaknesses. Verified by the orchestrator
  instead. One real defect was found and fixed here: the phone glyph's home-button rect ran 6 units
  past the phone body (invisible to the gate, which only tests the viewBox, not container overflow).
- **plate2:** the shared-coverage overlap "lens" is filled `#e4e9ec`, the same value as the building,
  so where a lens crosses the roof the overlap reads as a thin outline rather than a covered area.
  Candidate fixes: hatch the lens, or give it a distinct tint from the token set.
- **plate3:** the cloud carries ~17% less *stroked ink* than its three siblings (bounding boxes are
  exactly equal at 96×80); three arc junctions sit off the 4-unit grid; at 390 px the captions render
  ~3.8 px tall — legible texture, not readable text.
- **plate4:** detections D2 (on the wall) and D4 (inside the line) both necessarily sit in the right
  third of the site, so the lower-right remains the densest quadrant.

## 7. The one thing nobody has done: looked at it

**There is no working image-analysis backend on this machine** (`vision_analyze` returns HTTP 404;
the local 3B vision model returned near-empty output and fabricated one defect that was verified
absent from the markup). Every claim in this document is geometry, bounding boxes, contrast ratios,
DOM state and gate counters. The plates pass every measurable test, but whether the new cloud reads
as a cloud, or the 96-unit building as a building, is **unverified by eye**. That judgement needs a
human, or a working vision backend.

## 8. Reproduce / rebuild

```bash
P=~/visionbridge-plates
$P/harness/runall.sh $P/v3 $P/shots/gate-v3-final 1440     # per-plate gate, all widths
$P/harness/runall.sh $P/v3 $P/shots/gate-v3-final-768 768
$P/harness/runall.sh $P/v3 $P/shots/gate-v3-final-390 390
node $P/glyphbox.mjs $P/v3/plate3-modes.svg p3-mode-site p3-mode-vm p3-mode-instance p3-mode-mobile
node $P/cloudcmp.mjs                                        # old vs new cloud, bbox + stroked length
python3 $P/build-v3.py                                      # reassemble site/index-v3.html
node $P/pagecheck.mjs http://127.0.0.1:8899/index-v3.html    # whole-page, three widths
node ~/.hermes/skills/creative/auteur/scripts/slopscan.mjs <site>/index-v3.html
```

## 9. Artifact hashes (md5, after the final write)

```
index-v3.html       a2493cf64451e8486282233bc137481b
plate1-pipeline.svg 58d0b0ef0bce5c17dc214a7e3856c662
plate2-fleet.svg    d3c2187cfbcee9b3dbd34f926aafac69
plate3-modes.svg    0af1931e3df807c2304974209513e4f4
plate4-elevation.svg 6e35c3cbacb4f1c096a03376c1a99118
beats-cards.html    aefadc5adb8f79b24f349ab27a46abd8
readout-band.html   c7a03d4ddfb1f4b4f7c25793d1c65899
beats.css           54fd4c5147abcb63a03ad27968aa9225
```

## 10. Open decision

Plate 1 carries an in-SVG title strip (`how it works · three inputs, one core, one frame out`) that
its specialist added on its own initiative; plates 2, 3 and 4 have none. Options: **drop it from
plate 1** (redundant with the section `<h2>`; requires re-laying out ~50 coordinates so its 84-unit
dead top margin returns to 28) or **add a matching strip to the other three**. Held pending the
owner's call rather than guessed at, because it requires blind re-layout of verified artifacts.

---

**Superseded in part by `VERIFICATION-PASS2.md`**, which corrected a misdiagnosis in plate 2 (the
overlap lenses enclosed zero area — a worse defect than the declared colour problem), fixed
phone-width plate legibility, and re-validated all four plates at three widths. The plate 2 and
`index-v3.html` hashes in §9 above predate those fixes.
