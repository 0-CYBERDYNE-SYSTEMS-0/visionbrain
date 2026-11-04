# DESIGN NOTES — v2 (edit route, per auteur §edit)

**What this is.** An edit of `index.html` (v1) under the auteur `edit` protocol: read the
committed contract first (`DESIGN-NOTES.md`), reuse its tokens, section-opening patterns and
motion families, then re-run the gate. v1 is preserved byte-for-byte at `index.html` and at
`../_backups/index-v1-20260917-144828.html` (md5 `dc86a9ba577083111abbdec06932bdd6`, verified).

**New file:** `index-v2.html`. Nothing was deleted from v1.

## What changed and why

The brief: the product is not one camera on one roof. It reads drone feeds and existing CCTV
networks; it runs on local compute, a virtual machine, cloud or a mobile unit; several drones can
cover an area and **share their detections and reasoning**; and the surfaces in scope include
walls, perimeters and building faces.

### Unchanged (deliberately)

- **Peak.** The hero plate — one real masked frame with the latency as an instrument readout. The
  page's argument is still *this one frame, this fast*. Broadening the story does not get a new peak.
- **Tokens.** `#eef1f3` paper, `#101820` ink, `#5b6570` graphite, `#c8361a` signal. Red is still
  reserved for detection and alert only — it now appears on the four detections in the elevation
  plate and the marker in the fleet plan, and nowhere else.
- **Type.** Fraunces display / IBM Plex Sans text / IBM Plex Mono for measured data only. No new faces.
- **Motion budget: still two families.** (1) the plate reveal via IntersectionObserver; (2) the single
  `pulse`. No third family was added — the new plates express motion through the *existing* pulse on
  their detection markers. `prefers-reduced-motion` still removes both.
- **Grid break.** The hero still bleeds off the right edge of the measure.

### Added

1. **Plate A reworked (the diagram that was "just a camera").** Three input classes — existing CCTV,
   cameras you already own, drone feeds — fan into **one reasoning core** (detect → mask → fuse),
   which fans out to field phones and an operator console. This is the direct answer to the brief:
   the camera is now one of three sources, and the core is the subject.
2. **New section — Inputs.** Prose on reading an existing NVR/VMS, any surface-mounted camera, and
   drone video, with the point that mixing sources is the feature.
3. **New section — Drone fleets.** Plan-view SVG: three aircraft with overlapping coverage, dashed
   mesh links labelled *shared detections + reasoning*, a detection on the roof and a handover arc
   labelled *no gap between fields*.
4. **New section — Where it runs.** A four-panel technical plate (on the site / virtual machine /
   cloud instance / mobile unit), each drawn distinctly, plus the honest note that on-site and mobile
   keep frames inside the perimeter by construction.
5. **New section — Not only roofs.** An elevation plate with four detections: on the roof, on the
   wall, on the face, inside the line after hours. Followed by the anomaly framing (a door open that
   is always shut, a lifted hatch) and the explicit boundary: surfaces and objects, not identifying people.
6. **New block — the ledger.** A two-column block that separates *measured on this machine* from
   *supported in the architecture*. This is the credibility guard: the broadened story must not read
   as claimed performance. The measured table itself is unchanged.
7. **Readout band** extended from four lines to six (inputs / surfaces / where it runs added).
8. **Deploy list** extended with inputs, compute (four options) and aircraft (single vs fleet).

## Defects found and fixed during validation

- **Deployment plate: the cloud panel was drawn in the wrong column.** Its glyph and caps landed in
  panel 2's space (measured at 26.9–74.6% of the plate width, spanning two panels). Redrawn inside
  panel 3 (now 52.3–71.7%). Caught by measuring rendered group bounds against the plate box, not by eye.
- **Plate A: two mono labels collided** on the same baseline — `operator console` and a right-aligned
  `zone + confidence + log` overlapped 45 × 17 px. The right-aligned label was moved below the panel.
- **Fleet plate: the building caption sat on the building fill and spilled past its edge.** Moved
  below the building and shortened.
- **Deployment panel 4: a mono line ran to 99.4% of the plate width** (edge-of-frame). Rewritten shorter.
- Plate B now carries its own arrow marker (`#arB`) instead of reaching into Plate A's `#ar`.

## Verification (what actually ran)

| Gate | Result |
|---|---|
| `slopscan.mjs site/index-v2.html` | **0 fails, 0 warns, 0 suppressed** |
| Horizontal overflow @1440 / 768 / 390 | **0 px** at all three |
| Text-block collisions @1440 | **0** across 107 text blocks |
| SVG text-vs-text collisions, all 4 plates | **0** |
| SVG groups outside their plate box | **0** |
| Image loads | 3/3, natural size 1024², 1024², 1080×1920 |
| Console / page errors | none (only the browser's automatic `favicon.ico` 404, same as v1) |
| WCAG AA contrast, all text ≥4.5:1 (≥3:1 large) | **0 failures** |
| `prefers-reduced-motion: reduce` | reveal opacity 1, transform none, pulse animation none |
| JS disabled | content present in markup; reveal rule is inside the `no-preference` query, so the default is visible |

**Honest gap.** `vision_analyze` was returning HTTP 404 for every image this session, including a
known-good asset — an external backend outage, not a problem with the screenshots. So the visual read
that auteur's verify gate asks for ("look at every frame") **did not happen**. Screenshots were
produced at all three widths (`shots/final-1440.png`, `final-768.png`, `final-390.png`) and the
layout was validated numerically instead: bounds, collisions, overflow, contrast. Numeric checks
catch misplaced and overlapping geometry — they caught four real defects above — but they cannot
judge whether the result is *beautiful*. That judgement still needs a human or a working vision pass.

## Files

- `index.html` — v1, untouched.
- `index-v2.html` — this edit.
- `../_backups/index-v1-20260917-144828.html` — byte-identical v1 backup.
- `../shots/final-{1440,768,390}.png` — full-page renders of v2.
- `../shots/s00.png … s07.png` — v2 sliced at 1350 px for review.
