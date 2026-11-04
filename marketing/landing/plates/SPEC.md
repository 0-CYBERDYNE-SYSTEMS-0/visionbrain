# SPEC — VisionBridge plate system v3

**Status: AUTHORITATIVE.** This is the single source of truth for all v3 plate work.
If your task brief and this spec disagree, **this spec wins** — say so in your report.

Owner: the orchestrating agent. Baseline page: `site/index-v2.html` (v2, live).
Target: v3 fragments that get integrated into a new `site/index-v3.html`. **You do not
touch the live page or the v2 file. You produce fragments only.**

---

## 1. What the page is, and the design contract it must obey

VisionBridge — a landing page for a roof/wall/perimeter intrusion-watch system that reads
existing CCTV and drone feeds and pushes a masked frame to field phones. The page's whole
credibility argument is *measured, not claimed*: it looks like an instrument, not a hacker tool.

Committed contract (from `site/DESIGN-NOTES.md` — do not renegotiate it):

| Token | Value | Role |
|---|---|---|
| paper | `#eef1f3` | page ground (the plate ground is transparent → paper) |
| panel | `#e4e9ec` | inner panel fill |
| ink | `#101820` | primary text, primary strokes, structure |
| graphite | `#5b6570` | secondary text, connectors, de-emphasised ink |
| rule | `#c8d0d6` | hairlines, panel outlines, separators |
| **signal** | `#c8361a` | **detection / alert marks ONLY** |
| signal-ink | `#9c2a13` | text labels attached to a detection |

**The single most important rule: red is semantic.** `#c8361a` may appear *only* on a
detection marker (a person found), its bounding/mask box, and its label. Never on a section
number, an arrow, a panel border, a hover state, or anything decorative. One plate's red
count is a number you must report (§6).

Type (only these three faces exist on the page): **Fraunces** display, **IBM Plex Sans** body,
**IBM Plex Mono for measured data only**. Inside SVG you have exactly two classes:

- `.svg-label` → 15px, `#101820` (Sans)
- `.svg-mono` → 12.5px, `#5b6570` (Mono) — for data, measurement, and technical annotation

Motion budget is **two families, total, for the whole page**, and you may not add a third:
(1) the `.reveal` opacity/translate on entry, (2) the single `.pulse` class. Your plates may
reuse `.pulse` on detection markers only.

---

## 2. Plate geometry

- All four plates are **1120 user units wide** (the page measure is 1180px at a 1440px viewport,
  so a plate renders at **1.0536×** — figures are reported back in user units so they compare).
- Current viewBoxes: `0 0 1120 350` (P1), `0 0 1120 400` (P2), `0 0 1120 250` (P3), `0 0 1120 360` (P4).
  You may change your plate's **height** if the improvement genuinely needs it; report old → new
  and why. Do not change the width.
- **Baseline grid: 4 user units.** Every panel edge, glyph centre, connector attach point and
  text baseline snaps to it. Panel gutters: **≥24 uu**, prefer 40.
- Minimum ink margin to the plate edge: **28 uu** (no element may touch or cross the frame).
  Coverage circles and any other "bleeding" element must still be fully inside the viewBox —
  clipping is a defect, not a style.
- Stroke widths: **1, 1.5, 2, 2.5 only.** Existing convention: panel/frame outline `1.5`,
  structure and hairlines `1`, data/emphasis `2`, detection box `2`. Connectors `1.5`.
- Corner radius: `rx=0` for all structure. `rx ≤ 9` allowed only on device glyphs (phone, tablet).

## 3. Label rules

- Mono labels are **lowercase** and use `·` as an internal separator. Existing style:
  `NVR or VMS · we read the stream`, `handover — no gap between fields`.
- **Do not mix sentence style and fragment style inside one plate.** v2 mixes them in P3
  (`Rides hardware and backups / you already operate.` = sentence, next to `a region you choose`
  = fragment) — that is a defect to fix. Pick one style per plate: either every caption is a
  fragment, or every caption is a full sentence that keeps its terminal period.
- Max label length **46 characters**, except a plate-level footnote line which may run to 70.
- Every label must be unambiguously attached: inside its own panel, or joined by a leader line.
  Minimum **8 uu** clearance between a leader line and any other ink.
- No label may sit on top of a fill it did not create. `#e4e9ec` panel fills and `#101820` glyph
  fills are the two cases that broke v2 — keep text clear of both.
- Text must never be smaller than 12.5px, and never a third size unless you report it.
  Max **2 additional font sizes** across your plate, declared as presentation attributes.

## 4. Hard prohibitions

1. **No new CSS classes.** Use only `.svg-label`, `.svg-mono`, `.pulse`, `.reveal`. Everything else
   goes in presentation attributes (`fill`, `stroke`, `font-weight`, `text-anchor`, `font-size`).
2. **No `<style>` element inside your fragment.** Style lives on the page, not in the plate.
3. **No external references.** No `<image>`, no `href`/`xlink:href` to any URL, no external fonts,
   no icon sprites, no `<use href="…">` to another file.
4. **No gradients, filters, drop-shadows, blur, masks, or clip-paths.** Flat ink and stroke only.
   (Model *masks* as filled paths if you need a mask shape — that is different from an SVG `<mask>`.)
5. **Namespaced ids.** All four plates are inlined into ONE page, so ids share one global
   namespace. Every id you create must be prefixed with your plate, e.g. `p1-ar`, `p3-panel2`.
   Existing v2 plates use bare `ar` / `arB` — that is exactly the collision-shaped bug to avoid.
   Never reference an id you did not define.
6. **No capability invention.** See §5.
7. **Do not write, move, or delete anything outside the file paths assigned to you in your brief.**
   Never touch `site/` in the demo repo, `baseline/`, or another agent's fragment.

## 5. Truth constraint (this is a security product — do not draw a lie)

Everything you draw must correspond to a capability the page already states. The permitted
inventory is exactly:

- inputs: existing **CCTV / NVR / VMS** streams; **fixed cameras you already own** (any surface);
  **drone feeds**, single aircraft or a fleet
- core: **detect** (is a person on a surface?) → **mask** (pixel-accurate shape, not a box) → **fuse**
  (multiple views reconcile into one picture)
- outputs: **field phones** (annotated frame, under 2 s after the mask) and an **operator console**
  (zone, confidence, timestamp, log)
- drone fleet: overlapping coverage, **shared detections and reasoning between aircraft**, handover
  with no gap between fields
- deployment modes: **on the site** (nothing leaves the building) · **virtual machine** · **cloud
  instance in a region you choose** · **mobile unit in a vehicle**
- surfaces: **roof**, **wall / perimeter**, **building face**, and after-hours inside-the-line
- anomalies that are architectural, not a benchmark: a door standing open that is always shut, a
  service hatch lifted, an object left on a surface that is always clear
- explicit non-claims, which you may **not** draw or imply: face recognition, person identification,
  weapons, a replacement for guards, radios, or a security plan, and **no measured performance
  number that is not in `out/falcon_results.json` / the page's table**

The page's honesty device is a two-column block separating what was **measured on this machine**
from what is **supported in the architecture**. If your plate implies measured performance, that
is a lie. Keep architecture claims architectural.

## 6. The professional bar — what "finalised and perfected" concretely means

1. **Node alignment.** Panel tops, panel floors and glyph centres line up across the plate. Not
   "close" — aligned. A reader should feel the grid even if they cannot see it.
2. **Optical balance.** No dead zone wider than ~20% of the plate width without a reason. Ink
   margin consistent on all four sides.
3. **Three readable levels.** A reader at 100% who reads only the *titles* must still understand
   the flow. Titles are `.svg-label` at weight 500; data annotations are `.svg-mono`. If a reader
   needs every mono label to follow the diagram, the hierarchy has failed.
4. **No orphans, no tangents.** Every label either sits inside its own panel with clear padding, or
   is joined by a leader that is unambiguous and clears all other ink by 8 uu.
5. **Connectors are uniform.** One curvature family, entering and leaving at right angles,
   arrowheads (`marker-end`) only where direction carries meaning. Every plate defines its own
   marker with a prefixed id.
6. **Density discipline.** No ink that adds no meaning. Delete decorative ticks, redundant rules,
   and any element whose removal loses nothing.
7. **Cross-plate consistency.** Someone reading all four plates should see one hand: same stroke
   scale, same label offset from its glyph, same panel padding, same leader style.
8. **It must survive inspection at 100% zoom and at 390px.** At 390px the plate is ~342px wide
   (0.305×) — plates are allowed to become dense there, but must not produce overflow or
   illegible collision.

## 7. Evidence you must produce (non-negotiable)

Run this and paste the real output. It is the same gate for everyone:

```bash
cd /Users/scrimwiggins/visionbridge-plates/harness
node measure.mjs /Users/scrimwiggins/visionbridge-plates/v3/<your-plate>.svg --width 1440 --png /Users/scrimwiggins/visionbridge-plates/shots/<your-plate>-v3-1440.png
```

It exits **1** on any hard failure and prints JSON. **Your plate must report `PASS: true`** with
all eight `HARD` counters at `0`: `textCollisions`, `outOfFrame`, `contrastFails`,
`missingRoleOrAria`, `externalRefs`, `classesNotInPage`, `dupIds`, `missingMarkers`.

Also run it at `--width 390` and `--width 768` and report those two results too (they must not
show `externalRefs`/`classesNotInPage`/`missingMarkers`, and must not show >0 `textCollisions`).

`classesNotInPage` counts any class that is not one of the four permitted — if it is non-zero you
used a class you were told not to.

**Baseline for comparison (measured, v2, at 1440):**

| plate | PASS | texts | collisions | outOfFrame | contrastMin |
|---|---|---|---|---|---|
| plate1-pipeline | true | 20 | 0 | 0 | 4.85 |
| plate2-fleet | **false** | 7 | **1** | **2** | 4.85 |
| plate3-modes | true | 15 | 0 | 0 | 4.85 |
| plate4-elevation | true | 9 | 0 | 0 | 4.85 |

So: P1, P3 and P4 are *numerically clean but not yet professionally resolved* — your job there is
craft and hierarchy, not bug-fixing. **P2 has two real defects** and is the only plate that
currently fails.

## 8. What you must put in your final answer

1. `what changed, and why` — as decisions, not a changelog. Include the reasoning behind the
   single hardest call you made.
2. The **verbatim gate JSON summary** at 1440 (all eight HARD counters) plus the 390 and 768
   results.
3. Plate facts: old viewBox → new viewBox; stroke widths used; font sizes used; **count of
   `#c8361a` occurrences and what each one marks**; new id list.
4. Any place the plate is still not good enough, and what it would need. **A confident "done" that
   hides a known weakness is a failed submission.** If you could not verify something, say which
   thing and why.
5. The absolute path of every file you created.

You may render and look at your own output (Playwright is installed and the gate writes a PNG at
`--png`). You have no working image-analysis backend on this machine — do not claim you
"visually inspected" a screenshot you could not actually see. Verify numerically and say so.
