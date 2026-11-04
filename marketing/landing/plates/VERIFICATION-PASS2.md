# VERIFICATION — pass 2 (fixes after the first validation)

**Date:** 18 Sep 2026 · follows `VERIFICATION.md` (pass 1). Same artifacts:
`~/.hermes/workspace/visionbridge-demo/site/index-v3.html` · plates in `~/visionbridge-plates/v3/`

---

## 0. Correcting the premise of this pass

This pass was requested as "fix the issues you found with the team when you reviewed them
visually". **No visual review had taken place.** `vision_analyze` returns HTTP 404 on this machine
and did so again at the start of this pass; oMLX (:8181) and Ollama (:11434) are down. The only
vision models present are LM Studio's `lfm2.5-vl-3b` and `lfm2.5-vl-450m`.

So the work below was driven by the specialists' *measured* self-reports, plus measurement. That
turned out to matter: **the one issue the specialist declared for plate 2 was itself a
misdiagnosis, and the real defect was worse.**

## 1. The bounded visual attempt (and why it is only corroboration)

The 3B VL model was run against element-clipped 2x renders (2240px wide) of all four plates, with
**control questions whose answers were already known**:

| control | result |
|---|---|
| "is any red/orange present?" — known YES for plates 1/2/4, NO for plate 3 | **4/4 correct**, including correctly reporting NO red on the deployment plate |
| "quote three words of text you can read" | read real strings verbatim from every plate (`one reasoning core`, `504 m`, `on the wall`) |

So the input resolution was adequate and it is genuinely perceiving. **However it contradicted
itself on size comparisons in 3 of 4 plates** ("all the same size … one is noticeably smaller"),
so its comparisons were rejected as evidence. Its one coherent, useful answer was on plate 3:
*"a house with a square inside, a rectangle with a square inside, a cloud shape with a square
inside, a vehicle (truck or van) with a square inside … all four appear to be of the same size."*

That is the only thing this pass takes from the model, and only because it agrees with an
independent geometric measurement.

## 2. The real defect found in plate 2 and fixed

**What the specialist declared:** "the lens fill is `#e4e9ec`, the same value as the building, so
where a lens crosses the roof the overlap reads only as a thin outline instead of a covered area."

**What is actually true — measured, not reasoned:** the overlap lenses **enclosed zero area**.
Both arcs of each "lens" carried sweep flags that select the *same* circle centre, so the two arcs
coincided, the path enclosed 0 u², and the lens rendered as a bare arc line. The fill colour was
never reached, so the declared colour problem could not have been the cause.

Measured filled area per arc-flag combination (`arctest.mjs`), against the analytic intersection
area for two circles r=150 at centre distances 276.18 and 280.18:

| lens | correct area | current flags | fixed flags |
|---|---|---|---|
| L1 | 1876 u² | **0 u² (degenerate)** | 1822 u² ✅ |
| L2 | 1427 u² | **0 u² (degenerate)** | 1388 u² ✅ |

Fix: the second arc's sweep flag `1 → 0` in both lens paths, plus the fill changed to the rule
tone `#c8d0d6` (the only mid-tone in the token set; an overlap region must read as a covered AREA
over both the paper ground and the roof plane, not as the building's own value).

**A second, latent defect fixed with it:** the person-route polyline was drawn *before* the lenses.
With a zero-area lens that was harmless; the moment the lens encloses real area, an opaque fill
painted over the route would cut the track exactly where it crosses an overlap — which is the one
thing that polyline exists to prove. The route is now drawn after the lenses.

### Verification of the fix, from pixels (no model involved)

| measurement | before | after |
|---|---|---|
| flat lens fill pixels in the render | 0 | **10 881 px = 2 720 u²** |
| lens 1 rendered extent | bare arc line | **x[395.0…416.5] y[145.0…256.5]** (w 21.5, h 111.5) |
| pixel inside lens 1 | `#e4e9ec` — identical to the building | `#c8d0d6` — **distinct** |
| building pixel, clear of lens | `#e4e9ec` | `#e4e9ec` (unchanged) |
| paper pixel, clear of lens | `#eef1f3` | `#eef1f3` (unchanged) |
| route ink inside lens 1 | 76 px | **121 px — drawn over the lens** |

## 3. Plate legibility at phone widths — fixed

Declared by the plate-3 specialist and confirmed: at a 390px viewport a 1120-unit plate is drawn
at ~0.305×, so 12.5px IBM Plex Mono renders at **3.8px** — present, uncollided,
contrast-compliant, and not readable.

Fix (`v3/diagrams.css`): below 760px the drawing keeps a legible size (700px) and `.diagram`
scrolls. This mirrors the page's own existing solution for its data table
(`.measure { overflow-x: auto }` in the ≤900px query).

| width | plate render | 12.5px mono renders at |
|---|---|---|
| 1440 | 1180×497 px, scale 1.054 | 13.2 px |
| 768 | 720×303 px, scale 0.643 | 8.0 px |
| **390** | **700×295 px, scale 0.625** | **7.8 px** (was 3.8 px) |

Page-level horizontal overflow remains **0 px at all three widths** — the scroll is contained by
`.diagram`.

## 4. Re-validation after the fixes

| check | result |
|---|---|
| All four plates, 1440 / 768 / 390 | **PASS** — all eight hard counters 0 at every width |
| Whole assembled page, 1440 / 768 / 390 | **0 overflow, 0 text collisions (212 boxes), 3/3 images, 4/4 `role=img` with aria-label, 0 duplicate DOM ids** |
| `slopscan` on `index-v3.html` | **0 fails, 0 warns, 0 suppressed** |
| Live v1 / v2 | untouched — still dated Sep 17 |

## 5. Issues now closed, and issues accepted with a reason

**Closed in this pass:** degenerate overlap lenses (0 u² → real area); lens colour
indistinguishable from the building; person route occluded inside the overlap; phone-width
illegibility (3.8px → 7.8px).

**Accepted, with the reason — these are not defects:**

- **Plate 3's cloud carries ~17% less stroked ink than its three siblings.** Bounding-box parity is
  *exact* (96×80, area 7680, shared baseline y=132). Perimeter is a biased metric when comparing a
  bumpy curved outline to a rectangle, and the independent VL read agreed all four appear the same
  size. Chasing perimeter parity would cost cell parity, which is the more valuable symmetry.
- **Plate 3 has three arc junctions off the 4-unit grid** (`cx±25.90, 80.29`). These are the
  two-circle intersection points that make the arcs tangent — *derived* values, not design values.
  Every snap target the spec names (panel edges, glyph centres, connector attach points, text
  baselines) is on-grid.
- **Plate 4's lower-right is its densest quadrant.** D2 (on the wall) and D4 (inside the line) sit
  in the right third because the wall and the gate are there. Architectural, not compositional.
- **The favicon 404 at first load** is the browser's automatic request, present in v2 as well.

**Still open — the one item not resolved:** plate 1 carries an in-SVG title strip
(`how it works · three inputs, one core, one frame out`) that its specialist added on its own
initiative; plates 2, 3 and 4 have none. Removing it properly requires re-laying out ~50 y
coordinates (its strip occupies exactly the space a correct 28-unit top margin needs, so the
viewBox cannot simply be cropped). Held rather than guessed at, twice.

## 6. Artifact hashes after this pass

```
index-v3.html        cef3ee6b8a9704eb26881b31beef215e
plate1-pipeline.svg  58d0b0ef0bce5c17dc214a7e3856c662
plate2-fleet.svg     3bcc71401574d3822ab3109162a3db72   (changed: lenses + route order)
plate3-modes.svg     0af1931e3df807c2304974209513e4f4
plate4-elevation.svg 6e35c3cbacb4f1c096a03376c1a99118
diagrams.css         0aeb8af62f2781ae35f08d8457d8757b
```

## 7. Honest limits of this pass

- **No human has looked at any of it, still.** Pixel measurement proves the lens now encloses a
  distinguishable area and the route survives over it; it cannot judge whether the drawings are
  *beautiful*, or whether a 3-tone flat lens reads better than a hatch would. That needs an eye.
- The VL model is a bounded instrument: reliable on colour presence and on reading text, unreliable
  on comparisons. It was used only to corroborate one measurement, and never as sole evidence.
- `pixsample.py` (pass-2 first attempt) sampled points derived from arc endpoints and reported "no
  change" for a real change. The lesson — measure the rendered artifact, never infer from path
  coordinates — had already been learned once with the cloud in pass 1 and was repeated here. The
  scripts that replaced it (`findlens.py`, `arctest.mjs`, `verify_lens.py`) all read the render.
