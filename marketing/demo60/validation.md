# validation — script_final.md (60-second VisionBrain client demo)

**Validator:** adversarial pre-render pass, 2026-09-09
**Inputs:** BRIEF.md, script_final.md, EDITOR_NOTES.md, marketing/copy.md, plus on-disk asset inspection (ffprobe durations, frame extraction of every contested source clip).

## OVERALL VERDICT: PASS WITH FIXES

Timing, honesty, voice, TTS, and pipeline story all pass. One **BLOCKER** in the asset check: scene 1's two "annotated windows" in `rotterdam_analyzed.mp4` are the **inverse of the source's real overlay structure** — as written, ~7.7 of the hook's 9.0 seconds would render as clean, unannotated drone footage while the VO says "Three hundred thirty-eight objects tracked." Fixable by re-windowing takes only; VO, timing, and all other claims are sound.

---

## A. TIMING MATH — PASS

Recounted independently; the script's totals table is correct.

1. **NOTE — durations and TCs verified.** 9.0 + 10.0 + 11.0 + 12.0 + 11.0 + 7.0 = 60.0 exactly. TC ranges are contiguous (00:00.0 → 00:09.0 → 00:19.0 → 00:30.0 → 00:42.0 → 00:53.0 → 01:00.0) and each matches its stated duration; scene 6's `01:00.0` is consistent mm:ss.d notation for 60.0 s.
2. **NOTE — per-scene word counts verified** (whitespace tokens of the exact VO lines): S1 = 12, S2 = 17, S3 = 22, S4 = 21, S5 = 22, S6 = 12. Against duration × 2.2 limits: 12 ≤ 19.8, 17 ≤ 22.0, 22 ≤ 24.2, 21 ≤ 26.4, 22 ≤ 24.2, 12 ≤ 15.4 — all pass with breathing room. The per-scene w/s column (1.33 / 1.70 / 2.00 / 1.75 / 2.00 / 1.71) recomputes correctly.
3. **NOTE — total VO verified.** 106 words ≤ 132 cap. Overall pace 106 / 60 = 1.77 w/s ≤ 2.4. Header claim "(1.77 w/s)" is correct.
4. **NOTE — silent time verified.** Every scene carries VO; scene-internal visual timings (S1: 4.5+4.5; S3: 7.5+2.0+1.5; S4: 8.0+4.0; S5: 6.0+5.0; S6: 3.0+2.0+2.0) each sum exactly to the scene duration. Total silent time 0 s ≤ 8 s.

No timing findings beyond notes. No blockers.

---

## B. HONESTY — PASS (2 minors, 1 note)

Checked every VO line and on-screen string against all seven hard constraints, plus a number-by-number trace.

1. **MINOR — "unique" dropped from the approved 338 claim.** Scene 1 on-screen `480 frames · 96 processed · 338 objects tracked` and scene 1 VO `"Three hundred thirty-eight objects tracked."` both omit "unique" from the brief's approved number ("338 unique objects tracked"). The claim stays true and does not over-claim, but the unqualified form invites a "338 objects in frame at once" misreading. Replacement (on-screen): `480 frames · 96 processed · 338 unique objects tracked`. Replacement (VO): `"Three hundred thirty-eight unique objects tracked. One query. One mac, on a desk."` (13 words, still ≤ 19.8; new total 107 ≤ 132).
2. **MINOR — optional compass overlay in scene 4 flirts with constraint 3.** Scene 4 visuals: "optional small compass overlay build if wanted, otherwise none." A compass-rose graphic reads as true-north/georeferenced bearing; direction is 8-way in image coordinates only. Frame inspection confirms the source (`demo/video/live_harbor_tracking.mp4`) carries text heading labels only (`boat 25% · south`, `boat 62% · southeast`) and no rose. Replacement: delete the option — "…heading label ticks over as it moves; no compass graphic exists in the source and none is built."
3. **NOTE — two sequential live clips under a "one live feed" banner.** Scene 4 cuts `live_masks_container_truck.mp4` → `live_harbor_tracking.mp4` (separate sessions) while VO and on-screen say "One live feed." The constraint bans multi-camera/multi-site *dashboards*; sequential cuts are standard editing and the claim (one live source per instance) remains true, and the traceability block already discloses it. Defensible as-is. Optional hardening: keep the `one live feed · …` body line pinned only over the first clip.
4. **NOTE — all numbers traced.** 480 / 96 / 338 (brief approved list); 32-gigabyte mac (approved list); 35 clips in one session (constraint 7, documented); `● hub ready · 5 fps relay` and 5 fps relay (copy deck: "5 fps observer relay" bullet + status chip set 4); `4 tracked objects in frame. no perimeter events in the last 60 s.` (copy deck bubble set 1, numbers verbatim — editor note 8 is accurate); "20-second clip" (approved CTA). No unapproved numbers found.
5. **NOTE — constraint sweep clean.** (1) No multi-camera/multi-site wording; "watch every site" absent. (2) Live detection never attributed to a named model; scene 2's model verbs play over `analyze_running_1920.png` (analyze pipeline). (3) Only "direction"/"heading"/"8-way"; no "georeferenced"/"gps". (4) "dwell" listed as a feature, no time-in-named-zone claim. (5) No accuracy figures or percentages in any claim (the 40%/30% figures are dimming levels in production notes, not on-screen). (6) Privacy in the exact approved shape: `no cloud. no subscription. your footage never leaves the building.` is the copy-deck positioning line verbatim; no certification/compliance claims; scene 1's "One mac, on a desk" is locality, not privacy. (7) Zones, direction triggers, evidence clips all real and correctly scoped.

No honesty blockers.

---

## C. VOICE — PASS

1. **NOTE — no market-speak, no hype.** No banned adjectives ("revolutionary", "powerful", "cutting-edge") or their cousins (no "seamless", "instantly", "unlock", "supercharge"). Lines are ones a senior field engineer would say: "before the crew lands", "A fence that writes its own log" (both brief-supplied), "Back comes an answer" — dry, plain, number-forward.
2. **NOTE — no emoji.** `●` in the scene 1/4 chips is the copy deck's own status-chip glyph (bubble set 4), not emoji. Fine.
3. **NOTE — casing rules verified string by string.** Labels small caps (`ONE QUERY · PORT OF ROTTERDAM`, `TRACKING · SAM 3.1`, `PIPELINE 01 · RECORDED FOOTAGE`, `PIPELINE 02 · LIVE SITE`, `PIPELINE 03 · ASK THE SCENE`, `● hub ready · 5 fps relay`, `VISIONBRAIN · GROUND CONTROL`); body lowercase everywhere, including the bubbles (`operator — what do you see?`, `visionbrain — 4 tracked objects in frame. …`) and the CTA (`bring a 20-second clip. leave with a report.`); the one heading, `Three models, one pipeline`, is sentence case. No violations.

No voice findings. No blockers.

---

## D. ASSETS — FAIL (1 blocker)

**Paths:** all 11 referenced assets exist on disk under `/Users/scrimwiggins/VisionBrain/marketing/` — `demo/video/rotterdam_analyzed.mp4`, `demo/video/mission_marina_boats_everyframe.mp4`, `demo/video/live_masks_container_truck.mp4`, `demo/video/live_harbor_tracking.mp4`, `demo/video/mission_containers_truck_everyframe.mp4`, `shots/analyze_running_1920.png`, `shots/analyze_results_1920.png`, `shots/analyze_report_1920.png` (scenes 3 and 5), `shots/live_terminal_1920.png`, `shots/live_quay_1920.png`, `shots/annotated_frame.jpg`.

**Durations (ffprobe, actual):** rotterdam 20.02 s · marina 10.36 s · containers 12.93 s · live truck 81.12 s · harbor 20.00 s — all match the brief, and every stated take is feasible (S1: 4.5+4.5 from 20.02; S3: 7.5 from 10.36; S4: 8.0 from 81.12, 4.0 from 20.0; S6: 3.0 from 12.93).

1. **BLOCKER — scene 1's annotated windows are inverted; the hook would render ~7.7 s of clean footage.** Offending string (scene 1 visuals): "two annotated windows only, source 00:00–00:04.5 then source 00:15.5–00:20.0 (the source's middle is clean footage, no overlays)". Frame-by-frame scan of `demo/video/rotterdam_analyzed.mp4` (0.2 s steps, visually confirmed at full resolution) shows the **opposite** structure — the source is mostly clean footage with ~1 s annotation flashes:
   - `00:00.0–00:00.8` annotated → `00:01.0–00:04.8` CLEAN
   - `00:04.9–00:05.9` annotated → `00:06.0–00:09.6` CLEAN
   - `00:09.7–00:10.7` annotated → `00:10.8–00:14.6` CLEAN
   - `00:14.7–00:15.7` annotated → `00:15.8–00:19.4` CLEAN
   - `00:19.5–20.02` annotated

   The planned window 00:00–04.5 is annotated only for its first ~0.8 s, and 15.5–20.0 only for its final ~0.5 s — i.e. ~7.7 s of the 9.0 s hook plays with **no overlays** while the VO claims "Three hundred thirty-eight objects tracked" and the numbers line types on. EDITOR_NOTES.md item 2 ("carries overlays only in its first ~3.3 s and last ~3.5 s") is also wrong and is the source of this error; do not trust it. **Replacement (scene 1 visuals):** "cut the five measured annotated flashes — source 00:00.0–00:00.8, 00:04.9–00:05.9, 00:09.7–00:10.7, 00:14.7–00:15.7, 00:19.5–00:20.0 (~4.3 s total; all intervening source footage is clean) — on the VO beats with push-in carried across cuts, and cover the remaining scene time with `demo/video/mission_marina_boats_everyframe.mp4` (verified annotated on every frame, e.g. source 00:00–00:04.7); cut hidden on the type-on beat." If the marina covers scene 1's gap, shift scene 3's marina take to the back half (source 00:04.7–00:10.2, 5.5 s) and rebalance its stills (results 3.0 s, report 2.5 s) so the same boats are not shown twice.
2. **NOTE — scene 4 source claims verified true.** `live_harbor_tracking.mp4`: hand-drawn target box holding the subject, text heading labels (`boat 25% · south`, `boat 62% · southeast`), no compass graphic — editor note 3 is correct, and the brief's "compass heading" asset description refers to those labels. `live_masks_container_truck.mp4`: painted masks with persistent labels (`truck 80% · west`) and a marked truck lane — "across the lane" is accurate. `mission_marina_boats_everyframe.mp4` (t=1, t=9) and `mission_containers_truck_everyframe.mp4` (t=2) verified annotated.
3. **NOTE — scene-internal take arithmetic verified.** Every scene's stated take lengths sum exactly to its duration (see A.4).

One blocker (finding D.1).

---

## E. TTS SAFETY — PASS (1 minor, 1 note)

1. **MINOR — digits left in VO contradict the script's own TTS claim; pin a TTS-specific transcript.** Scene 2 VO contains `Sam 3.1` and scene 6 VO contains `20-second`, while the totals block claims "spoken numbers are written out for TTS" — they are not; they are digits whose rendering depends on the voice. macOS default reading ("three point one", "twenty second") is acceptable, but for a deterministic render, generate the TTS transcript with the spelled forms: `"Three models, one pipeline. Sam three point one tracks. Falcon grounds. Gemma writes. All three on one thirty-two gigabyte mac."` and `"… Bring a twenty-second clip. Leave with a report."` Keep the script VO as-is for readability if the render pipeline takes a separate transcript; otherwise spell them out in place. Either way, correct the totals-block sentence to match what is actually done.
2. **NOTE — symbols: clean.** Verified directly: no `→`, `·`, `%`, or `&` in any VO line (they appear only on-screen, where they belong). The colon in "One live feed:" renders as a pause; fine.
3. **NOTE — homographs: resolved or context-safe.** `live` in "One live feed" is disambiguated by "feed" (reads /laɪv/); the riskier "live on one mac" was already removed (editor note 6 — correct call). `grounds` is /ɡraʊndz/ as noun and verb alike; `mac` reads "mack"; `masks`, `counts`, `tracked` unambiguous.

No TTS blockers.

---

## F. PIPELINE STORY — PASS

1. **NOTE — the three monetizable pipelines are explicit and distinguishable.** Scene 3 = RECORDED-FOOTAGE REVIEW (query at recorded video → tracks, counts, report "before the crew lands" — brief wording verbatim), scene 4 = LIVE SITE MONITORING (one live feed, zone counters, 8-way direction, dwell, evidence clips, "a fence that writes its own log" — brief metaphor verbatim), scene 5 = OBSERVATION → BUSINESS ACTION (ask the scene → answer, field report, evidence → work order). Each is numbered on-screen (`PIPELINE 01/02/03`) and named first in its VO ("Pipeline one/two/three"), so the three stories cannot blur.
2. **NOTE — repeat-back test passes.** After 60 s a business viewer can state: what it does (one query → tracked objects, counts, written report — scenes 1–2), the three things you can hire it for (scenes 3–5), who it's for (ports & terminals, logistics yards, corridors, shipyards — scene 6), and why it's different (runs on one mac on a desk; footage never leaves the building; no cloud, no subscription — scenes 1 and 6). No code, no developer framing.

No findings. No blockers.

---

## required fixes

1. **[BLOCKER — scene 1 visuals, check D]** The claimed annotated windows in `demo/video/rotterdam_analyzed.mp4` are inverted. Measured annotated segments are 00:00.0–00:00.8, 00:04.9–00:05.9, 00:09.7–00:10.7, 00:14.7–00:15.7, 00:19.5–20.02; everything between is clean. Re-plan scene 1 as the five flash cuts (~4.3 s) plus `demo/video/mission_marina_boats_everyframe.mp4` (annotated every frame) for the remainder — and if the marina moves into scene 1, shift scene 3's marina window to source 00:04.7–00:10.2 (5.5 s) with results 3.0 s / report 2.5 s. Do not render from the current 00:00–04.5 + 15.5–20.0 windows.

Recommended (non-blocking) fixes, in priority order: restore "unique" to the 338 claim on-screen and in VO (B.1); delete the optional compass-rose build in scene 4 (B.2); spell out "Sam three point one" / "twenty-second" in the TTS transcript and fix the totals-block sentence that claims numbers are already written out (E.1).
