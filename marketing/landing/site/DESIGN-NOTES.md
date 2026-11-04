# DESIGN — VisionBridge landing (commit-sheet, per auteur §commit-sheet)

**Peak (signature element):** one annotated frame, presented like a technical plate — the masked
person on the roof at hero scale, with the measured latency printed underneath as an instrument
readout rather than a stat row. The page's argument is *this one frame, this fast*.

**Color:** cool paper `#eef1f3` ground, ink `#101820`, graphite `#5b6570` secondary, one committed
signal `#c8361a` used **only** where a detection or alert happens — never as decoration.
Why not the category reflex: every security product on the market is dark with neon green or amber.
Deliberately light, instrument-grade (Swiss technical atlas), so it reads as a measuring device, not
a hacker tool. Not lavender, not cream (banned warmth reflex), background mean lightness ~0.93.
Deliberate: an alert is the *only* red thing on the page, so red keeps its meaning.

**Type:** Fraunces (editorial serif, optical sizing) for display; IBM Plex Sans for text; IBM Plex
Mono **only** for measured data. Why not Inter/Space Grotesk: they are the 2024–26 AI default, and
nobody in this category sets headlines in a serif — it reads as published, not generated.

**Grid break:** the hero image bleeds off the right edge of the measure; the diagram runs full-bleed
as a horizontal plate; the readout band breaks the uniform section rhythm with a mono wire.

**Motion budget (2 families):** (1) one reveal for the hero plate + wide plate (opacity/translateY
via IntersectionObserver — no scroll listeners); (2) a single slow pulse on the tripwire marker in
the diagram. `prefers-reduced-motion` removes both and the page is fully readable with JS off.

**Reflex check:** (a) category reflex = black page, neon HUD, "AI-powered" gradient hero, stat row
of three numbers; (b) second-order reflex = dark "military" + amber + terminal type; (c) chosen
deviation = light technical atlas, serif display, red used once and semantically, measurements as a
table with method notes (truth over vibes).

**House tells broken (auteur §2.5):** (1) *not a dark page* — eight of nine auteur showcase builds
were dark; this one is deliberately light. (2) *no mono service labels as the whole identity* —
mono is reserved for data; headlines are serif, body is a humanist sans.

**Verified gates:** slopscan run against `site/` before delivery. Images are composited from REAL
SAM 3.1 masks produced on this machine (not decorated stock), and every number in the table comes
from `out/falcon_results.json` measured 15 Sept 2026.
