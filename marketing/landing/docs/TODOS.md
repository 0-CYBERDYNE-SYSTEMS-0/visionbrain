# TODO — VisionBridge roof-intrusion demo kit (event security GTM)

## goal
Assets to advertise a field-deployable visual-intelligence system: detects a human on a roof,
paints the frame, pushes the annotated image to field phones. Landing page + diagram + demo images.

## checklist
- [x] recon: real stack (VisionBrain repo, Falcon-Perception + SAM 3.1 cached, fp_inference API)
- [x] recon: Butler PA facts (Wikipedia, sourced)
- [x] generate scenes A (day roof+field), B (dusk perimeter) via local FLUX Klein :4030
- [x] generate scene C (close roof crop)
- [x] run REAL masking (Falcon detect/segment + SAM 3.1) — measured: tripwire detect 1.3 s warm, SAM 3.1 mask 1.8 s warm (constant), Falcon hi-res detect 3.9 s / mask 9.8–11 s, cold 1.8–35 s
- [x] composite product-grade overlays (PIL) → 3 final assets in out/ (hero, wide+magnifier, field alert)
- [x] field-alert asset (phone-shaped broadcast mock)
- [x] landing page + inline SVG diagram (site/index.html) + DESIGN-NOTES.md commit-sheet
- [x] slopscan gate: 0 fails / 0 warns; negative control verified the gate detects 4 seeded violations
- [x] serve on :8024 (tailnet http://100.72.41.118:8024) + opened in browser + images delivered in chat
- [ ] pending: user approves → publish (here.now / deploy), rename if wanted, journal-feature wiring

## facts (sourced — Wikipedia, Attempted assassination of Donald Trump in Pennsylvania)
- Crooks scaled an A/C unit to the AGR International roof; crossed interconnected roofs to the southernmost
- firing position 400–450 ft (120–140 m) north of the stage
- the building housed three police counter-snipers tasked with covering the rally — none were on the roof (staffing shortage)
- roof slant may have blocked USSS snipers' view; trees blocked the northern team
- bystanders saw him on the roof and alerted police minutes before the shots
- regarded as the Secret Service's most significant security failure since 1981

## notes
- assets: ~/.hermes/workspace/visionbridge-demo/  (scenes/, out/)
- real measured latency goes on the page — no invented numbers
- copy: factual, no hype words (auteur ban list bans "Seamless/Effortless/Unleash/Elevate/Revolutionize")
