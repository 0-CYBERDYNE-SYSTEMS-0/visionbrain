#!/usr/bin/env python3
"""
copy_v4.py — build site/index-v4.html: broadened scope, same visuals.

Reads index-v3.html, applies the v4 copy deck, writes index-v4.html.
Every replacement must match EXACTLY ONCE or the script aborts without writing —
a stale anchor fails loudly instead of silently skipping a section.

NOTHING here touches the four inline SVG plates, the three images, or the plate CSS.
The only CSS addition is a self-contained block for the new "where it holds" section.
"""
import os, re, sys, hashlib

SITE = os.path.expanduser("~/.hermes/workspace/visionbridge-demo/site")
SRC = os.path.join(SITE, "index-v3.html")
DST = os.path.join(SITE, "index-v4.html")

# ---------------------------------------------------------------- new section
CONTEXTS = '''<section class="wrap" id="contexts">
    <h2>Where it holds</h2>
    <p class="big" style="max-width:70ch">A roof is the sharpest version of the problem, and the reason this exists. But the core does not care what the site is called. Four shapes of job it fits, with the same pipeline and the same alert at the end of it.</p>

    <div class="contexts">
      <div>
        <h3>A site that stands still</h3>
        <p>A school, a substation, a depot, a works yard. One roofline and one fence line that nobody watches continuously. An aircraft holds station inside a geofence over the site, or a camera on a mast covers it — the watch stays where the site is, and it does not get bored at 11pm.</p>
      </div>
      <div>
        <h3>An event with a date on it</h3>
        <p>A venue, a fairground, a race meeting, a rally. Temporary perimeters, a roofline that was never anyone's job, and a crowd outside the gate. The same core, a mobile unit on site, and no new estate to buy — then gone when the site packs up.</p>
      </div>
      <div>
        <h3>A fleet instead of a camera</h3>
        <p>Aircraft rotate out on a schedule and the next one is already up, taking the same track over, so the identification does not lapse in the swap. A geofence states where each aircraft may be and where it may not — no continuous pilot on the sticks.</p>
      </div>
      <div>
        <h3>Ground you cannot put a camera on</h3>
        <p>A long boundary, a rail corridor, a turning circle, the back of a shed. No pole, no power, and no line of sight for a fixed camera. An aircraft passes over it on a route instead, and the alert lands in the same stream as everything else.</p>
      </div>
    </div>

    <p class="quiet" style="margin-top:26px;max-width:74ch">Every item above is a deployment choice, not a measured benchmark. The numbers on this page come from one machine doing one frame at a time; the site shapes above are what the architecture is built to hold.</p>
  </section>

  '''

# ---------------------------------------------------------------- copy deck
# (label, old, new)
EDITS = [
# ---- head
("nav: add Where it holds",
 '<a href="#fleet">Drone fleets</a>',
 '<a href="#fleet">Drone fleets</a>\n    <a href="#contexts">Where it holds</a>'),

# ---- hero
("hero h1",
 "<h1>A roof is a blind spot.</h1>",
 "<h1>A roof is a blind spot. It is not the only one.</h1>"),

("hero lede 1",
 "<p class=\"lede\">VisionBridge turns <b>the cameras, CCTV networks and drones you already run</b> into one persistent watch over roofs, walls and perimeters — and puts a masked frame of anything that shouldn't be there on your field team's phones in about three seconds.</p>",
 "<p class=\"lede\">VisionBridge turns <b>the cameras, CCTV networks and drones you already run</b> into one persistent watch — over a roofline, a fence line, a whole event site, or a boundary with nothing to bolt a camera to — and puts a masked frame of anything that shouldn't be there on your field team's phones in about three seconds.</p>"),

("hero lede 2",
 "<p class=\"lede\">It runs on hardware you choose: a machine on site, a virtual machine, a cloud instance, or a mobile unit that drives to the job. Nothing has to leave your building, and none of your cameras get replaced.</p>",
 "<p class=\"lede\">It runs on hardware you choose: a machine on site, a virtual machine, a cloud instance, or a mobile unit that drives to the job. Nothing has to leave your building, and none of your cameras get replaced. The same watch scales from one roof to a campus, a four-day event, or a site with no fence at all.</p>"),

# ---- readout band: broaden surfaces, add an aircraft line (still architecture side)
("readout row: add aircraft",
 "<div><b>inputs</b><span>existing CCTV · fixed cameras · drone feeds · one core reads all of them</span></div>",
 "<div><b>inputs</b><span>existing CCTV · fixed cameras · drone feeds · one core reads all of them</span></div>\n    <div><b>aircraft</b><span>one holding station inside a geofence · patrol · a fleet rotating out mid-watch</span></div>"),

("readout row: surfaces",
 "<div><b>surfaces</b><span>roofs · walls · perimeters · building faces</span></div>",
 "<div><b>surfaces</b><span>roofs · walls · perimeters · building faces · open ground</span></div>"),

# ---- #how
("how intro",
 "<p class=\"big\" style=\"max-width:70ch\">Inputs come in from wherever you already have eyes. One reasoning core asks the narrow question, draws the answer, and merges what every view is telling it. The result goes to the people who can walk over and look.</p>",
 "<p class=\"big\" style=\"max-width:70ch\">Inputs come in from wherever you already have eyes — a camera estate you already own, one aircraft overhead, a fleet taking turns, or all of them at once. One reasoning core asks the narrow question, draws the answer, and merges what every view is telling it. The result goes to the people who can walk over and look.</p>"),

("beat 01",
 "<p>Your existing cameras, a handful of cheap ones, or a drone overhead. Rooflines, walls, gates, the outside face of the perimeter — the places a person can be and nobody can see.</p>",
 "<p>Your existing cameras, a handful of cheap ones, or an aircraft holding a position overhead. Rooflines, walls, gates, the outside face of the perimeter, open ground — the places a person can be and nobody can see.</p>"),

# ---- #inputs
("inputs: drone feeds",
 "<p><b>Drone feeds.</b> A single aircraft on a slow orbit, or a fleet. Video comes down to the same core and lands in the same alert stream as the fixed cameras — one queue, one picture.</p>",
 "<p><b>Drone feeds.</b> A single aircraft on a slow orbit, one holding station over a fixed point, or a fleet that takes turns. Video comes down to the same core and lands in the same alert stream as the fixed cameras — one queue, one picture.</p>"),

# ---- #fleet: the station-keeping + autonomous rotation capability
("fleet: add hold + rotation",
 "<p>The same sharing works sideways. Two fixed cameras with overlapping views, or a fixed camera and a drone passing over it, reconcile into a single tracked picture instead of two unrelated alerts on two screens.</p>",
 "<p>The same sharing works sideways. Two fixed cameras with overlapping views, or a fixed camera and a drone passing over it, reconcile into a single tracked picture instead of two unrelated alerts on two screens.</p>\n        <p><b>A hold is the other half of it.</b> An aircraft can be told to stay over one point — a school roof, a substation yard, a gate approach — and it stays there, because the site boundary and the flight volume are drawn as a geofence rather than flown by hand on the sticks.</p>\n        <p><b>And a fleet can change the guard.</b> When the aircraft on station needs to come down, the next one is already airborne and takes the same track over. The detection, its shape and its confidence hand across the swap, so the identification does not lapse while one aircraft replaces another. From the field team's side it is one continuous alert stream, not a series of separate flights.</p>"),

# ---- #runs
("runs intro",
 "<p class=\"big\" style=\"max-width:70ch\">The same core, on whichever footing you already have — or on one you can carry to the site. There is no single blessed deployment, and nothing about the product forces your video into somebody else's data centre.</p>",
 "<p class=\"big\" style=\"max-width:70ch\">The same core, on whichever footing you already have — or on one you can carry to the site. There is no single blessed deployment, and nothing about the product forces your video into somebody else's data centre. It is the same whether the job is one building that never moves or an event that packs up on Sunday.</p>"),

# ---- #surfaces
("surfaces intro",
 "A roof is simply the first one, because it is the one nobody watches.</p>",
 "A roof is simply the first one, because it is the one nobody watches. The list after it is not short: fence lines and gate approaches, building faces, yards, and the open ground around an event where the perimeter exists only as a line on a map.</p>"),

# ---- #measured: extend the ARCHITECTURE side only (the measured side is untouched)
("ledger: architecture list",
 "<li>Surfaces beyond the roof: walls, perimeters, building faces, and scene anomalies.</li>",
 "<li>Surfaces beyond the roof: walls, perimeters, building faces, open ground, and scene anomalies.</li>\n          <li>An aircraft holding station over a fixed point, and a fleet rotating out mid-watch with the track handed over between aircraft inside a geofence.</li>\n          <li>The same watch applied to a site that stands still — a school, a yard, a substation — or a temporary one, such as an event or a fairground.</li>"),

# ---- #butler: reframe the section as the anchor case, not the whole scope
("butler h2",
 "<h2>Why roofs</h2>",
 "<h2>Why roofs first</h2>"),

("butler: closing para",
 "<p class=\"stamp\">Sources: public reporting on the 13 July 2024 shooting",
 "<p>It also explains the order we have gone in. Roofs come first because that is where the failure was most visible and most preventable. The same watch turned to a fence line, an event, or a site that stands still is the same pipeline doing the same job with a different zone drawn on it.</p>\n        <p class=\"stamp\">Sources: public reporting on the 13 July 2024 shooting"),

# ---- #field
("field: last bullet",
 "<li>When several feeds cover the same ground, they reconcile into one alert rather than three.</li>",
 "<li>When several feeds cover the same ground — or when one aircraft hands over to another mid-watch — they reconcile into one alert rather than three.</li>"),

# ---- #deploy
("deploy: aircraft",
 "<dt>aircraft</dt><dd>One drone is useful for a moving view. A fleet is useful for area coverage: overlapping fields, with detections and reasoning shared between aircraft so a person moving across the site never leaves the watch.</dd>",
 "<dt>aircraft</dt><dd>One aircraft is useful for a moving view, or one holding station over a fixed point inside a geofence. A fleet is useful for coverage and for endurance: overlapping fields, detections and reasoning shared between aircraft, and the next aircraft taking over the track when the one on station needs to come down.</dd>"),

("deploy: setup",
 "<dt>setup</dt><dd>Streams pointed at the surfaces you care about, a zone drawn around each, the field phones added to the alert list. That is the whole configuration.</dd>",
 "<dt>setup</dt><dd>Streams pointed at the surfaces you care about, a zone drawn around each, a geofence and a hold point set for any aircraft, the field phones added to the alert list. That is the whole configuration.</dd>"),

("deploy: what it is not",
 "<dd>Not a weapons system, not face recognition, and not a replacement for your security plan or your radio net.",
 "<dd>Not a weapons system, not face recognition, and not a replacement for your security plan or your radio net. It also is not a decision-maker: autonomous here means the aircraft keeps its own position and the fleet changes the guard without a pilot on the sticks — a person still decides what happens to anything it reports."),

# ---- #pilot
("cta paragraph",
 "<p>We're taking a small number of pilot sites — a surface, the cameras or aircraft you already have, and a few field phones. If it misses anything that matters, you tell us and we fix it.",
 "<p>We're taking a small number of pilot sites — a surface, the cameras or aircraft you already have, and a few field phones. A roof, a fence line, a school that stands still, or an event with a date on it: pick the one that worries you. If it misses anything that matters, you tell us and we fix it."),

# ---- footer
("footer",
 "<div>VisionBridge · roof, wall and perimeter watch · runs on your hardware ·",
 "<div>VisionBridge · a persistent watch over roofs, perimeters and open ground · runs on your hardware ·"),
]

# ---------------------------------------------------------------- CSS for the new section
CSS = '''

/* ---------------------------------------------------------------- v4 · "where it holds"
   Four parallel site shapes. Deliberately NO ordinal eyebrow: plate 3's card row is
   numbered because it is a sequence, and these are not — they are four equal choices.
   Same grid and hairline language as the existing card row so the page has one hand. */
.contexts {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0;
  margin-top: 34px;
  border-top: 1px solid var(--rule);
}
.contexts > div { padding: 22px 22px 4px 0; }
.contexts > div + div { border-left: 1px solid var(--rule); padding-left: 22px; }
.contexts h3 { font-family: Fraunces, Georgia, serif; font-size: 17px; margin: 0 0 8px; letter-spacing: -0.01em; }
.contexts p { margin: 0; font-size: 14.5px; color: var(--graphite); }
@media (max-width: 900px) {
  .contexts { grid-template-columns: 1fr 1fr; }
  .contexts > div { border-left: 0; padding-left: 0; }
}
@media (max-width: 560px) {
  .contexts { grid-template-columns: 1fr; }
}
'''

# ---------------------------------------------------------------- apply
html = open(SRC).read()
before = len(html)
problems = []

# title + meta by regex (their exact strings are long; match structurally)
html, n = re.subn(r"<title>.*?</title>",
  "<title>VisionBridge — one persistent watch over roofs, perimeters and open ground, on any camera or drone feed</title>",
  html, count=1, flags=re.S)
if n != 1: problems.append("title")
html, n = re.subn(r'<meta name="description" content="[^"]*">',
  '<meta name="description" content="VisionBridge turns the CCTV, cameras and drone feeds you already run into one persistent watch — over a roofline, a fence line, a whole event site, or an open boundary — and puts a masked frame of anything that should not be there on your field team\'s phones in about three seconds. A drone can hold station over a fixed point, or a fleet can rotate out mid-watch inside a geofence. Runs on an on-site machine, a virtual machine, a cloud instance or a mobile unit.">',
  html, count=1, flags=re.S)
if n != 1: problems.append("meta")

# the copy deck
for label, old, new in EDITS:
    c = html.count(old)
    if c != 1:
        problems.append(f"{label} (matched {c}x)")
        continue
    html = html.replace(old, new, 1)

# insert the new section immediately before #runs
anchor = '<section class="wrap" id="runs">'
if html.count(anchor) != 1:
    problems.append(f"#runs anchor (matched {html.count(anchor)}x)")
else:
    html = html.replace(anchor, CONTEXTS + anchor, 1)

# append the CSS inside the page's single <style>
if "</style>" not in html:
    problems.append("</style>")
else:
    html = html.replace("</style>", CSS + "</style>", 1)

if problems:
    print("ABORTED — these anchors did not match exactly once:")
    for p in problems:
        print("   -", p)
    sys.exit(1)

open(DST, "w").write(html)
print(f"wrote {DST}")
print(f"  {before} -> {len(html)} chars  (+{len(html) - before})")
print(f"  md5 {hashlib.md5(html.encode()).hexdigest()}")
print(f"  copy edits applied: {len(EDITS)}   new section: #contexts   CSS block: .contexts")
# sanity: the visuals must be intact (comment-stripped — the fragments' documentation
# comments quote tag names, and counting raw text flags them as structure)
probe_src = re.sub(r"<!--.*?-->", "", html, flags=re.S)
for probe, expect in [("<svg viewBox=", 4), ("assets/01-hero-locked-person.jpg", 1),
                      ("assets/02-wide-magnified.jpg", 1), ("assets/03-field-alert.jpg", 1),
                      ('class="beats" role="list"', 1), ('class="wrap readout"', 1)]:
    got = probe_src.count(probe)
    print(f"  intact: {probe:34s} {got} (expected {expect}) {'OK' if got == expect else '!! MISMATCH'}")
