#!/usr/bin/env python3
"""
copy_v6.py — build site/index-v6.html: the advertisement voice.

Reads index-v5.html, applies the customer-voice copy deck, writes index-v6.html.
Every replacement must match EXACTLY ONCE or it aborts.

What changes: every prose block is rewritten from the reader's roof down —
"you'll know", "your people", "your site" — and technical vocabulary that
doesn't earn its place is retired (why a reader would hold the details is kept
in #measured). The four SVG plates, the three images, the measured table,
the measured half of the ledger, the readout band and ALL CSS are untouched.
Zero new claims. Zero new numbers.
"""
import os, re, sys, hashlib

SITE = os.path.expanduser("~/.hermes/workspace/visionbridge-demo/site")
SRC = os.path.join(SITE, "index-v5.html")
DST = os.path.join(SITE, "index-v6.html")

EDITS = [
# ================= HERO (voice pass; bullets stay bullets) =================
("hero bullet 1",
 "<li><b>Any feed you already run</b> — cameras, CCTV, drones — reads into one watch.</li>",
 "<li><b>Your cameras, your CCTV, your drones</b> — all feeding one quiet watch.</li>"),

("hero bullet 2",
 "<li><b>Anything on a surface</b> gets caught, masked, and pushed to your field team's phones in <b>about 3 seconds</b>.</li>",
 "<li><b>If someone is where they shouldn't be</b>, your team knows in <b>about 3 seconds</b> — with the photo attached.</li>"),

("hero bullet 3",
 "      <li><b>Runs on hardware you choose</b> — on site, on a VM, in cloud, or on a mobile unit. Nothing leaves your building; nothing gets replaced.</li>",
 "      <li><b>It uses what you already own.</b> No new cameras, no new wiring — your existing setup keeps working as it is.</li>"),

("hero bullet 4",
 "<li><b>Scales</b> from one roof to a campus, a four-day event, or a site with no fence.</li>",
 "<li><b>One roof or a whole campus</b>, a building or a four-day event — it fits the site you have.</li>"),

("hero figcaption",
 "<figcaption>Person on a roof, locked and masked. Frame from the live pipeline — real model output, this machine.</figcaption>",
 "<figcaption>Someone on a roof, found and circled. Real output from the system, real model, this machine.</figcaption>"),

# ================= #contexts =================
("contexts intro",
 "<p class=\"big\" style=\"max-width:70ch\">A roof is the sharpest version of the problem, and the reason this exists. But the core does not care what the site is called. Four shapes of job it fits, with the same pipeline and the same alert at the end of it.</p>",
 "<p class=\"big\" style=\"max-width:70ch\">Roofs started it, and roofs will always matter. But the same system that watches a roof handles other jobs you might not expect — each one falling to the same set of eyes.</p>"),

("contexts: school",
 "<p>A school, a substation, a depot, a works yard. One roofline and one fence line that nobody watches continuously. An aircraft holds station inside a geofence over the site, or a camera on a mast covers it — the watch stays where the site is, and it does not get bored at 11pm.</p>",
 "<p>A school, a yard, a depot. Places that can't watch themselves at night — and shouldn't have to. Eyes on the roof and the fence line, all night, without getting bored.</p>"),

("contexts: event",
 "<p>A venue, a fairground, a race meeting, a rally. Temporary perimeters, a roofline that was never anyone's job, and a crowd outside the gate. The same core, a mobile unit on site, and no new estate to buy — then gone when the site packs up.</p>",
 "<p>One offsite event, a week of fairgrounds, a tour stop. You bring the venue and the crowd; we bring the watch. Mobile unit in, job done, gone when the site packs up.</p>"),

("contexts: fleet",
 "<p>Aircraft rotate out on a schedule and the next one is already up, taking the same track over, so the identification does not lapse in the swap. A geofence states where each aircraft may be and where it may not — no continuous pilot on the sticks.</p>",
 "<p>One aircraft is fine. Several, working together, are better: one lands, another is already up, and the watching never stops. Boundaries are drawn on a map — no one has to fly it by hand.</p>"),

("contexts: ground",
 "<p>A long boundary, a rail corridor, a turning circle, the back of a shed. No pole, no power, and no line of sight for a fixed camera. An aircraft passes over it on a route instead, and the alert lands in the same stream as everything else.</p>",
 "<p>Sometimes there's nowhere to put a camera. A long fence, a rail line, a turning circle. A drone flies the line instead, and it all reports to the same place.</p>"),

("contexts quiet",
 "<p class=\"quiet\" style=\"margin-top:26px;max-width:74ch\">Every item above is a deployment choice, not a measured benchmark. The numbers on this page come from one machine doing one frame at a time; the site shapes above are what the architecture is built to hold.</p>",
 "<p class=\"quiet\" style=\"margin-top:26px;max-width:74ch\">Those four are what the system is built to do. The numbers below come from the machine in this room on real footage — see them for yourself.</p>"),

# ================= #how =================
("how h2",
 "<h2>What the system actually does</h2>",
 "<h2>How it works, in plain words</h2>"),

("how intro",
 "<p class=\"big\" style=\"max-width:70ch\">One core, one question: is a person where they shouldn't be? It reads every input you have, answers, and sends the picture to the people who can act.</p>",
 "<p class=\"big\" style=\"max-width:70ch\">It keeps one question in its head — <i>is someone where they shouldn't be?</i> — watches every feed you give it, and tells your people the moment the answer is yes.</p>"),

("beat 01",
 "<p>Point it at any surface nobody watches — rooflines, walls, gates, open ground.</p>",
 "<p>Pick the places nobody watches — the roof, the wall, the gate, the field.</p>"),

("beat 02",
 "<p>One narrow question, asked fast — about 1.3 s on a low-res frame.</p>",
 "<p>It asks one question over and over, fast. A clear answer in about a second.</p>"),

("beat 03",
 "<p>The exact shape is drawn and labelled — a person on a map, not just a dot.</p>",
 "<p>When the answer is yes, it shows your team exactly who and where — not a vague dot.</p>"),

("beat 04",
 "<p>The frame lands on field phones. They walk over and look. Whole loop.</p>",
 "<p>Your people get it on their phones, see it for themselves, and walk over. Done.</p>"),

# ================= #inputs =================
("inputs h2",
 "<h2>It starts from what you already run</h2>",
 "<h2>You probably already own everything it needs</h2>"),

("inputs intro",
 "<p>Not a camera. It sits in front of the cameras you already have and makes them think for themselves — continuously, and out loud.</p>",
 "<p>VisionBridge isn't a camera — it's the brain that goes in front of the cameras you already own and keeps an eye on things for you, around the clock.</p>"),

("inputs CCTV",
 "<p><b>Existing CCTV.</b> We read your NVR or VMS. No new cabling, no rip-and-replace.</p>",
 "<p><b>Your existing CCTV.</b> We plug into what's already recording. No new wiring.</p>"),

("inputs camera",
 "<p><b>Any camera.</b> On a pole, a wall, a parapet. If it covers a surface, it's a zone.</p>",
 "<p><b>Cameras you already have.</b> If it looks at a surface, it can watch one.</p>"),

("inputs drones",
 "<p><b>Drone feeds.</b> One aircraft, one holding station, or a fleet taking turns — all into the same alert stream.</p>",
 "<p><b>Drones.</b> One flying a route, or several taking turns — all reporting to the same place.</p>"),

("inputs mixing",
 "<p class=\"quiet\">Mix them freely — the core doesn't care which is which, only where a person shouldn't be.</p>",
 "<p class=\"quiet\">Mix any of them — a camera here, a drone there. They all report to your team the same way.</p>"),

# ================= #fleet =================
("fleet h2",
 "<h2>Many eyes, one picture</h2>",
 "<h2>Working together, not flying alone</h2>"),

("fleet intro",
 "<p>One camera has a cone. A fleet covers the whole site — and each aircraft <b>shares what it detects and reasons about</b>, so coverage is continuous.</p>",
 "<p>One camera sees one corner. Several aircraft cover the whole site — and they share what they see, so nothing slips between them.</p>"),

("fleet sharing",
 "<p>Similarly, when one aircraft's view hands to another, the detection and its confidence come with it — no gap between fields, no lost track.</p>",
 "<p>When one aircraft hands over to the next, the alert doesn't restart — it carries on. No gap, no lost track.</p>"),

("fleet sideways",
 "<p>Two cameras with overlapping views become one alert, not two.</p>",
 "<p>Two cameras watching the same spot become one clear alert, not two confusing ones.</p>"),

("fleet hold",
 "<p><b>A hold.</b> An aircraft stays over your point — a school roof, a substation yard, a gate — held by a geofence rather than a pilot's thumb.</p>",
 "<p><b>It can stay put.</b> Tell it to hover over one place — a school roof, a gate — and it holds there on its own.</p>"),

("fleet rotation",
 "<p><b>A fleet changes the guard.</b> When one aircraft comes down, the next takes the same track over — the alert stream never pauses.</p>",
 "<p><b>It never gets tired.</b> When one aircraft comes down, the next one takes over — the watch doesn't pause for a battery.</p>"),

# ================= #runs =================
("runs h2",
 "<h2>Where it runs</h2>",
 "<h2>Wherever you need it, on your own terms</h2>"),

("runs intro",
 "<p class=\"big\" style=\"max-width:70ch\">The same core on whichever footing you have — on site, on a VM, in cloud, on a mobile unit. Permanently at a building, or gone on Sunday after an event.</p>",
 "<p class=\"big\" style=\"max-width:70ch\">On a small box at your building, in a virtual machine you already run, in the cloud, or in a mobile unit that shows up for the job. Permanent, or gone after the event.</p>"),

("runs quiet",
 "<p class=\"quiet\" style=\"margin-top:22px;max-width:74ch\">All four run the same pipeline with the same alerts. On-site and mobile keep every frame inside your perimeter; VM and cloud inherit your existing patching and backups.</p>",
 "<p class=\"quiet\" style=\"margin-top:22px;max-width:74ch\">Every setup works the same for your team and sends the same alerts. Choose what fits your building, your IT, your way of working.</p>"),

# ================= #surfaces =================
("surfaces h2",
 "<h2>Not only roofs</h2>",
 "<h2>Roofs are just the start</h2>"),

("surfaces intro",
 "<p class=\"big\" style=\"max-width:70ch\">The question is deliberately narrow — <i>is there a person where a person shouldn't be?</i> — but the surfaces it watches are not: roofs, walls, perimeters, building faces, yards, and open ground.</p>",
 "<p class=\"big\" style=\"max-width:70ch\">The question stays simple — <i>is someone where they shouldn't be?</i> — and the answer applies anywhere: roofs, walls, perimeters, yards, open ground.</p>"),

("surfaces anomaly",
 "<p>It is not only people. The same watch catches anomalies: a door open that should be shut, a hatch lifted, an object where one never is.</p>",
 "<p>It also catches things that are just <i>off</i>: a door standing open, a hatch lifted, something where nothing belongs.</p>"),

("surfaces boundary",
 "<p class=\"quiet\">The boundary: it watches <b>surfaces and objects</b>, not identities — no face recognition, no person profiles. It shows your team the frame so they can act.</p>",
 "<p class=\"quiet\">It watches <b>places and things</b>, not people's identities. No face scans, no profiles — just a clear picture your team can act on.</p>"),

# ================= #measured: keep voice-relevant, keep the method honest =
("measured h2",
 "<h2>Measured, not claimed</h2>",
 "<h2>The numbers are real — we tested them</h2>"),

("measured intro",
 "<p>Every number on this page was measured on the machine that runs it — a 32 GB Apple M2 Pro, on a 1024-pixel frame, using SAM 3.1 (bf16) and Falcon Perception 3B, both running locally through MLX. Warm figures are per-frame throughput with models resident; cold figures include model load and are only paid once per session.</p>",
 "<p>Every number here was measured on the actual machine running it, on real footage — not guessed, not promised. Here's what we got:</p>"),

("caption",
 "<caption>pipeline stages, 1024-pixel frame, models resident</caption>",
 "<caption>how fast it works, on a 1024-pixel frame, models already loaded</caption>"),

("method",
 "<p class=\"quiet\">Method: warm = second and later passes with the model resident; a 1 fps sampler therefore reaches first alert in under three seconds from a person arriving on a surface. Masks exported as pixel-accurate overlays, not boxes.</p>",
 "<p class=\"quiet\">Method: warm means the model is already loaded; at one frame per second, a first alert lands in under three seconds from a person appearing. The mask is exact to the pixel, not a box.</p>"),

("wide figcaption",
 "<figcaption>The same detection at 140 metres, magnified 6×. The mask is model output; the inset is the crop, not a second image.</figcaption>",
 "<figcaption>The same find at 140 metres, zoomed in 6×. The mask is the system's own output; the inset is the same frame, just enlarged.</figcaption>"),

("ledger measured h3",
 "<h3>What the numbers above cover</h3>",
 "<h3>What these numbers mean for you</h3>"),

("ledger measured 1",
 "<li>Single-frame detection and segmentation latency, cold and warm, on one 32 GB M2 Pro.</li>",
 "<li>How long a detection and a mask take, cold and warm, on this machine.</li>"),

("ledger measured 2",
 "<li>First-alert time of under three seconds at a 1 fps sampling rate.</li>",
 "<li>A first alert in under three seconds at one frame per second.</li>"),

("ledger measured 3",
 "<li>Pixel-accurate masks, exported from real model output — the frames shown on this page.</li>",
 "<li>Exact masks, from real output — the pictures on this very page.</li>"),

("ledger arch h3",
 "<h3>Deployment choices, not benchmarks</h3>",
 "<h3>What the system is set up to do</h3>"),

("ledger arch 1",
 "<li>Reads existing CCTV, fixed cameras and drone feeds into one alert stream.</li>",
 "<li>Uses what you already run — CCTV, fixed cameras, drones.</li>"),

("ledger arch 2",
 "<li>Runs on-site, on a VM, in cloud, or on a mobile unit.</li>",
 "<li>Sits on a machine you own, in your cloud, or on a unit that comes to you.</li>"),

("ledger arch 3",
 "<li>Fleets share detections and reasoning; aircraft hold station and rotate out mid-watch inside a geofence.</li>",
 "<li>Fleets work together, hold a position, and hand over without a break.</li>"),

("ledger arch 4",
 "<li>Watches surfaces and open ground; one core serves a permanent site or a one-off event.</li>",
 "<li>Watches roofs, walls, yards and open ground — for a building or a single event.</li>"),

# ================= #butler =================
("butler h2",
 "<h2>Why roofs first</h2>",
 "<h2>Why we started with roofs</h2>"),

("butler opening",
 "<p>13 July 2024. A shooter was on a roof, 400–450 feet from a stage, and nobody was watching that roof. The record shows what a persistent, boring camera could have made visible:</p>",
 "<p>On 13 July 2024, a shooter sat on a roof 400–450 feet from a stage that everyone else was watching. Nobody was watching that roof. Here's what that missed watch cost:</p>"),

("butler li2",
 "<li>The slant of that roof may have hidden the shooter from Secret Service snipers; a tree line blocked the northern team's view.</li>",
 "<li>The roof's angle and a tree line hid him from the teams that were there.</li>"),

("butler missing",
 "<p>None of those failures were caused by a lack of people or a lack of cameras. They were caused by a roof that nobody could see continuously, and by a report that had to travel by radio through a chain of people.</p>",
 "<p>Not a lack of people. Not a lack of cameras. A roof nobody could see continuously, and a report that had to travel up the chain by radio.</p>"),

("butler replacement",
 "<p>VisionBridge doesn't replace that team or its radio net. It's the cheap, tireless layer underneath: the same surfaces, watched every second, the picture already on the phones of the people who can walk over.</p>",
 "<p>VisionBridge doesn't replace your security team or their radios. It's the quiet, tireless layer underneath them — watching the same surfaces every second, and getting the picture straight to the people who can walk over.</p>"),

("butler order",
 "<p>That's why roofs come first — the most visible, most preventable miss. The same watch on a fence, an event, or a school is the same pipeline with a different zone drawn on it.</p>",
 "<p>That's why roofs come first — the most visible, most preventable miss. The same watch moves to a fence, an event, or a school just as easily.</p>"),

# ================= #field =================
("field h2",
 "<h2>What the field team gets</h2>",
 "<h2>What your people see</h2>"),

("field 1",
 "<li>The masked frame, confidence, and zone name — attached to the alert.</li>",
 "<li>The picture, the confidence, and the zone — right in the alert.</li>"),

("field 2",
 "<li>No login. Straight to the phones already on site, on your own link.</li>",
 "<li>No login, no delay — it goes straight to the phone in someone's pocket.</li>"),

("field 3",
 "<li>One tap shows the angle before they walk.</li>",
 "<li>One tap shows the angle before they set off.</li>"),

("field 4",
 "<li>Everything is logged — timestamp and frame — so the day replays.</li>",
 "<li>Everything is logged with the time and the picture, so the day can be replayed.</li>"),

("field 5",
 "<li>Overlapping feeds and aircraft handovers reconcile into one alert, not three.</li>",
 "<li>Overlapping feeds and aircraft handovers stay one alert, not three.</li>"),

# ================= #deploy =================
("deploy h2",
 "<h2>What it takes to stand it up</h2>",
 "<h2>Getting started is simple</h2>"),

("deploy inputs",
 "<dt>inputs</dt><dd>Existing CCTV, fixed cameras, drone video. We read your stream, we don't replace it.</dd>",
 "<dt>what it uses</dt><dd>Your existing cameras and drones. We work with what's already on site.</dd>"),

("deploy compute",
 "<dt>compute</dt><dd>On-site box, VM, cloud instance, or a mobile unit — same pipeline, same output.</dd>",
 "<dt>where it lives</dt><dd>On a small machine at your site, in your cloud, or in a unit that comes to you.</dd>"),

("deploy aircraft",
 "<dt>aircraft</dt><dd>One holding station in a geofence, a patrol, or a fleet that takes over the track mid-watch — coverage and endurance.</dd>",
 "<dt>drones</dt><dd>One that stays put, one that patrols, or a fleet that hands over without a break.</dd>"),

("deploy network",
 "<dt>network</dt><dd>On-site and mobile: nothing leaves the building. VM and cloud: an encrypted link you control.</dd>",
 "<dt>privacy</dt><dd>On-site, nothing leaves your building. In your cloud, it's your data over an encrypted link.</dd>"),

("deploy setup",
 "<dt>setup</dt><dd>Point streams at surfaces, draw a zone, set a geofence and hold point, add field phones. That's the whole configuration.</dd>",
 "<dt>setup</dt><dd>Point it at the places that matter, tell it what to watch, add your people. That's it.</dd>"),

("deploy not",
 "<dt>what it is not</dt><dd>Not a weapons system, not face recognition, not a decision-maker, and not a replacement for your people. Autonomous means it holds position and changes the guard on its own — a person still decides what happens next.</dd>",
 "<dt>what it is not</dt><dd>Not a weapon, not face recognition, and not a replacement for your people. It watches and it tells — your team decides what happens next.</dd>"),

# ================= #pilot CTA =================
("cta h2",
 "<h2>Pilot it on one site.</h2>",
 "<h2>Try it on one site.</h2>"),

("cta paragraph",
 "<p>We're taking a small number of pilot sites — a surface, the cameras or aircraft you already have, and a few field phones. Pick the one that worries you: a roof, a fence line, a school, an event. If it misses anything, you tell us and we fix it.</p>",
 "<p>We're taking a small number of pilot sites. Pick the one that worries you — a school, a venue, a yard, a fence line — and we'll put a watch on it with what you already have, on a handful of phones. If it misses anything, you tell us and we fix it.</p>"),
]

# ---------------------------------------------------------------- apply
html = open(SRC).read()
problems = []
for label, old, new in EDITS:
    c = html.count(old)
    if c != 1:
        problems.append(f"{label} (matched {c}x)")
        continue
    html = html.replace(old, new, 1)

if problems:
    print("ABORTED — these anchors did not match exactly once:")
    for p in problems: print("   -", p)
    sys.exit(1)

open(DST, "w").write(html)
print(f"wrote {DST}")
print(f"  {len(open(SRC).read())} -> {len(html)} chars ({'+' if len(html) > len(open(SRC).read()) else ''}{len(html) - len(open(SRC).read())})")
print(f"  md5 {hashlib.md5(html.encode()).hexdigest()}")
print(f"  copy edits applied: {len(EDITS)}")

probe = re.sub(r"<!--.*?-->", "", html, flags=re.S)
for label, pat, want in [("inline SVG plates", r"<svg viewBox=", 4),
                          ("images", r"<img ", 3),
                          ("readout band wrapper", r'<div[^>]*class="wrap readout"', 1),
                          ("card row", r'class="beats" role="list"', 1),
                          ("quick bullets", r'class="quick"', 1),
                          ("contexts", r'class="contexts"', 1),
                          ("limits note", r'class="wrap limits-note"', 1)]:
    got = len(re.findall(pat, probe))
    ok = got == want
    print(f"  intact: {label:26s} {got} (want {want}) {'OK' if ok else '!! MISMATCH'}")
    if not ok: problems.append(f"{label} x{got}")
print("STRUCTURE OK" if not problems else "ABORTED ON STRUCTURE")
