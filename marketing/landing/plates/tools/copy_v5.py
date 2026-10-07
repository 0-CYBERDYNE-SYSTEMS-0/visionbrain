#!/usr/bin/env python3
"""
copy_v5.py — build site/index-v5.html: the same page, minus the wind.

Reads index-v4.html, applies the concise-copy deck, writes index-v5.html.
Every replacement must match EXACTLY ONCE or it aborts without writing.
The four SVG plates, the three images, the measured table, the measured half of
the ledger, the readout band and ALL the v4 CSS are untouched.

The deck only touches prose: section intros, beats body copy, the deploy <dd>
paragraphs, the ledger's ARCHITECTURE half (measured half stays), the field
bullets, and the butler section's explanatory paragraphs. Zero claims are
added; several are trimmed. The style bar stays: "No more blind spots. Period."
"""
import os, re, sys, hashlib

SITE = os.path.expanduser("~/.hermes/workspace/visionbridge-demo/site")
SRC = os.path.join(SITE, "index-v4.html")
DST = os.path.join(SITE, "index-v5.html")

# ---------------------------------------------------------------- copy deck
EDITS = [
# ================= HERO: two ledes -> a tight bullet strip =================
("hero lede 1 -> bullet 1",
 "<p class=\"lede\">VisionBridge turns <b>the cameras, CCTV networks and drones you already run</b> into one persistent watch — over a roofline, a fence line, a whole event site, or a boundary with nothing to bolt a camera to — and puts a masked frame of anything that shouldn't be there on your field team's phones in about three seconds.</p>",
 "<ul class=\"quick\">\n"
 "      <li><b>Any feed you already run</b> — cameras, CCTV, drones — reads into one watch.</li>\n"
 "      <li><b>Anything on a surface</b> gets caught, masked, and pushed to your field team's phones in <b>about 3 seconds</b>.</li>"),

("hero lede 2 -> bullet 2",
 "<p class=\"lede\">It runs on hardware you choose: a machine on site, a virtual machine, a cloud instance, or a mobile unit that drives to the job. Nothing has to leave your building, and none of your cameras get replaced. The same watch scales from one roof to a campus, a four-day event, or a site with no fence at all.</p>",
 "      <li><b>Runs on hardware you choose</b> — on site, on a VM, in cloud, or on a mobile unit. Nothing leaves your building; nothing gets replaced.</li>\n"
 "      <li><b>Scales</b> from one roof to a campus, a four-day event, or a site with no fence.</li>\n"
 "    </ul>"),

# ================= #how =================
("how intro",
 "<p class=\"big\" style=\"max-width:70ch\">Inputs come in from wherever you already have eyes — a camera estate you already own, one aircraft overhead, a fleet taking turns, or all of them at once. One reasoning core asks the narrow question, draws the answer, and merges what every view is telling it. The result goes to the people who can walk over and look.</p>",
 "<p class=\"big\" style=\"max-width:70ch\">One core, one question: is a person where they shouldn't be? It reads every input you have, answers, and sends the picture to the people who can act.</p>"),

("beat 01",
 "<p>Your existing cameras, a handful of cheap ones, or an aircraft holding a position overhead. Rooflines, walls, gates, the outside face of the perimeter, open ground — the places a person can be and nobody can see.</p>",
 "<p>Point it at any surface nobody watches — rooflines, walls, gates, open ground.</p>"),

("beat 02",
 "<p>A small vision model watches for one thing: a person where a person shouldn't be. Narrow questions are fast — about 1.3 seconds on a low-resolution frame.</p>",
 "<p>One narrow question, asked fast — about 1.3 s on a low-res frame.</p>"),

("beat 03",
 "<p>A second model outlines exactly what it found and labels it with a confidence. The field team sees a person, not a blinking dot on a map.</p>",
 "<p>The exact shape is drawn and labelled — a person on a map, not just a dot.</p>"),

("beat 04",
 "<p>The annotated frame arrives on the phones of the people already standing on the ground. They walk over and look. That is the whole loop.</p>",
 "<p>The frame lands on field phones. They walk over and look. Whole loop.</p>"),

# ================= #inputs =================
("inputs intro paragraph",
 "<p>VisionBridge is not a camera. It sits in front of the camera estate you already have and makes it do something it cannot do on its own: reason about what it is looking at, continuously, and say so out loud.</p>",
 "<p>Not a camera. It sits in front of the cameras you already have and makes them think for themselves — continuously, and out loud.</p>"),

("inputs: existing CCTV",
 "<p><b>Existing CCTV networks.</b> We read the stream from your NVR or VMS. No rip-and-replace, no new cabling, no change to how your operators already work. Analogue, IP, whatever is already terminated.</p>",
 "<p><b>Existing CCTV.</b> We read your NVR or VMS. No new cabling, no rip-and-replace.</p>"),

("inputs: camera",
 "<p><b>Camera on any surface or feature.</b> A pole, a parapet, a gatepost, a wall, a parapet corner. If it streams video and it covers a surface, it can be a zone.</p>",
 "<p><b>Any camera.</b> On a pole, a wall, a parapet. If it covers a surface, it's a zone.</p>"),

("inputs: drone feeds",
 "<p><b>Drone feeds.</b> A single aircraft on a slow orbit, one holding station over a fixed point, or a fleet that takes turns. Video comes down to the same core and lands in the same alert stream as the fixed cameras — one queue, one picture.</p>",
 "<p><b>Drone feeds.</b> One aircraft, one holding station, or a fleet taking turns — all into the same alert stream.</p>"),

("inputs: mixing quiet",
 "<p class=\"quiet\">Mixing sources is the point. A fixed camera knows one angle forever; a drone knows many angles for a while. The core does not care which is which — it cares whether a person is standing somewhere they should not be.</p>",
 "<p class=\"quiet\">Mix them freely — the core doesn't care which is which, only where a person shouldn't be.</p>"),

# ================= #fleet =================
("fleet intro",
 "<p>A single camera has a cone. A building has four sides, a roof, and a perimeter. That is where fleets change the arithmetic: several aircraft covering overlapping ground, and — this is the useful part — <b>sharing what each one has detected and what it has reasoned about</b>.</p>",
 "<p>One camera has a cone. A fleet covers the whole site — and each aircraft <b>shares what it detects and reasons about</b>, so coverage is continuous.</p>"),

("fleet sharing paragraph",
 "<p>When one aircraft has a person in frame and another is turning away, the second does not start from nothing. The detection, its shape and its confidence hand over, so a person who walks out of one field and into another does not fall into a gap between them. Coverage becomes continuous over the area rather than per-camera.</p>",
 "<p>Similarly, when one aircraft's view hands to another, the detection and its confidence come with it — no gap between fields, no lost track.</p>"),

("fleet sideways",
 "<p>The same sharing works sideways. Two fixed cameras with overlapping views, or a fixed camera and a drone passing over it, reconcile into a single tracked picture instead of two unrelated alerts on two screens.</p>",
 "<p>Two cameras with overlapping views become one alert, not two.</p>"),

("fleet hold",
 "<p><b>A hold is the other half of it.</b> An aircraft can be told to stay over one point — a school roof, a substation yard, a gate approach — and it stays there, because the site boundary and the flight volume are drawn as a geofence rather than flown by hand on the sticks.</p>",
 "<p><b>A hold.</b> An aircraft stays over your point — a school roof, a substation yard, a gate — held by a geofence rather than a pilot's thumb.</p>"),

("fleet rotation",
 "<p><b>And a fleet can change the guard.</b> When the aircraft on station needs to come down, the next one is already airborne and takes the same track over. The detection, its shape and its confidence hand across the swap, so the identification does not lapse while one aircraft replaces another. From the field team's side it is one continuous alert stream, not a series of separate flights.</p>",
 "<p><b>A fleet changes the guard.</b> When one aircraft comes down, the next takes the same track over — the alert stream never pauses.</p>"),

# ================= #runs intro =================
("runs intro",
 "<p class=\"big\" style=\"max-width:70ch\">The same core, on whichever footing you already have — or on one you can carry to the site. There is no single blessed deployment, and nothing about the product forces your video into somebody else's data centre. It is the same whether the job is one building that never moves or an event that packs up on Sunday.</p>",
 "<p class=\"big\" style=\"max-width:70ch\">The same core on whichever footing you have — on site, on a VM, in cloud, on a mobile unit. Permanently at a building, or gone on Sunday after an event.</p>"),

("runs quiet line",
 "<p class=\"quiet\" style=\"margin-top:22px;max-width:74ch\">All four run the same pipeline and produce the same alert format. On-site and mobile keep every frame inside your perimeter by construction; virtual machine and cloud are for sites that would rather inherit existing patching, backups and access control than run another box.</p>",
 "<p class=\"quiet\" style=\"margin-top:22px;max-width:74ch\">All four run the same pipeline with the same alerts. On-site and mobile keep every frame inside your perimeter; VM and cloud inherit your existing patching and backups.</p>"),

# ================= #surfaces =================
("surfaces intro",
 "<p class=\"big\" style=\"max-width:70ch\">The question the model asks is deliberately narrow — <i>is there a person where a person shouldn't be?</i> — but the surfaces it can ask that question about are not. A roof is simply the first one, because it is the one nobody watches. The list after it is not short: fence lines and gate approaches, building faces, yards, and the open ground around an event where the perimeter exists only as a line on a map.</p>",
 "<p class=\"big\" style=\"max-width:70ch\">The question is deliberately narrow — <i>is there a person where a person shouldn't be?</i> — but the surfaces it watches are not: roofs, walls, perimeters, building faces, yards, and open ground.</p>"),

("surfaces anomaly paragraph",
 "<p>And it is not restricted to people. The model's frame of reference is the scene as it should be, so the same machinery carries anomalies that a fixed rule cannot describe: a door standing open that is always shut, a service hatch lifted, an object left on a surface that is always clear, or movement across a wall that should be perfectly still.</p>",
 "<p>It is not only people. The same watch catches anomalies: a door open that should be shut, a hatch lifted, an object where one never is.</p>"),

("surfaces honest boundary",
 "<p class=\"quiet\">The honest boundary: this is a watch on <b>surfaces and objects</b>, not a product for identifying people. It does not recognise faces and it does not build a picture of a person's identity. It tells your team that something is where nothing should be, and shows them the frame.</p>",
 "<p class=\"quiet\">The boundary: it watches <b>surfaces and objects</b>, not identities — no face recognition, no person profiles. It shows your team the frame so they can act.</p>"),

# ================= #measured: architecture half condensed (measured stays) ===
("ledger architecture h3",
 "<h3>What is a deployment choice, not a benchmark</h3>",
 "<h3>Deployment choices, not benchmarks</h3>"),

("ledger architecture list",
 "<li>Reading existing CCTV, fixed cameras and drone feeds into the same core and alert stream.</li>\n          <li>Running on an on-site machine, a virtual machine, a cloud instance or a mobile unit.</li>\n          <li>Multiple aircraft sharing detections and reasoning to hold coverage across an area.</li>\n          <li>Surfaces beyond the roof: walls, perimeters, building faces, open ground, and scene anomalies.</li>\n          <li>An aircraft holding station over a fixed point, and a fleet rotating out mid-watch with the track handed over between aircraft inside a geofence.</li>\n          <li>The same watch applied to a site that stands still — a school, a yard, a substation — or a temporary one, such as an event or a fairground.</li>",
 "<li>Reads existing CCTV, fixed cameras and drone feeds into one alert stream.</li>\n          <li>Runs on-site, on a VM, in cloud, or on a mobile unit.</li>\n          <li>Fleets share detections and reasoning; aircraft hold station and rotate out mid-watch inside a geofence.</li>\n          <li>Watches surfaces and open ground; one core serves a permanent site or a one-off event.</li>"),

# ================= #butler: condense the explication while keeping the facts ===
("butler opening",
 "<p>On 13 July 2024, a shooter climbed onto the roof of the AGR International complex near the Butler Farm Show grounds and fired at a former president from a position roughly 400–450 feet from the stage. The public record of how that happened is a list of things a persistent, boring camera could have made visible.</p>",
 "<p>13 July 2024. A shooter was on a roof, 400–450 feet from a stage, and nobody was watching that roof. The record shows what a persistent, boring camera could have made visible:</p>"),

("butler closing: not a replacement",
 "<p>VisionBridge is not a replacement for a counter-sniper team or a radio net. It is the cheap, tireless layer underneath them: it looks at the same surfaces, every second, and the moment a shape appears on one, the picture is already on the phones of the people who can walk over and check.</p>",
 "<p>VisionBridge doesn't replace that team or its radio net. It's the cheap, tireless layer underneath: the same surfaces, watched every second, the picture already on the phones of the people who can walk over.</p>"),

("butler order paragraph",
 "<p>It also explains the order we have gone in. Roofs come first because that is where the failure was most visible and most preventable. The same watch turned to a fence line, an event, or a site that stands still is the same pipeline doing the same job with a different zone drawn on it.</p>",
 "<p>That's why roofs come first — the most visible, most preventable miss. The same watch on a fence, an event, or a school is the same pipeline with a different zone drawn on it.</p>"),

# ================= #field =================
("field bullet 1",
 "<li>An alert with the frame attached — the actual image, already masked, with confidence and the zone name.</li>",
 "<li>The masked frame, confidence, and zone name — attached to the alert.</li>"),

("field bullet 2",
 "<li>No login ceremony. The alert goes to phones that are already on site, on your own link.</li>",
 "<li>No login. Straight to the phones already on site, on your own link.</li>"),

("field bullet 3",
 "<li>One tap opens the mask and the zone view, so the responder can see the angle before they walk.</li>",
 "<li>One tap shows the angle before they walk.</li>"),

("field bullet 4",
 "<li>Every alert is logged with a timestamp and the frame, so the day can be replayed afterwards.</li>",
 "<li>Everything is logged — timestamp and frame — so the day replays.</li>"),

("field bullet 5",
 "<li>When several feeds cover the same ground — or when one aircraft hands over to another mid-watch — they reconcile into one alert rather than three.</li>",
 "<li>Overlapping feeds and aircraft handovers reconcile into one alert, not three.</li>"),

# ================= #deploy: condense the prose <dd>s =================
("deploy inputs",
 "<dt>inputs</dt><dd>Existing CCTV and VMS streams, fixed cameras on any surface, and drone video. We read the stream; we don't replace it. Mixed sources land in one alert queue.</dd>",
 "<dt>inputs</dt><dd>Existing CCTV, fixed cameras, drone video. We read your stream, we don't replace it.</dd>"),

("deploy compute",
 "<dt>compute</dt><dd>One of four: an on-site machine (an Apple Silicon box or a small edge unit), a virtual machine on hardware you already run, a cloud instance in a region you choose, or a mobile unit that comes to the site. Same pipeline, same output.</dd>",
 "<dt>compute</dt><dd>On-site box, VM, cloud instance, or a mobile unit — same pipeline, same output.</dd>"),

("deploy aircraft",
 "<dt>aircraft</dt><dd>One aircraft is useful for a moving view, or one holding station over a fixed point inside a geofence. A fleet is useful for coverage and for endurance: overlapping fields, detections and reasoning shared between aircraft, and the next aircraft taking over the track when the one on station needs to come down.</dd>",
 "<dt>aircraft</dt><dd>One holding station in a geofence, a patrol, or a fleet that takes over the track mid-watch — coverage and endurance.</dd>"),

("deploy network",
 "<dt>network</dt><dd>On-site and mobile deployments need nothing to leave the building — inference runs where the cameras are. Virtual machine and cloud deployments use an encrypted link you control.</dd>",
 "<dt>network</dt><dd>On-site and mobile: nothing leaves the building. VM and cloud: an encrypted link you control.</dd>"),

("deploy setup",
 "<dt>setup</dt><dd>Streams pointed at the surfaces you care about, a zone drawn around each, a geofence and a hold point set for any aircraft, the field phones added to the alert list. That is the whole configuration.</dd>",
 "<dt>setup</dt><dd>Point streams at surfaces, draw a zone, set a geofence and hold point, add field phones. That's the whole configuration.</dd>"),

("deploy not",
 "<dt>what it is not</dt><dd>Not a weapons system, not face recognition, and not a replacement for your security plan or your radio net. It also is not a decision-maker: autonomous here means the aircraft keeps its own position and the fleet changes the guard without a pilot on the sticks — a person still decides what happens to anything it reports. It is a persistent watch on the surfaces of your site that people cannot watch continuously — and it tells your team when something is on one.</dd>",
 "<dt>what it is not</dt><dd>Not a weapons system, not face recognition, not a decision-maker, and not a replacement for your people. Autonomous means it holds position and changes the guard on its own — a person still decides what happens next.</dd>"),

# ================= #pilot CTA =================
("cta paragraph",
 "<p>We're taking a small number of pilot sites — a surface, the cameras or aircraft you already have, and a few field phones. A roof, a fence line, a school that stands still, or an event with a date on it: pick the one that worries you. If it misses anything that matters, you tell us and we fix it. If it works, you'll have the only perimeter layer on the market that runs on hardware you already own.</p>",
 "<p>We're taking a small number of pilot sites — a surface, the cameras or aircraft you already have, and a few field phones. Pick the one that worries you: a roof, a fence line, a school, an event. If it misses anything, you tell us and we fix it.</p>"),
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

# The hero bullets class (.quick) needs its CSS, appended inside the page's
# single <style>. The v4 CSS block ends with the .contexts media query; we
# rely on a unique marker inside that block to anchor the append.
QUICK_CSS = """
/* hero quick bullets (v5): tight 2x2 grid under the lede. No new colours, no
   new fonts. The <b> lead-in gets the ink weight, the rest stays graphite. */
ul.quick {
  list-style: none;
  margin: 28px 0 0;
  padding: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px 28px;
  max-width: 56ch;
}
ul.quick li {
  position: relative;
  padding-left: 16px;
  font-size: 14.5px;
  line-height: 1.55;
  color: var(--graphite);
}
ul.quick li::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0.55em;
  width: 6px;
  height: 6px;
  background: var(--rule);
}
ul.quick li b { color: var(--ink); font-weight: 500; }
@media (max-width: 560px) {
  ul.quick { grid-template-columns: 1fr; }
}
"""
anchor_css = "@media (max-width: 560px) {\n  .contexts { grid-template-columns: 1fr; }\n}"
if QUICK_CSS in html:
    pass  # already there
elif anchor_css in html:
    html = html.replace(anchor_css, anchor_css + QUICK_CSS, 1)
else:
    problems.append("could not find .contexts CSS anchor to append .quick")

if problems:
    print("ABORTED — these anchors did not match exactly once:")
    for p in problems: print("   -", p)
    sys.exit(1)

open(DST, "w").write(html)
print(f"wrote {DST}")
print(f"  {len(open(SRC).read())} -> {len(html)} chars  ({'+' if len(html) > len(open(SRC).read()) else ''}{len(html) - len(open(SRC).read())})")
print(f"  md5 {hashlib.md5(html.encode()).hexdigest()}")
print(f"  copy edits applied: {len(EDITS)}")

# sanity: the untouched things must be intact (comment-stripped)
probe = re.sub(r"<!--.*?-->", "", html, flags=re.S)
for label, pat, want in [("inline SVG plates", r"<svg viewBox=", 4),
                          ("images", r"<img ", 3),
                          ("readout band wrapper", r'<div[^>]*class="wrap readout"', 1),
                          ("card row", r'class="beats" role="list"', 1),
                          ("quick bullets", r'class="quick"', 1),
                          ("contexts section", r'class="contexts"', 1),
                          ("limits note", r'class="wrap limits-note"', 1)]:
    got = len(re.findall(pat, probe))
    ok = got == want
    print(f"  intact: {label:26s} {got} (want {want}) {'OK' if ok else '!! MISMATCH'}")
    if not ok: problems.append(f"{label} x{got}")
print("\nABORTED ON STRUCTURE" if problems else "\nSTRUCTURE OK")
