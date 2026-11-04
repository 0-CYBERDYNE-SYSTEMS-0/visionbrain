#!/usr/bin/env python3
"""
build-v3.py — assemble site/index-v3.html from the v3 plate fragments.

Non-destructive: reads index-v2.html, writes index-v3.html. Never modifies
index.html or index-v2.html. Only substitutes components whose v3 file exists,
so it can be run incrementally as specialists land.

    python3 build-v3.py            # assemble and report
    python3 build-v3.py --dry-run  # report only, write nothing
"""
import os, re, sys, hashlib, collections

SITE = "/Users/scrimwiggins/.hermes/workspace/visionbridge-demo/site"
V3   = "/Users/scrimwiggins/visionbridge-plates/v3"
SRC  = os.path.join(SITE, "index-v2.html")
DST  = os.path.join(SITE, "index-v3.html")
DRY  = "--dry-run" in sys.argv

PLATES = [  # document order in index-v2.html
    ("plate1-pipeline",  "P1 pipeline / how it works"),
    ("plate2-fleet",     "P2 drone fleet plan view"),
    ("plate3-modes",     "P3 the four deployment modes"),
    ("plate4-elevation", "P4 surfaces elevation"),
]

def read(p):
    try:
        with open(p) as f: return f.read()
    except FileNotFoundError:
        return None

html = read(SRC)
if html is None:
    sys.exit("FATAL: cannot read " + SRC)

log, subs = [], []

# ---- 1. the four inline SVG plates, in document order -------------------
blocks = list(re.finditer(r'<svg viewBox=.*?</svg>', html, re.S))
assert len(blocks) == 4, f"expected 4 SVG blocks in v2, found {len(blocks)}"
out, cursor = [], 0
for i, m in enumerate(blocks):
    name, desc = PLATES[i]
    out.append(html[cursor:m.start()])
    frag = read(os.path.join(V3, f"{name}.svg"))
    if frag:
        frag = frag.strip()
        if 'role="img"' not in frag or 'aria-label' not in frag:
            log.append(f"  !! {name}.svg is missing role=\"img\" or aria-label")
        out.append(frag)
        subs.append(f"{desc}  <- v3/{name}.svg  ({len(frag)} bytes, v2 was {m.end()-m.start()})")
    else:
        out.append(m.group(0))
        log.append(f"  -- {desc}: no v3 file yet, kept v2")
    cursor = m.end()
out.append(html[cursor:])
html = "".join(out)

# ---- 2. the numbered card row (.beats) ---------------------------------
beats_frag = read(os.path.join(V3, "beats-cards.html"))
mb = re.search(r'<div class="beats">.*?</div>\s*</section>', html, re.S)
if mb and beats_frag:
    f = beats_frag.strip()
    if f.rstrip().endswith("</section>"):
        replacement = f                      # fragment already carries the section close
    elif 'class="beats"' in f:
        replacement = f + "\n  </section>"    # fragment IS the beats row (any element)
    else:
        replacement = '<div class="beats">\n' + f + '\n    </div>\n  </section>'
    html = html[:mb.start()] + replacement + html[mb.end():]
    subs.append(f"card row (.beats)  <- v3/beats-cards.html  ({len(f)} bytes, v2 was {mb.end()-mb.start()})")
elif mb:
    log.append("  -- card row (.beats): no v3 file yet, kept v2")
else:
    log.append("  !! card row (.beats): anchor not found in v2")

# ---- 3. the readout band (.readout) ------------------------------------
ro_frag = read(os.path.join(V3, "readout-band.html"))
m_ro = re.search(r'<div class="wrap readout">.*?(?=<section class="wrap" id="how">)', html, re.S)
if m_ro and ro_frag:
    f = ro_frag.strip()
    # Detect an existing wrapper STRUCTURALLY, not by a literal class string.
    # A previous version tested `'class="readout"'`, which does NOT occur in
    # `class="wrap readout"` — so the guard never matched, the band was wrapped a
    # second time, and two nested grid containers shipped inside a build that had
    # passed every gate. Nothing caught it because nesting creates no id clash and
    # no text collision.
    if not re.search(r'<div[^>]*class="[^"]*\breadout\b', f):
        f = '<div class="wrap readout">\n' + f + '\n  </div>'
    html = html[:m_ro.start()] + f + "\n\n  " + html[m_ro.end():]
    subs.append(f"readout band  <- v3/readout-band.html  ({len(f)} bytes, v2 was {m_ro.end()-m_ro.start()})")
elif m_ro:
    log.append("  -- readout band: no v3 file yet, kept v2")
else:
    log.append("  !! readout band: anchor not found in v2")

# ---- 3b. structural assertions: each substituted component must appear ONCE -
# (guards against the double-wrap class of bug above; scans comment-stripped markup
#  because the fragments' documentation comments quote tag names)
_probe = re.sub(r'<!--.*?-->', '', html, flags=re.S)
for label, pat, want in [
    ("card row (.beats)",        r'class="beats"',                       1),
    ("readout band wrapper",     r'<div[^>]*class="wrap readout"',       1),
    ("readout rows",             r'<div[^>]*class="[^"]*\bro-measured\b', 2),
    ("inline SVG plates",        r'<svg viewBox=',                       4),
    ("images",                   r'<img ',                               3),
    ("diagram containers",       r'<div class="diagram">',               4),
]:
    c = len(re.findall(pat, _probe))
    if c != want:
        log.append(f"  !! STRUCTURE: {label} appears {c}x, expected {want}")
    else:
        log.append(f"  ok structure: {label} x{c}")

# ---- 4. the v3 CSS, appended to the page's single <style> block --------
beats_css = read(os.path.join(V3, "beats.css"))
diag_css  = read(os.path.join(V3, "diagrams.css"))
parts = [c.strip() for c in (beats_css, diag_css) if c]
if parts and '</style>' in html:
    css = "\n\n".join(parts)
    if '/* v3 */' in html:
        html = re.sub(r'/\* v3 \*/.*?(?=</style>)', '/* v3 */\n' + css + '\n', html, flags=re.S)
    else:
        html = html.replace('</style>', '\n  /* v3 */\n' + css + '\n</style>', 1)
    subs.append(f"CSS  <- v3/beats.css + v3/diagrams.css  ({len(css)} bytes appended under a /* v3 */ marker)")
else:
    log.append("  -- v3 CSS: nothing to append yet, page CSS untouched")

# ---- 5. whole-page integrity checks (these are the real merge risks) ---
# scan a comment-stripped copy: an id or url(#x) mentioned inside a documentation
# comment is not real markup, and flagging it would be a false positive.
scan = re.sub(r'<!--.*?-->', '', html, flags=re.S)
ids = re.findall(r'\sid="([^"]+)"', scan)
dups = [k for k, v in collections.Counter(ids).items() if v > 1]
svgs = len(re.findall(r'<svg viewBox=', scan))
roles = len(re.findall(r'<svg[^>]*role="img"', scan))
cls = sorted(set(re.findall(r'class="(svg-[a-z]+|pulse|reveal)"', scan)))
markers = sorted(set(re.findall(r'<marker id="([^"]+)"', scan)))
marker_refs = sorted(set(re.findall(r'url\(#([^)]+)\)', scan)))
dangling = [r for r in marker_refs if r not in markers]

print("=" * 72)
print("ASSEMBLY" + (" (dry run)" if DRY else ""))
print("=" * 72)
for s in subs: print("  ok  " + s)
for l in log: print(l)
print()
print(f"  svg plates in output : {svgs}  (role=img: {roles})")
print(f"  chars                : {len(html)}  (v2 was {len(read(SRC))})")
print(f"  text classes in use  : {cls}")
print(f"  marker ids           : {markers}")
print(f"  marker references    : {marker_refs}")
print(f"  DANGLING marker refs : {dangling or 'none'}")
print(f"  DUPLICATE page ids   : {dups or 'none'}")

ok = not dups and not dangling and roles == svgs
print()
print("  VERDICT:", "integrable" if ok else "NOT integrable — fix the above before shipping")

if not DRY:
    with open(DST, "w") as f: f.write(html)
    print(f"  wrote {DST}")
    print(f"  md5   {hashlib.md5(html.encode()).hexdigest()}")
