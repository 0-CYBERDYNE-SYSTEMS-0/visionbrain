#!/usr/bin/env python3
"""dump_raw.py — the editable prose of index-v3.html with tags intact,
the four SVG plates and <style> replaced by placeholders so they cannot be
mistaken for editable copy."""
import os, re

P = os.path.expanduser("~/.hermes/workspace/visionbridge-demo/site")
src = open(os.path.join(P, "index-v3.html")).read()

src = re.sub(r"<style>.*?</style>", "<STYLE/>", src, flags=re.S)
src = re.sub(r"<svg\b.*?</svg>", "<SVG_PLATE/>", src, flags=re.S)

out = []
# head bits
out.append("########## HEAD ##########")
for pat in [r"<title>.*?</title>", r'<meta name="description"[^>]*>']:
    m = re.search(pat, src, re.S)
    out.append(m.group(0) if m else "(not found)")

out.append("\n########## HEADER / NAV ##########")
m = re.search(r"<header.*?</header>", src, re.S)
out.append(m.group(0) if m else "(not found)")

out.append("\n########## HERO ##########")
m = re.search(r'<section class="wrap bleed hero">.*?</section>', src, re.S)
out.append(m.group(0) if m else "(not found)")

out.append("\n########## READOUT ##########")
m = re.search(r'<div class="wrap readout">.*?\n  </div>', src, re.S)
out.append(m.group(0) if m else "(not found)")

for sid in ["how", "inputs", "fleet", "runs", "surfaces", "measured", "butler", "field", "deploy"]:
    out.append(f"\n########## SECTION #{sid} ##########")
    m = re.search(r'<section[^>]*id="' + sid + r'">.*?</section>', src, re.S)
    out.append(m.group(0) if m else "(not found)")

out.append("\n########## PILOT / CTA ##########")
m = re.search(r'<section class="cta" id="pilot">.*?</section>', src, re.S)
out.append(m.group(0) if m else "(not found)")

out.append("\n########## FOOTER ##########")
m = re.search(r"<footer.*?</footer>", src, re.S)
out.append(m.group(0) if m else "(not found)")

path = os.path.expanduser("~/visionbridge-plates/copy-v3-raw.txt")
open(path, "w").write("\n".join(out))
print(f"wrote {path}  ({os.path.getsize(path)} bytes)")
