#!/usr/bin/env python3
"""extract_copy.py — dump the current copy of index-v3.html with precise anchors.

Prints each section's id, its h2, and every prose node (p / dt / dd / li / figcaption)
with a short index, so v4 edits can be written against exact strings instead of
rewriting whole sections and putting the four SVG plates at risk.
"""
import os, re, html as ht

P = os.path.expanduser("~/.hermes/workspace/visionbridge-demo/site")
src = open(os.path.join(P, "index-v3.html")).read()

def clean(s):
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", ht.unescape(s)).strip()

print("TITLE:", clean(re.search(r"<title>(.*?)</title>", src, re.S).group(1)))
m = re.search(r'<meta name="description" content="([^"]*)"', src)
print("META :", m.group(1)[:160], "…\n")

# sections, in document order
for sm in re.finditer(r'<section([^>]*)>(.*?)</section>', src, re.S):
    attrs, body = sm.group(1), sm.group(2)
    sid = (re.search(r'id="([^"]+)"', attrs) or [None, "hero(no id)"])[1]
    h2 = re.search(r"<h2>(.*?)</h2>", body, re.S)
    print("=" * 78)
    print(f"SECTION #{sid}" + (f"   h2: {clean(h2.group(1))}" if h2 else "   (no h2)"))
    print("=" * 78)
    n = 0
    for pm in re.finditer(r"<(p|li|dt|dd|figcaption|h3)\b([^>]*)>(.*?)</\1>", body, re.S):
        tag, a, inner = pm.group(1), pm.group(2), clean(pm.group(3))
        if not inner:
            continue
        n += 1
        cls = (re.search(r'class="([^"]*)"', a) or [None, ""])[1]
        print(f"  [{n:02d}] <{tag}{' class=' + cls if cls else ''}> {inner[:230]}")
    print()

# the hero lives in a section without an id, plus the beats row outside any <p>
print("=" * 78); print("NAV + READOUT + BEATS + FOOTER"); print("=" * 78)
nav = re.search(r"<nav>(.*?)</nav>", src, re.S)
if nav:
    print("NAV:", " | ".join(re.findall(r">([^<]+)</a>", nav.group(1))))
ro = re.search(r'<div class="wrap readout">(.*?)\n  </div>', src, re.S)
if ro:
    for i, line in enumerate(re.findall(r"<div[^>]*>(.*?)</div>", ro.group(1), re.S), 1):
        print(f"READOUT[{i}]: {clean(line)[:150]}")
    for i, h in enumerate(re.findall(r'<p class="ro-[^"]*">(.*?)</p>', ro.group(1), re.S), 1):
        print(f"RO-HEAD[{i}]: {clean(h)}")
bt = re.search(r'<ol class="beats"[^>]*>(.*?)</ol>', src, re.S)
if bt:
    for i, c in enumerate(re.findall(r"<li>(.*?)</li>", bt.group(1), re.S), 1):
        print(f"BEAT[{i}]: {clean(c)[:160]}")
ft = re.search(r"<footer.*?</footer>", src, re.S)
if ft:
    print("FOOTER:", clean(ft.group(0))[:200])
