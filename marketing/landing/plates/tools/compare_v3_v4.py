#!/usr/bin/env python3
"""
compare_v3_v4.py — prove the v4 rebuild kept every visual identical.

Extracts the four inline SVG plates and the three <img> tags from both files and
compares hashes. Copy can change; the plates and the images must not.
"""
import os, re, hashlib

SITE = os.path.expanduser("~/.hermes/workspace/visionbridge-demo/site")
v3 = open(os.path.join(SITE, "index-v3.html")).read()
v4 = open(os.path.join(SITE, "index-v4.html")).read()

def sha(s): return hashlib.sha256(s.encode()).hexdigest()[:16]

p3 = re.findall(r"<svg\b.*?</svg>", v3, re.S)
p4 = re.findall(r"<svg\b.*?</svg>", v4, re.S)
print(f"plates: v3 {len(p3)}   v4 {len(p4)}")
names = ["plate1 pipeline", "plate2 fleet", "plate3 modes", "plate4 elevation"]
allok = True
for i, (a, b) in enumerate(zip(p3, p4)):
    ok = sha(a) == sha(b)
    allok &= ok
    print(f"  {names[i]:20s} v3 {sha(a)}  v4 {sha(b)}  {'IDENTICAL' if ok else '!! CHANGED'}")

print("\nimages:")
i3 = re.findall(r'<img [^>]*>', v3)
i4 = re.findall(r'<img [^>]*>', v4)
for a, b in zip(i3, i4):
    ok = sha(a) == sha(b)
    allok &= ok
    src = re.search(r'src="([^"]+)"', a).group(1)
    print(f"  {src:34s} {'IDENTICAL' if ok else '!! CHANGED'}")
print(f"  counts v3 {len(i3)} / v4 {len(i4)}")

# every referenced asset must exist on disk
print("\nassets resolve on disk:")
for src in set(re.findall(r'src="([^"]+)"', v4)):
    if src.startswith(("http", "data:")): continue
    p = os.path.join(SITE, src)
    exists = os.path.isfile(p)
    allok &= exists
    print(f"  {src:34s} {'OK' if exists else '!! MISSING'}")

print("\nsections:")
# scan the real markup: documentation comments in the fragments quote tag names,
# and a regex over raw text counts those as structure (they are not).
v3c = re.sub(r"<!--.*?-->", "", v3, flags=re.S)
v4c = re.sub(r"<!--.*?-->", "", v4, flags=re.S)
s3 = re.findall(r'<section[^>]*id="([^"]+)"', v3c)
s4 = re.findall(r'<section[^>]*id="([^"]+)"', v4c)
print(f"  v3: {s3}")
print(f"  v4: {s4}")
print(f"  added: {[s for s in s4 if s not in s3]}   removed: {[s for s in s3 if s not in s4]}")

print("\nstructure (comments stripped):")
for probe, expect in [("<svg viewBox=", 4), ('class="contexts"', 1),
                      ("assets/01-hero-locked-person.jpg", 1), ("assets/02-wide-magnified.jpg", 1),
                      ("assets/03-field-alert.jpg", 1),
                      ('<div class="wrap readout">', 1),
                      ('class="beats" role="list"', 1),
                      ('class="ledger"', 1)]:
    got = v4c.count(probe)
    ok = got == expect
    allok &= ok
    print(f"  {probe:38s} {got} (want {expect}) {'OK' if ok else '!!'}")

# the measured table must be untouched (it is the page's honesty anchor)
t3 = re.search(r'<div class="measure">.*?</table>', v3, re.S).group(0)
t4 = re.search(r'<div class="measure">.*?</table>', v4, re.S).group(0)
same = sha(t3) == sha(t4)
allok &= same
print(f"\nmeasured table: {'IDENTICAL (untouched)' if same else '!! CHANGED — the numbers must not move'}")

# and the measured half of the ledger
lg3 = re.search(r'<span class="tag">measured on this machine.*?</div>', v3, re.S).group(0)
lg4 = re.search(r'<span class="tag">measured on this machine.*?</div>', v4, re.S).group(0)
same2 = sha(lg3) == sha(lg4)
allok &= same2
print(f"ledger measured half: {'IDENTICAL (untouched)' if same2 else '!! CHANGED'}")

print(f"\nVERDICT: {'all visuals and the measured evidence are intact' if allok else 'PROBLEMS ABOVE'}")
