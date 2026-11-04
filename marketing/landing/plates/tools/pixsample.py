#!/usr/bin/env python3
"""
pixsample.py — verify a fill/occlusion fix by sampling REAL rendered pixels, before vs after.

No vision model involved: renders the current plate, then compares named points against a
pre-fix render. Coordinates are given in plate user units and converted at 2x.

Proves, for plate 2:
  1. the overlap lens was the same colour as the building (invisible) and now is not;
  2. the person route was occluded by the opaque lens and is now continuous.
"""
import os, subprocess, json
from PIL import Image

P = os.path.expanduser("~/visionbridge-plates")
BEFORE = os.path.join(P, "review", "plate2-fleet-2x.png")          # rendered pre-fix
AFTER = os.path.join(P, "review", "plate2-AFTER-2x.png")
SCALE = 2

def render():
    r = subprocess.run(["node", os.path.join(P, "shot.mjs"), os.path.join(P, "v3", "plate2-fleet.svg"),
                        AFTER, "--scale", "2"], capture_output=True, text=True, cwd=P)
    assert r.returncode == 0, r.stderr[:400]

# name -> (user x, user y)
POINTS = {
    "paper only (no lens)":        (150, 60),
    "building only (no lens)":     (600, 250),
    "LENS over paper":             (407, 165),
    "LENS over building/roof":     (406, 240),
}
ROUTE_BOX = (396, 214, 414, 230)   # inside lens 1, strictly inside the detection rect outline

def px(im, x, y):
    return im.getpixel((int(x * SCALE), int(y * SCALE)))

def is_graphite(p):
    r, g, b = p[:3]
    return abs(r - 91) < 26 and abs(g - 101) < 26 and abs(b - 112) < 26

def lum(c):
    def f(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = [f(v) for v in c[:3]]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def ratio(a, b):
    la, lb = lum(a), lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)

render()
b_im = Image.open(BEFORE).convert("RGB")
a_im = Image.open(AFTER).convert("RGB")
print(f"before: {b_im.size}   after: {a_im.size}\n")

print(f"{'sample point':30s} {'BEFORE':18s} {'AFTER':18s} verdict")
print("-" * 92)
for name, (x, y) in POINTS.items():
    bp, ap = px(b_im, x, y), px(a_im, x, y)
    changed = bp != ap
    print(f"{name:30s} {str(bp):18s} {str(ap):18s} {'CHANGED' if changed else 'same'}")

b_lens = px(b_im, *POINTS["LENS over building/roof"])
b_bldg = px(b_im, *POINTS["building only (no lens)"])
a_lens = px(a_im, *POINTS["LENS over building/roof"])
a_bldg = px(a_im, *POINTS["building only (no lens)"])

print("\n--- defect 1: was the lens distinguishable from the building it crosses? ---")
print(f"  BEFORE  lens {b_lens} vs building {b_bldg}  -> identical: {b_lens == b_bldg}  "
      f"contrast {ratio(b_lens, b_bldg):.2f}:1")
print(f"  AFTER   lens {a_lens} vs building {a_bldg}  -> identical: {a_lens == a_bldg}  "
      f"contrast {ratio(a_lens, a_bldg):.2f}:1")
print(f"  lens over paper: BEFORE contrast {ratio(b_lens, (238,241,243)):.2f}:1  "
      f"AFTER {ratio(a_lens, (238,241,243)):.2f}:1")

print("\n--- defect 2: was the person route occluded inside the overlap lens? ---")
def count_graphite(im):
    n = 0
    for x in range(ROUTE_BOX[0], ROUTE_BOX[2] + 1):
        for y in range(ROUTE_BOX[1], ROUTE_BOX[3] + 1):
            for dx in range(SCALE):
                for dy in range(SCALE):
                    if is_graphite(im.getpixel((x * SCALE + dx, y * SCALE + dy))):
                        n += 1
    return n
bg, ag = count_graphite(b_im), count_graphite(a_im)
print(f"  graphite (route) pixels inside the lens, box {ROUTE_BOX}:")
print(f"    BEFORE {bg}   AFTER {ag}   -> route {'was OCCLUDED' if bg == 0 else 'was visible'} before, "
      f"{'now visible' if ag > 0 else 'still hidden'} after")
print(f"  ratio {ag}/{max(bg,1)} = {ag / max(bg,1):.1f}x more route ink inside the overlap")
