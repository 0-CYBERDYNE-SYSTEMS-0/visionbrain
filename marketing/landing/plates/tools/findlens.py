#!/usr/bin/env python3
"""
findlens.py — locate the lens in the RENDERED pixels instead of in my arithmetic, then
sample proper before/after points. Ground truth comes from the image, not from a model of it.
"""
import os, subprocess
from PIL import Image

P = os.path.expanduser("~/visionbridge-plates")
SCALE = 2
BEFORE = os.path.join(P, "review", "plate2-fleet-2x.png")
AFTER = os.path.join(P, "review", "plate2-AFTER-2x.png")

def near(p, c, t=6):
    return all(abs(p[i] - c[i]) <= t for i in range(3))

LENS = (200, 208, 214)     # #c8d0d6 rule tone
BLDG = (228, 233, 236)     # #e4e9ec panel/building fill
PAPER = (238, 241, 243)    # #eef1f3
GRAPHITE = (91, 101, 112)  # #5b6570

a = Image.open(AFTER).convert("RGB")
b = Image.open(BEFORE).convert("RGB")

# --- where are the lens-coloured pixels in the AFTER render? ---
xs, ys, n = [], [], 0
W, H = a.size
for y in range(0, H, 2):
    for x in range(0, W, 2):
        if near(a.getpixel((x, y)), LENS, 4):
            xs.append(x / SCALE); ys.append(y / SCALE); n += 1
if n:
    print(f"lens-coloured pixels found: {n} samples (every 2nd px)")
    print(f"  user-unit bbox x[{min(xs):.1f}..{max(xs):.1f}] y[{min(ys):.1f}..{max(ys):.1f}]"
          f"  -> {max(xs)-min(xs):.1f} x {max(ys)-min(ys):.1f} units")
    # column histogram: find the widest row to pick a sample point safely inside
    from collections import Counter
    col = Counter(round(x) for x in xs)
    print(f"  widest columns: {col.most_common(5)}")
else:
    print("NO lens-coloured pixels found — the fill change did not render")

# --- is the SAME pixel now different from the building? ---
def probe(name, x, y):
    bp = b.getpixel((int(x * SCALE), int(y * SCALE)))
    ap = a.getpixel((int(x * SCALE), int(y * SCALE)))
    print(f"  {name:34s} ({x:4.0f},{y:4.0f})  BEFORE {bp}  AFTER {ap}  "
          f"{'CHANGED' if bp != ap else 'same'}")
    return bp, ap

if n:
    midy = (min(ys) + max(ys)) / 2
    # a point guaranteed inside the lens: median x of the widest row
    widest_row_y = max(set(round(y) for y in ys), key=lambda yy: sum(1 for y in ys if round(y) == yy))
    row_xs = sorted(x for x, y in zip(xs, ys) if round(y) == widest_row_y)
    midx = row_xs[len(row_xs) // 2]
    print(f"\nwidest lens row y={widest_row_y}, spans x {row_xs[0]:.0f}..{row_xs[-1]:.0f}")
    print("\n--- before/after at points derived from the render ---")
    probe("LENS over either ground", midx, widest_row_y)
    probe("LENS upper tip", row_xs[len(row_xs)//2], min(ys) + (max(ys)-min(ys))*0.08)
    probe("building, well clear of lens", 600, 250)
    probe("paper, well clear of lens", 150, 60)
