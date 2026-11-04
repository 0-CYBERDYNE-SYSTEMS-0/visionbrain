#!/usr/bin/env python3
"""
verify_lens.py — confirm the overlap-lens fix in the REAL plate render, from pixels.
Ground truth: exact flat-fill pixels of #c8d0d6, found in the render itself.
Compares against the pre-fix render (review/plate2-fleet-2x.png) where the lens had 0 area.
"""
import os, subprocess
from PIL import Image
from collections import Counter

P = os.path.expanduser("~/visionbridge-plates")
SCALE = 2
BEFORE = os.path.join(P, "review", "plate2-fleet-2x.png")   # pre-fix
AFTER = os.path.join(P, "review", "plate2-FIXED-2x.png")

r = subprocess.run(["node", os.path.join(P, "shot.mjs"), os.path.join(P, "v3", "plate2-fleet.svg"),
                    AFTER, "--scale", "2"], capture_output=True, text=True, cwd=P)
assert r.returncode == 0, r.stderr[:500]

LENS = (200, 208, 214)      # #c8d0d6 exact
BLDG = (228, 233, 236)      # #e4e9ec
PAPER = (238, 241, 243)     # #eef1f3
GRAPH = (91, 101, 112)      # #5b6570

a = Image.open(AFTER).convert("RGB")
b = Image.open(BEFORE).convert("RGB")
W, H = a.size
print(f"render {W}x{H}px = {W//SCALE} x {H//SCALE} user units\n")

# ---- 1. exact flat-fill pixels -> the lens's true area and extent ----
pts = [(x, y) for y in range(H) for x in range(W) if a.getpixel((x, y)) == LENS]
print(f"exact #c8d0d6 pixels in the FIXED render : {len(pts)}")
if pts:
    ux = [p[0] / SCALE for p in pts]; uy = [p[1] / SCALE for p in pts]
    print(f"  area = {len(pts)/(SCALE*SCALE):.0f} u^2   (expected 1822 + 1388 = 3210 u^2)")
    # split into the two lenses by x
    for label, lo, hi in [("L1", 0, 550), ("L2", 550, 1120)]:
        cl = [p for p in pts if lo <= p[0] / SCALE < hi]
        if not cl: print(f"  {label}: none"); continue
        cx = [p[0] / SCALE for p in cl]; cy = [p[1] / SCALE for p in cl]
        print(f"  {label}: {len(cl)/(SCALE*SCALE):.0f} u^2  bbox x[{min(cx):.1f}..{max(cx):.1f}] "
              f"y[{min(cy):.1f}..{max(cy):.1f}]  w={max(cx)-min(cx):.1f} h={max(cy)-min(cy):.1f}")
else:
    print("  !! no flat fill found - lens still does not enclose area")

# ---- 2. was it visible before? (same coordinates, before vs after) ----
print("\n--- same pixel, before vs after ---")
def probe(name, x, y):
    bp = b.getpixel((int(x*SCALE), int(y*SCALE))); ap = a.getpixel((int(x*SCALE), int(y*SCALE)))
    print(f"  {name:32s} ({x:5.0f},{y:5.0f})  BEFORE {str(bp):18s} AFTER {str(ap):18s} "
          f"{'CHANGED' if bp != ap else 'same'}")
if pts:
    l1 = [p for p in pts if p[0] / SCALE < 550]
    cx = sorted(p[0] for p in l1)[len(l1)//2] / SCALE
    cy = sorted(p[1] for p in l1)[len(l1)//2] / SCALE
    probe("inside lens 1", cx, cy)
    probe("building, clear of lens", 600, 250)
    probe("paper, clear of lens", 150, 60)

# ---- 3. is the person route visible where it crosses the lens? ----
if pts:
    l1x = [p[0]/SCALE for p in pts if p[0]/SCALE < 550]
    l1y = [p[1]/SCALE for p in pts if p[0]/SCALE < 550]
    x0, x1 = min(l1x), max(l1x)
    y0 = max(l1y) - 22; y1 = max(l1y) - 4        # route band inside lens 1
    box = (int(x0*SCALE), int(y0*SCALE), int(x1*SCALE)+SCALE, int(y1*SCALE)+SCALE)
    def graph_count(im):
        n = 0
        for yy in range(box[1], box[3]):
            for xx in range(box[0], box[2]):
                px = im.getpixel((xx, yy))
                if all(abs(px[i]-GRAPH[i]) < 30 for i in range(3)): n += 1
        return n
    print(f"\n--- person route inside lens 1 (box {box} px) ---")
    print(f"  graphite pixels BEFORE {graph_count(b)}   AFTER {graph_count(a)}")
    print(f"  {'route now drawn over the lens' if graph_count(a) > graph_count(b) else 'no improvement'}")
