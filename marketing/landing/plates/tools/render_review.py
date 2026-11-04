#!/usr/bin/env python3
"""
render_review.py — render each v3 plate to a tight, high-resolution review PNG.

No shell loops: one command, one script. Writes review/<plate>-2x.png for each plate
and prints the rendered pixel size so the output is auditable.
"""
import os, subprocess, sys, json

P = os.path.expanduser("~/visionbridge-plates")
REVIEW = os.path.join(P, "review")
os.makedirs(REVIEW, exist_ok=True)

PLATES = ["plate1-pipeline", "plate2-fleet", "plate3-modes", "plate4-elevation"]

for name in PLATES:
    svg = os.path.join(P, "v3", name + ".svg")
    png = os.path.join(REVIEW, name + "-2x.png")
    r = subprocess.run(
        ["node", os.path.join(P, "shot.mjs"), svg, png, "--scale", "2"],
        capture_output=True, text=True, cwd=P,
    )
    if r.returncode != 0:
        print(f"{name:22s} FAILED  {r.stdout.strip()[:200]} {r.stderr.strip()[:200]}")
        continue
    try:
        d = json.loads(r.stdout)
        print(f"{name:22s} {d['renderedPx']['w']}x{d['renderedPx']['h']}px  "
              f"viewBox {d['viewBox']}  -> {os.path.relpath(png, P)}  "
              f"({os.path.getsize(png)//1024} KB)")
    except Exception as e:
        print(f"{name:22s} unparsed output: {r.stdout.strip()[:200]} ({e})")
