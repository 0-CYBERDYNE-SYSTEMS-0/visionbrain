#!/usr/bin/env python3
"""
do_final.py — the full validation chain after the fixes, one command.

  1. gate all four plates at 1440 / 768 / 390
  2. re-render the 2x review PNGs
  3. reassemble site/index-v3.html
  4. whole-page check at three widths against the served file
  5. slopscan design gate
"""
import os, subprocess, sys

P = os.path.expanduser("~/visionbridge-plates")
SITE = os.path.expanduser("~/.hermes/workspace/visionbridge-demo/site")
URL = "http://127.0.0.1:8899/index-v3.html"

def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=kw.get("cwd", P))

def head(title):
    print("\n" + "=" * 78); print(title); print("=" * 78)

# ---- 1. gates ----------------------------------------------------------------
for w in (1440, 768, 390):
    head(f"1. PLATE GATE @ {w}")
    r = run([os.path.join(P, "harness", "runall.sh"), os.path.join(P, "v3"),
             os.path.join(P, "shots", f"gate-v3-fixed-{w}"), str(w)])
    print(r.stdout.strip() or r.stderr.strip()[:400])
    if "PASS=False" in r.stdout:
        print("  !! a plate is failing")

# ---- 2. review renders -------------------------------------------------------
head("2. REVIEW RENDERS (2x, element-clipped)")
r = run(["python3", os.path.join(P, "render_review.py")])
print(r.stdout.strip() or r.stderr.strip()[:400])

# ---- 3. reassemble -----------------------------------------------------------
head("3. ASSEMBLE index-v3.html")
r = run(["python3", os.path.join(P, "build-v3.py")])
print(r.stdout.strip() or r.stderr.strip()[:400])

# ---- 4. whole page -----------------------------------------------------------
head("4. WHOLE-PAGE CHECK")
r = run(["node", os.path.join(P, "pagecheck.mjs"), URL])
print(r.stdout.strip() or r.stderr.strip()[:400])

# ---- 5. slopscan -------------------------------------------------------------
head("5. SLOPSCAN (auteur design gate)")
r = run(["node", os.path.expanduser("~/.hermes/skills/creative/auteur/scripts/slopscan.mjs"),
         os.path.join(SITE, "index-v3.html")])
print(r.stdout.strip()[-800:] or r.stderr.strip()[:400])
