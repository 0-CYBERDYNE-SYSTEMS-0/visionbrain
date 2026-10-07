#!/bin/bash
# Run the uniform plate gate over a directory of SVG fragments.
# usage: ./runall.sh <dir-with-svgs> <out-dir> [viewport]
DIR="${1:-../baseline}"
OUT="${2:-../shots/gate}"
W="${3:-1440}"
mkdir -p "$OUT"
cd "$(dirname "$0")" || exit 1
for f in "$DIR"/*.svg; do
  b=$(basename "$f" .svg)
  node measure.mjs "$f" --width "$W" > "$OUT/$b.json" 2>&1
  python3 - "$OUT/$b.json" "$b" <<'PY'
import json, sys
p, name = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(p))
except Exception as e:
    print(f"{name:22s} PARSE-FAIL ({e})"); print(open(p).read()[:600]); raise SystemExit
h = d["HARD"]
bad = [k for k, v in h.items() if v]
print(f"{name:22s} PASS={str(d['PASS']):5s} texts={d['textCount']:3d} collisions={h['textCollisions']} "
      f"outOfFrame={h['outOfFrame']} contrastFails={h['contrastFails']} "
      f"contrastMin={d['contrastMin']} classes={h['classesNotInPage']} dupIds={h['dupIds']} markers={h['missingMarkers']}")
for c in d["collisions"][:8]:
    print(f"    COLLIDE  {c['a']!r} x {c['b']!r} ov={c['overlapUU']} area={c['area']}")
for o in d["outOfFrame"][:8]:
    print(f"    OUT      <{o['tag']}> {o['label']!r} {o['bbox']}")
for c in d["contrastFails"][:10]:
    print(f"    CONTRAST {c['ratio']} : {c['fill']} {c['size']} {c['t']!r}")
if d.get("missingMarkers"):
    print("    MISSING MARKERS", d["missingMarkers"])
if d.get("classesNotInPage"):
    print("    EXTRA CLASSES", d["classesNotInPage"])
PY
done
