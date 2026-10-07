#!/usr/bin/env python
"""Real Falcon Perception masking pass on the demo scenes — timings + masks."""
import sys, os, json, time
os.chdir("/Users/scrimwiggins/VisionBrain")
sys.path.insert(0, "/Users/scrimwiggins/VisionBrain/src")
from PIL import Image
from visionbrain.fp_inference import detect, segment

SC = "/Users/scrimwiggins/.hermes/workspace/visionbridge-demo/scenes"
OUT = "/Users/scrimwiggins/.hermes/workspace/visionbridge-demo/out"
os.makedirs(OUT, exist_ok=True)

results = {}
for name in ["A-day-roof-field", "C-close-roof"]:
    img = Image.open(f"{SC}/{name}.png").convert("RGB")
    entry = {"size": img.size}
    for label, fn in (("detect", detect), ("segment", segment)):
        t0 = time.perf_counter()
        out_cold, stats_cold = fn(img, "person")
        cold_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        out, stats = fn(img, "person")
        warm_ms = (time.perf_counter() - t1) * 1000

        entry[label] = {
            "cold_ms": round(cold_ms),
            "warm_ms": round(warm_ms),
            "n_cold": len(out_cold),
            "n_warm": len(out),
            "items": [o.to_dict() for o in out],
            "stats": {"preprocess_ms": round(getattr(stats, "preprocess_ms", 0), 1),
                      "generation_ms": round(getattr(stats, "generation_ms", 0), 1)},
        }
        if label == "segment":
            entry["segment_rles"] = [o.rle for o in out]
            entry["segment_areas"] = [o.area_fraction for o in out]
        print(f"{name} {label}: cold {cold_ms:.0f}ms ({len(out_cold)} hits) | warm {warm_ms:.0f}ms ({len(out)} hits)", flush=True)
    results[name] = entry

json.dump(results, open(f"{OUT}/falcon_results.json", "w"), indent=1)
print("WROTE", f"{OUT}/falcon_results.json")
