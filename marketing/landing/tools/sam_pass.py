#!/usr/bin/env python
"""SAM 3.1 masking + Falcon low-res fastscan timings on the demo scenes."""
import sys, os, json, time
import numpy as np
os.chdir("/Users/scrimwiggins/VisionBrain")
sys.path.insert(0, "/Users/scrimwiggins/VisionBrain/src")
from PIL import Image
from visionbrain.fp_inference import detect
from visionbrain.sam3_inference import detect_multi

SC = "/Users/scrimwiggins/.hermes/workspace/visionbridge-demo/scenes"
OUT = "/Users/scrimwiggins/.hermes/workspace/visionbridge-demo/out"
os.makedirs(OUT, exist_ok=True)

results = json.load(open(f"{OUT}/falcon_results.json"))

for name in ["A-day-roof-field", "C-close-roof"]:
    img = Image.open(f"{SC}/{name}.png").convert("RGB")

    # --- Falcon low-res fastscan (the tripwire path) ---
    t0 = time.perf_counter(); _, _ = detect(img, "person", min_dimension=360, max_dimension=640); lo_cold = (time.perf_counter()-t0)*1000
    t1 = time.perf_counter(); lo, _ = detect(img, "person", min_dimension=360, max_dimension=640); lo_warm = (time.perf_counter()-t1)*1000
    results[name]["falcon_lowres_detect"] = {"cold_ms": round(lo_cold), "warm_ms": round(lo_warm), "n_warm": len(lo),
                                             "items": [o.to_dict() for o in lo]}
    print(f"{name} falcon lowres detect: cold {lo_cold:.0f}ms | warm {lo_warm:.0f}ms ({len(lo)} hits)", flush=True)

    # --- SAM 3.1: bbox detect, then full segmentation ---
    t2 = time.perf_counter(); d_cold = detect_multi(img, ["person"], task="detect"); s_cold_ms = (time.perf_counter()-t2)*1000
    t3 = time.perf_counter(); d = detect_multi(img, ["person"], task="detect"); d_warm = (time.perf_counter()-t3)*1000
    t4 = time.perf_counter(); s_cold = detect_multi(img, ["person"], task="segment"); seg_cold = (time.perf_counter()-t4)*1000
    t5 = time.perf_counter(); s = detect_multi(img, ["person"], task="segment"); seg_warm = (time.perf_counter()-t5)*1000

    results[name]["sam31"] = {
        "detect_cold_ms": round(s_cold_ms), "detect_warm_ms": round(d_warm),
        "segment_cold_ms": round(seg_cold), "segment_warm_ms": round(seg_warm),
        "n": len(s),
        "items": [x.to_dict() for x in s],
    }
    print(f"{name} SAM3.1 detect: cold {s_cold_ms:.0f}ms | warm {d_warm:.0f}ms ({len(d)} hits)", flush=True)
    print(f"{name} SAM3.1 segment: cold {seg_cold:.0f}ms | warm {seg_warm:.0f}ms ({len(s)} hits)", flush=True)

    # --- save real masks for compositing ---
    for i, det in enumerate(s):
        if det.mask is not None:
            m = (np.asarray(det.mask) > 0).astype(np.uint8) * 255
            Image.fromarray(m).save(f"{OUT}/{name}.sam_mask{i}.png")
            print(f"  saved mask {i}: bbox={[round(v) for v in det.bbox_xyxy]} score={det.score:.3f}", flush=True)

json.dump(results, open(f"{OUT}/falcon_results.json", "w"), indent=1)
print("WROTE results")
