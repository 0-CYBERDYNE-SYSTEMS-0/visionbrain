#!/usr/bin/env python3
"""
freeze_v3.py — freeze the approved v3 exactly, two ways:

  1. an immutable local snapshot with a SHA256 manifest  (_versions/v3-20260918/)
  2. a lean staging copy inside the VisionBrain repo      (marketing/landing/)

Deliberately EXCLUDED from both: shots/ and scenes/ (37 MB of QA screenshots and
source renders), and every untracked file elsewhere in the repo. Only the site,
its docs, the generators, the plate tooling and the measurement evidence go in.
"""
import os, shutil, hashlib, json, datetime

DEMO = os.path.expanduser("~/.hermes/workspace/visionbridge-demo")
PLATES = os.path.expanduser("~/visionbridge-plates")
SNAP = os.path.join(PLATES, "_versions", "v3-20260918")
REPO = "/Users/scrimwiggins/VisionBrain"
DEST = os.path.join(REPO, "marketing", "landing")

# (source, destination-relative) — the lean, meaningful set
FILES = [
    # ---- the site itself ----
    (f"{DEMO}/site/index.html", "site/index.html"),
    (f"{DEMO}/site/index-v2.html", "site/index-v2.html"),
    (f"{DEMO}/site/index-v3.html", "site/index-v3.html"),
    (f"{DEMO}/site/assets/01-hero-locked-person.jpg", "site/assets/01-hero-locked-person.jpg"),
    (f"{DEMO}/site/assets/02-wide-magnified.jpg", "site/assets/02-wide-magnified.jpg"),
    (f"{DEMO}/site/assets/03-field-alert.jpg", "site/assets/03-field-alert.jpg"),
    (f"{DEMO}/site/DESIGN-NOTES.md", "site/DESIGN-NOTES.md"),
    (f"{DEMO}/site/DESIGN-NOTES-v2.md", "site/DESIGN-NOTES-v2.md"),
    # ---- how the assets were made, and the measured evidence ----
    (f"{DEMO}/TODOS.md", "docs/TODOS.md"),
    (f"{DEMO}/GTM-targets.md", "docs/GTM-targets.md"),
    (f"{DEMO}/out/falcon_results.json", "docs/falcon_results.json"),
    (f"{DEMO}/sam_pass.py", "tools/sam_pass.py"),
    (f"{DEMO}/falcon_pass.py", "tools/falcon_pass.py"),
    (f"{DEMO}/composite.py", "tools/composite.py"),
    (f"{DEMO}/crop.py", "tools/crop.py"),
    (f"{DEMO}/check.py", "tools/check.py"),
    (f"{DEMO}/serve.py", "tools/serve.py"),
    # ---- the plate system: spec, evidence, harness, fragments ----
    (f"{PLATES}/SPEC.md", "plates/SPEC.md"),
    (f"{PLATES}/VERIFICATION.md", "plates/VERIFICATION.md"),
    (f"{PLATES}/VERIFICATION-PASS2.md", "plates/VERIFICATION-PASS2.md"),
    (f"{PLATES}/build-v3.py", "plates/build-v3.py"),
    (f"{PLATES}/harness/measure.mjs", "plates/harness/measure.mjs"),
    (f"{PLATES}/harness/runall.sh", "plates/harness/runall.sh"),
    (f"{PLATES}/harness/page.css", "plates/harness/page.css"),
    (f"{PLATES}/glyphbox.mjs", "plates/tools/glyphbox.mjs"),
    (f"{PLATES}/cloudcmp.mjs", "plates/tools/cloudcmp.mjs"),
    (f"{PLATES}/arctest.mjs", "plates/tools/arctest.mjs"),
    (f"{PLATES}/measlens.mjs", "plates/tools/measlens.mjs"),
    (f"{PLATES}/findlens.py", "plates/tools/findlens.py"),
    (f"{PLATES}/verify_lens.py", "plates/tools/verify_lens.py"),
    (f"{PLATES}/pixsample.py", "plates/tools/pixsample.py"),
    (f"{PLATES}/shot.mjs", "plates/tools/shot.mjs"),
    (f"{PLATES}/render_review.py", "plates/tools/render_review.py"),
    (f"{PLATES}/pagecheck.mjs", "plates/tools/pagecheck.mjs"),
    (f"{PLATES}/do_final.py", "plates/tools/do_final.py"),
    (f"{PLATES}/vlmreview.py", "plates/tools/vlmreview.py"),
    (f"{PLATES}/v3/plate1-pipeline.svg", "plates/v3/plate1-pipeline.svg"),
    (f"{PLATES}/v3/plate2-fleet.svg", "plates/v3/plate2-fleet.svg"),
    (f"{PLATES}/v3/plate3-modes.svg", "plates/v3/plate3-modes.svg"),
    (f"{PLATES}/v3/plate4-elevation.svg", "plates/v3/plate4-elevation.svg"),
    (f"{PLATES}/v3/beats-cards.html", "plates/v3/beats-cards.html"),
    (f"{PLATES}/v3/readout-band.html", "plates/v3/readout-band.html"),
    (f"{PLATES}/v3/beats.css", "plates/v3/beats.css"),
    (f"{PLATES}/v3/diagrams.css", "plates/v3/diagrams.css"),
    (f"{PLATES}/v3/check-beats.mjs", "plates/v3/check-beats.mjs"),
]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def copy_into(root, pairs):
    made, missing = [], []
    for src, rel in pairs:
        if not os.path.exists(src):
            missing.append(rel); continue
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        made.append(rel)
    return made, missing

# ---- 1. immutable snapshot (written twice, once with the manifest) ----
shutil.rmtree(SNAP, ignore_errors=True)
made, missing = copy_into(SNAP, FILES)
manifest = {rel: sha256(os.path.join(SNAP, rel)) for rel in sorted(made)}
snap_manifest_path = os.path.join(SNAP, "MANIFEST.sha256")
with open(snap_manifest_path, "w") as f:
    f.write(f"# VisionBridge demo — frozen v3 snapshot\n# {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    f.write("# this is the presentation approved on 18 Sep 2026; do not edit in place\n")
    for rel, h in manifest.items():
        f.write(f"{h}  {rel}\n")
print(f"[1] SNAPSHOT  {SNAP}")
print(f"    files {len(made)}   manifest {os.path.basename(snap_manifest_path)}   missing {missing or 'none'}")
print(f"    index-v3.html sha256 {manifest.get('site/index-v3.html', '?')[:16]}…")

# ---- 2. lean staging copy in the repo ----
made2, missing2 = copy_into(DEST, FILES)
# a manifest of what went into the repo, and a README
with open(os.path.join(DEST, "MANIFEST.sha256"), "w") as f:
    for rel in sorted(made2):
        f.write(f"{sha256(os.path.join(DEST, rel))}  {rel}\n")
print(f"\n[2] STAGED    {DEST}")
print(f"    files {len(made2)}   missing {missing2 or 'none'}")

# ---- 3. also drop the snapshot manifest into the plates dir for reference ----
shutil.copy2(snap_manifest_path, os.path.join(PLATES, "_versions", "v3-20260918-MANIFEST.sha256"))
print(f"\n[3] manifest copied to {PLATES}/_versions/v3-20260918-MANIFEST.sha256")
print(json.dumps({"snapshot_files": len(made), "staged_files": len(made2),
                  "missing": missing + missing2}, indent=1))
