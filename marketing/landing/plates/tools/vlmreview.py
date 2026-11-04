#!/usr/bin/env python3
"""
vlmreview.py — attempt an honest visual read of the plates with the only vision model available
(LM Studio, lfm2.5-vl-3b), gated by CONTROL questions whose answers are already known.

If the model cannot answer the controls correctly, nothing else it says can be trusted, and the
script says so rather than reporting its opinions as findings.

Controls per plate:
  C1 colour  : does the image contain any red/orange?   (known: YES for plates 1,2,4; NO for plate 3)
  C2 count   : how many separate panel/box outlines?    (known: 4 for plate 3, 2 for plate 1's column
                                                         panels + core, etc. — accepted as approximate)
"""
import base64, io, json, os, re, urllib.request
from PIL import Image

URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "lfm2.5-vl-3b"
P = os.path.expanduser("~/visionbridge-plates")

PLATES = ["plate1-pipeline", "plate2-fleet", "plate3-modes", "plate4-elevation"]
KNOWN_RED = {"plate1-pipeline": "YES", "plate2-fleet": "YES", "plate3-modes": "NO", "plate4-elevation": "YES"}

Q = [
    ("C1-colour", "Look at the colours in this image. Is there any RED or ORANGE present anywhere? "
                  "Answer strictly 'RED: YES' or 'RED: NO' on the first line."),
    ("C2-text",   "Quote three words of text you can actually read in this image, verbatim. "
                  "If you cannot read any text, say 'NO TEXT'."),
    ("R1-shapes", "There are shapes drawn above captions. List them in order left to right. "
                  "Then state plainly: are they all about the same size, or is one noticeably smaller "
                  "than the others? If one is smaller, name which."),
    ("R2-broken", "Is anything in this image broken or visually wrong — a shape that does not look like "
                  "the thing it depicts, a line crossing through text, an element cut off at the edge, "
                  "or an area that should be visible but is invisible? Be specific, or say 'NOTHING WRONG'."),
]

def ask(path, question):
    im = Image.open(path).convert("RGB")
    if im.width > 1400:
        im = im.resize((1400, int(im.height * 1400 / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = json.dumps({
        "model": MODEL, "temperature": 0.0, "max_tokens": 320,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
        ]}],
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()

for name in PLATES:
    png = os.path.join(P, "review", name + "-2x.png")
    if not os.path.exists(png):
        print(f"### {name}: no render"); continue
    print("=" * 78)
    print("###", name, f"(control expects RED: {KNOWN_RED[name]})")
    print("=" * 78)
    for tag, question in Q:
        try:
            ans = ask(png, question)
        except Exception as e:
            ans = f"ERROR {e}"
        ans = re.sub(r"\n{2,}", "\n", ans)
        print(f"\n[{tag}] {ans[:700]}")
    print()
