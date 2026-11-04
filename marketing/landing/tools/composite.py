#!/usr/bin/env python
"""Composite product-grade demo assets from REAL SAM 3.1 masks."""
import os, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SC = "/Users/scrimwiggins/.hermes/workspace/visionbridge-demo/scenes"
OUT = "/Users/scrimwiggins/.hermes/workspace/visionbridge-demo/out"
os.makedirs(OUT, exist_ok=True)

INK = (247, 248, 249, 255)          # light ink for use on dark frames
RED = (214, 61, 31, 255)            # signal red — detections only
DIM = (247, 248, 249, 165)

def font(size, mono=False, bold=False):
    cands = (["/System/Library/Fonts/SFNSMono.ttf",
              "/System/Library/Fonts/Supplemental/Menlo.ttc",
              "/System/Library/Fonts/Supplemental/Courier New.ttf"] if mono else
             ["/System/Library/Fonts/SFNS.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"])
    for p in cands:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()

def bracket(d, box, pad, color, w=3, arm=26):
    x1, y1, x2, y2 = [v + (-pad if i in (0, 1) else pad) for i, v in enumerate(box)]
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        d.line([(cx, cy), (cx + dx * arm, cy)], fill=color, width=w)
        d.line([(cx, cy), (cx, cy + dy * arm)], fill=color, width=w)

def chip(d, xy, label, sub, f1, f2, anchor_right=False):
    x, y = xy
    tw = max(d.textlength(label, font=f1), d.textlength(sub, font=f2)) if sub else d.textlength(label, font=f1)
    padx, pady = 14, 10
    w, h = tw + padx * 2, (f1.size + (f2.size + 6 if sub else 0)) + pady * 2
    if anchor_right:
        x -= w
    d.rectangle([x, y, x + w, y + h], fill=(16, 24, 32, 214), outline=RED, width=2)
    cy = y + pady
    d.text((x + padx, cy), label, font=f1, fill=INK)
    if sub:
        d.text((x + padx, cy + f1.size + 6), sub, font=f2, fill=DIM)
    return x, y, w, h

# ---------------- HERO: scene C, real SAM mask ----------------
hero = Image.open(f"{SC}/C-close-roof.png").convert("RGBA")
mask = Image.open(f"{OUT}/C-close-roof.sam_mask0.png").convert("L").resize(hero.size, Image.NEAREST)
m = np.asarray(mask) > 0
box = (484, 447, 533, 534)

# soft glow, solid translucent fill on the real mask, crisp outline
glow = Image.new("RGBA", hero.size, (0, 0, 0, 0))
ga = np.zeros((hero.size[1], hero.size[0], 4), np.uint8)
ga[m] = (*RED[:3], 90)
glow = Image.fromarray(ga).filter(ImageFilter.GaussianBlur(14))

fill = np.zeros((hero.size[1], hero.size[0], 4), np.uint8)
fill[m] = (*RED[:3], 96)
hero = Image.alpha_composite(hero, glow)
hero = Image.alpha_composite(hero, Image.fromarray(fill))
# outline = mask minus eroded mask
er = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5))
outline = np.logical_and(m, np.asarray(er) == 0)
oa = np.zeros_like(fill); oa[outline] = (*RED[:3], 235)
hero = Image.alpha_composite(hero, Image.fromarray(oa))

d = ImageDraw.Draw(hero)
bracket(d, box, pad=14, color=RED)
f1, f2, fs = font(30, mono=True), font(20, mono=True), font(19, mono=True)
chip(d, (box[2] + 26, box[1] - 6), "PERSON  0.94", "SAM 3.1  ·  ROOF ZONE", f1, f2)
chip(d, (28, 28), "VISIONBRIDGE", "roof intrusion watch", font(26, bold=True), fs)
d.text((28, hero.size[1] - 44), "FRAME 0142 · 14:22:07 · MASK 1.8 s · ON-PREM", font=fs, fill=DIM)
hero.convert("RGB").save(f"{OUT}/01-hero-locked-person.jpg", quality=94)
print("hero:", hero.size)

# ---------------- WIDE + MAGNIFIER: scene A, real SAM mask ----------------
wide = Image.open(f"{SC}/A-day-roof-field.png").convert("RGBA")
maskA = Image.open(f"{OUT}/A-day-roof-field.sam_mask0.png").convert("L").resize(wide.size, Image.NEAREST)
mA = np.asarray(maskA) > 0
boxA = (524, 339, 535, 359)

fillA = np.zeros((*mA.shape, 4), np.uint8); fillA[mA] = (*RED[:3], 120)
wide = Image.alpha_composite(wide, Image.fromarray(fillA))
d = ImageDraw.Draw(wide)
bracket(d, boxA, pad=18, color=RED, arm=30)
chip(d, (28, 28), "VISIONBRIDGE", "wide area watch · 140 m", font(26, bold=True), font(19, mono=True))

# magnifier inset: crop around the person, upscale, same real mask styling
cx, cy = (boxA[0] + boxA[2]) // 2, (boxA[1] + boxA[3]) // 2
half = 46
crop_box = (cx - half, cy - half, cx + half, cy + half)
crop = Image.open(f"{SC}/A-day-roof-field.png").convert("RGBA").crop(crop_box)
mcrop = np.asarray(maskA.crop(crop_box)) > 0
cf = np.zeros((*mcrop.shape, 4), np.uint8); cf[mcrop] = (*RED[:3], 110)
crop = Image.alpha_composite(crop, Image.fromarray(cf))
Z = 260
inset = crop.resize((Z, Z), Image.LANCZOS)
dd = ImageDraw.Draw(inset)
dd.rectangle([0, 0, Z - 1, Z - 1], outline=RED, width=4)
dd.text((12, Z - 40), "6× · SAM 3.1", font=font(20, mono=True), fill=INK)

ix, iy = wide.size[0] - Z - 40, wide.size[1] - Z - 40
wide.alpha_composite(inset, (ix, iy))
d = ImageDraw.Draw(wide)
d.line([(boxA[0] - 22, boxA[3] + 8), (ix + 10, iy + 20)], fill=(*RED[:3], 200), width=2)
d.text((28, wide.size[1] - 44), "DETECT 1.3 s · MASK 1.8 s · 1024² FRAME · MAC MINI M2 PRO", font=font(19, mono=True), fill=DIM)
wide.convert("RGB").save(f"{OUT}/02-wide-magnified.jpg", quality=94)
print("wide:", wide.size)

# ---------------- FIELD ALERT: phone-shaped broadcast mock ----------------
W, H = 1080, 1920
phone = Image.new("RGB", (W, H), (16, 24, 32))
ph = phone.convert("RGBA")
# top: masked crop of the person, filling upper 2/3
top = Image.open(f"{OUT}/01-hero-locked-person.jpg").convert("RGBA")
scale = W / top.size[0]
top = top.resize((W, int(top.size[1] * scale)), Image.LANCZOS)
ph.alpha_composite(top.crop((0, 0, W, 1180)), (0, 0))
d = ImageDraw.Draw(ph)
d.rectangle([0, 0, W, 1180], outline=(16, 24, 32), width=0)
d.rectangle([0, 1180, W, 1180 + 6], fill=RED)
d.text((56, 1240), "ROOF ALERT", font=font(74, bold=True), fill=INK)
d.text((56, 1348), "ZONE 3 · NORTH WAREHOUSE", font=font(34, mono=True), fill=DIM)
d.text((56, 1404), "1 person · roof line · confidence 0.94", font=font(34), fill=INK)
d.text((56, 1470), "14:22:07 · mask 1.8 s · frame attached", font=font(30, mono=True), fill=DIM)
d.rectangle([56, 1560, W - 56, 1690], outline=RED, width=3)
d.text((88, 1600), "OPEN MASK  →", font=font(44, mono=True), fill=INK)
d.text((56, 1790), "VISION SCOUT · field link", font=font(28, mono=True), fill=DIM)
ph.convert("RGB").save(f"{OUT}/03-field-alert.jpg", quality=94)
print("alert:", ph.size)
print("ASSETS:", sorted(os.path.basename(p) for p in glob.glob(f"{OUT}/*.jpg")))
