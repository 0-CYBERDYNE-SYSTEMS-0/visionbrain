
from PIL import Image
import os
OUT = "/Users/scrimwiggins/.hermes/workspace/visionbridge-demo/shots"

def trim_bottom_bg(im, bg=(238,241,243), tol=6):
    w,h = im.size
    px = im.convert("RGB").load()
    last = 0
    step = 4
    for y in range(h-1, -1, -1):
        row_is_bg = True
        for x in range(0, w, 40):
            r,g,b = px[x,y]
            if abs(r-bg[0])>tol or abs(g-bg[1])>tol or abs(b-bg[2])>tol:
                row_is_bg = False; break
        if not row_is_bg:
            last = y; break
    return im.crop((0,0,w,min(h,last+40)))

for name in ["t1440","t768","t390"]:
    p = f"{OUT}/{name}.png"
    im = Image.open(p)
    print(name, "raw", im.size, end="  ")
    im2 = trim_bottom_bg(im)
    im2.save(f"{OUT}/{name}-full.png")
    print("trimmed", im2.size)

# slice 1440 into readable chunks
im = Image.open(f"{OUT}/t1440-full.png")
w,h = im.size
n = 0
y = 0
CH = 1500
while y < h:
    box = (0, y, w, min(h, y+CH))
    part = im.crop(box)
    part.save(f"{OUT}/slice-1440-{n:02d}.png")
    print("slice", n, box, part.size)
    n += 1
    y += CH
print("total slices", n)
