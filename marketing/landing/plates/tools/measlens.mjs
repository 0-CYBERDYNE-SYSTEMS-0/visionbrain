#!/usr/bin/env node
/**
 * measlens.mjs — where does the overlap lens ACTUALLY render?
 * Arc endpoints are not the shape; measure the rendered geometry and scan for real colour runs.
 * Renders plate2 via Chrome, screenshots the SVG into a canvas, and reports:
 *   - getBBox() of each lens path
 *   - colour runs along horizontal scanlines, so the lens's true x-extent is read from pixels
 */
import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

const P = '/Users/scrimwiggins/visionbridge-plates';
const CSS = fs.readFileSync(path.join(P, 'harness', 'page.css'), 'utf8');
const svg = fs.readFileSync(path.join(P, 'v3', 'plate2-fleet.svg'), 'utf8');
const vbW = 1120;

const html = `<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>${CSS} body{margin:0;background:#eef1f3} .solo{width:${vbW}px} .solo svg{width:100%;height:auto;display:block}</style>
</head><body><div class="solo">${svg}</div></body></html>`;

async function launch() {
  for (const o of [{ channel: 'chrome' }, { channel: 'chromium' }, {}]) {
    try { return await chromium.launch(o); } catch {}
  }
  throw new Error('no browser');
}
const browser = await launch();
const page = await browser.newPage({ viewport: { width: vbW, height: 420 }, deviceScaleFactor: 1 });
await page.setContent(html, { waitUntil: 'networkidle' }).catch(() => {});
await page.evaluate(() => document.fonts.ready);

const geo = await page.evaluate(() => {
  const out = [];
  const svgEl = document.querySelector('.solo svg');
  const all = [...svgEl.querySelectorAll('path, rect, polyline')];
  for (const el of all) {
    const b = el.getBBox();
    const fill = el.getAttribute('fill') || '(none)';
    const stroke = el.getAttribute('stroke') || '(none)';
    if (fill === '#c8d0d6' || fill === '#e4e9ec' || el.tagName === 'polyline')
      out.push({ tag: el.tagName, fill, stroke,
        bbox: { x: +b.x.toFixed(1), y: +b.y.toFixed(1), w: +b.width.toFixed(1), h: +b.height.toFixed(1) } });
  }
  return out;
});
console.log('--- rendered geometry (getBBox, user units) ---');
geo.forEach(g => console.log(` ${g.tag.padEnd(9)} fill=${g.fill.padEnd(9)} stroke=${g.stroke.padEnd(9)} bbox x[${g.bbox.x}..${(g.bbox.x + g.bbox.w).toFixed(1)}] y[${g.bbox.y}..${(g.bbox.y + g.bbox.h).toFixed(1)}] w=${g.bbox.w} h=${g.bbox.h}`));

// pixel scan: draw the live SVG into a canvas at 1:1 and read scanlines
const scans = await page.evaluate(async ({ vbW }) => {
  const svgEl = document.querySelector('.solo svg');
  const clone = svgEl.cloneNode(true);
  // inline the CSS classes so the serialized SVG renders standalone
  const st = document.createElementNS('http://www.w3.org/2000/svg', 'style');
  st.textContent = `.svg-label{font:15px "IBM Plex Sans",sans-serif;fill:#101820}.svg-mono{font:12.5px "IBM Plex Mono",monospace;fill:#5b6570}`;
  clone.insertBefore(st, clone.firstChild);
  clone.setAttribute('width', vbW); clone.setAttribute('height', 420);
  const xml = new XMLSerializer().serializeToString(clone);
  const img = new Image();
  await new Promise((res, rej) => {
    img.onload = res; img.onerror = rej;
    img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(xml)));
  });
  const c = document.createElement('canvas'); c.width = vbW; c.height = 420;
  const ctx = c.getContext('2d'); ctx.drawImage(img, 0, 0);
  const hex = (p) => '#' + [p[0], p[1], p[2]].map(v => v.toString(16).padStart(2, '0')).join('');
  const out = {};
  for (const y of [150, 165, 200, 240, 255]) {
    const runs = []; let prev = null, start = 0;
    for (let x = 300; x <= 760; x++) {
      const px = ctx.getImageData(x, y, 1, 1).data;
      const h = hex(px);
      if (h !== prev) { if (prev !== null && x - start >= 3) runs.push(`${prev} x${start}..${x - 1}`); prev = h; start = x; }
    }
    runs.push(`${prev} x${start}..760`);
    out['y=' + y] = runs.filter(r => !r.startsWith('#eef1f3') || true).slice(0, 14);
  }
  return out;
}, { vbW });

console.log('\n--- colour runs along scanlines (x 300..760), distinct colours only ---');
for (const [k, v] of Object.entries(scans)) console.log(` ${k}: ${v.join('  |  ')}`);

await browser.close();
