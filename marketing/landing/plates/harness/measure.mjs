#!/usr/bin/env node
/**
 * measure.mjs — uniform SVG plate gate for the VisionBridge page.
 *
 * Renders ONE inline SVG fragment in the REAL page context (page.css, .wrap, .diagram)
 * at a given viewport width, then reports hard numbers:
 *   - text-vs-text collisions  (user units, >2 uu² counts as a collision)
 *   - elements outside the viewBox
 *   - contrast ratio for every text fill against the plate ground
 *   - font/size census, aria/role presence, external refs
 *
 * Usage:  node measure.mjs <plate.svg> [--width 1440] [--png out.png]
 * Exit code 1 if any hard gate fails (collisions / out-of-frame / a11y / refs).
 *
 * Authoritative context: the page is 1180px measure at a 1440px viewport, so a
 * 1120-unit-wide plate renders at a 1.0536 scale. All reported geometry is
 * converted back to USER UNITS so it is comparable across plates.
 */
import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

const argv = process.argv.slice(2);
const file = argv[0];
if (!file) { console.error('usage: node measure.mjs <plate.svg> [--width N] [--png out.png]'); process.exit(2); }
const getFlag = (n, d) => { const i = argv.indexOf(n); return i === -1 ? d : argv[i + 1]; };
const WIDTH = parseInt(getFlag('--width', '1440'), 10);
const PNG = getFlag('--png', null);

const HARNESS = path.dirname(new URL(import.meta.url).pathname);
const CSS = fs.readFileSync(path.join(HARNESS, 'page.css'), 'utf8');
const svg = fs.readFileSync(file, 'utf8');

const vb = /viewBox="([^"]+)"/.exec(svg);
if (!vb) { console.error('FAIL: no viewBox'); process.exit(1); }
const [vbX, vbY, vbW, vbH] = vb[1].trim().split(/\s+/).map(Number);

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..700;1,9..144,300..600&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>${CSS}</style></head><body>
<section class="wrap"><div class="diagram">${svg}</div></section>
</body></html>`;

async function launch() {
  const tries = [
    { channel: 'chrome' },                       // installed Google Chrome — no download needed
    { channel: 'chromium' },
    {},
  ];
  let last;
  for (const opt of tries) {
    try { return await chromium.launch(opt); } catch (e) { last = e; }
  }
  throw last;
}
const browser = await launch();
const page = await browser.newPage({ viewport: { width: WIDTH, height: 1200 }, deviceScaleFactor: 2 });
await page.setContent(html, { waitUntil: 'networkidle' }).catch(() => {});
await page.evaluate(() => document.fonts.ready);

const data = await page.evaluate(({ vbW, vbH }) => {
  const svgEl = document.querySelector('.diagram svg');
  const sr = svgEl.getBoundingClientRect();
  const scale = sr.width / vbW;                      // px per user unit
  const toUU = (px) => px / scale;
  const originX = sr.left, originY = sr.top;

  const rectOf = (el) => {
    const r = el.getBoundingClientRect();
    return { x: (r.left - originX) / scale, y: (r.top - originY) / scale,
             w: r.width / scale, h: r.height / scale };
  };

  // ---- text census + collisions ----
  const texts = [...svgEl.querySelectorAll('text')].map((el, i) => {
    const r = rectOf(el);
    const cs = getComputedStyle(el);
    return { i, t: (el.textContent || '').replace(/\s+/g, ' ').trim(),
             cls: el.getAttribute('class') || '', x: +r.x.toFixed(2), y: +r.y.toFixed(2),
             w: +r.w.toFixed(2), h: +r.h.toFixed(2), size: cs.fontSize, fill: cs.fill };
  });
  const collisions = [];
  for (let a = 0; a < texts.length; a++) for (let b = a + 1; b < texts.length; b++) {
    const A = texts[a], B = texts[b];
    const ox = Math.min(A.x + A.w, B.x + B.w) - Math.max(A.x, B.x);
    const oy = Math.min(A.y + A.h, B.y + B.h) - Math.max(A.y, B.y);
    if (ox > 0 && oy > 0) {
      const area = ox * oy;
      if (area > 2) collisions.push({ a: A.t.slice(0, 42), b: B.t.slice(0, 42),
                                      overlapUU: [+ox.toFixed(1), +oy.toFixed(1)], area: +area.toFixed(1) });
    }
  }

  // ---- geometry outside the frame ----
  const outOfFrame = [];
  for (const el of svgEl.querySelectorAll('text,rect,circle,path,polyline,line')) {
    if (!el.getBBox) continue;
    let bb; try { bb = el.getBBox(); } catch { continue; }
    const pad = 0.5;
    if (bb.x < -pad || bb.y < -pad || bb.x + bb.width > vbW + pad || bb.y + bb.height > vbH + pad) {
      outOfFrame.push({ tag: el.tagName, label: (el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 40),
                        bbox: { x: +bb.x.toFixed(1), y: +bb.y.toFixed(1), w: +bb.width.toFixed(1), h: +bb.height.toFixed(1) } });
    }
  }

  // ---- a11y + refs ----
  const role = svgEl.getAttribute('role'); const aria = svgEl.getAttribute('aria-label');
  const externals = [];
  for (const el of svgEl.querySelectorAll('*')) {
    for (const at of ['href', 'src', 'xlink:href']) {
      if (el.hasAttribute(at)) externals.push({ tag: el.tagName, attr: at, val: el.getAttribute(at).slice(0, 60) });
    }
  }
  const ids = [...svgEl.querySelectorAll('[id]')].map(e => e.id);
  const dupIds = ids.filter((v, i) => ids.indexOf(v) !== i);
  const classes = [...new Set([...svgEl.querySelectorAll('[class]')].map(e => e.getAttribute('class')))]
    .filter(c => !['svg-label', 'svg-mono', 'pulse', 'reveal'].includes(c));

  return { scale: +scale.toFixed(4), renderedPx: { w: +sr.width.toFixed(1), h: +sr.height.toFixed(1) },
           texts, collisions, outOfFrame, role, aria, externals, ids,
           counts: Object.fromEntries(['rect','line','circle','path','text','polyline','g','marker'].map(t => [t, svgEl.querySelectorAll(t).length])),
           classesNotInPage: classes, dupIds };
}, { vbW, vbH });

// marker refs are resolved in Node — the raw SVG source is not available in the page
const markerRefs = [...svg.matchAll(/url\(#([^)]+)\)/g)].map(m => m[1]);
data.missingMarkers = [...new Set(markerRefs)].filter(r => !data.ids.includes(r));

// ---- contrast of every text fill vs the plate ground (#eef1f3 or #e4e9ec) ----
const lum = (color) => {
  let rgb;
  const m = /rgba?\((\d+),\s*(\d+),\s*(\d+)/.exec(color);
  if (m) rgb = [+m[1], +m[2], +m[3]];
  else {
    const h = /^#([0-9a-f]{6})$/i.exec(color.trim());
    if (h) rgb = [0, 2, 4].map(i => parseInt(h[1].slice(i, i + 2), 16));
  }
  if (!rgb) return null;
  const [r, g, b] = rgb.map(v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const ratio = (a, b) => {
  const la = lum(a), lb = lum(b);
  if (la === null || lb === null) return NaN;
  const [x, y] = [la, lb].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
};
const GROUNDS = ['#eef1f3', '#e4e9ec'];
const contrast = data.texts.map(t => {
  const worst = Math.min(...GROUNDS.map(g => ratio(t.fill, g)).filter(v => !Number.isNaN(v)));
  return { t: t.t.slice(0, 40), fill: t.fill, size: t.size, ratio: +worst.toFixed(2) };
});
const contrastFails = contrast.filter(c => {
  const px = parseFloat(c.size);
  return !(c.ratio >= (px >= 24 ? 3 : 4.5));
});

const HARD = {
  textCollisions: data.collisions.length,
  outOfFrame: data.outOfFrame.length,
  contrastFails: contrastFails.length,
  missingRoleOrAria: (!data.role || !data.aria) ? 1 : 0,
  externalRefs: data.externals.length,
  classesNotInPage: data.classesNotInPage.length,
  dupIds: data.dupIds.length,
  missingMarkers: data.missingMarkers.length,
};
const failed = Object.entries(HARD).filter(([, v]) => v > 0);

const report = { plate: path.basename(file), viewport: WIDTH, viewBox: [vbX, vbY, vbW, vbH],
                 scale: data.scale, renderedPx: data.renderedPx, counts: data.counts,
                 textCount: data.texts.length, HARD, PASS: failed.length === 0,
                 collisions: data.collisions, outOfFrame: data.outOfFrame,
                 contrastFails, classesNotInPage: data.classesNotInPage,
                 missingMarkers: data.missingMarkers, externals: data.externals,
                 contrastMin: Math.min(...contrast.map(c => c.ratio)) };

if (PNG) { await page.screenshot({ path: PNG, fullPage: true }); report.png = PNG; }
await browser.close();

console.log(JSON.stringify(report, null, 2));
process.exit(report.PASS ? 0 : 1);
