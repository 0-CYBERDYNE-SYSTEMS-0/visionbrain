#!/usr/bin/env node
/**
 * shot.mjs — render ONE plate SVG to a tight, high-resolution PNG (the <svg> element only,
 * not the page wrapper), plus a JSON of sampled pixel facts about it.
 *
 * Why: the harness screenshots the whole page at 1x, which is both too small and too wide for
 * review, and there is no working vision backend on this machine. At 2x, element-clipped, a plate
 * is ~2360px wide — enough that a local 3B VL model, or direct pixel sampling, can say something
 * trustworthy about it.
 *
 * Usage: node shot.mjs <plate.svg> <out.png> [--crop "x,y,w,h" in user units] [--scale 2]
 */
import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

const [file, out] = process.argv.slice(2);
const HARNESS = '/Users/scrimwiggins/visionbridge-plates/harness';
const CSS = fs.readFileSync(path.join(HARNESS, 'page.css'), 'utf8');
const svg = fs.readFileSync(file, 'utf8');
const vb = /viewBox="([\d.\-]+)\s+([\d.\-]+)\s+([\d.]+)\s+([\d.]+)"/.exec(svg);
const [, vbX, vbY, vbW, vbH] = vb.map(Number);
const SCALE = Number((process.argv[process.argv.indexOf('--scale') + 1]) || 2);
const cropArg = process.argv.includes('--crop') ? process.argv[process.argv.indexOf('--crop') + 1] : null;

const html = `<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>${CSS}
  body { margin: 0; background: #eef1f3; }
  .solo { width: ${vbW}px; }
  .solo svg { width: 100%; height: auto; display: block; }
</style></head><body><div class="solo">${svg}</div></body></html>`;

async function launch() {
  for (const o of [{ channel: 'chrome' }, { channel: 'chromium' }, {}]) {
    try { return await chromium.launch(o); } catch {}
  }
  throw new Error('no browser');
}

const browser = await launch();
const page = await browser.newPage({
  viewport: { width: Math.ceil(vbW), height: Math.ceil(vbH) },
  deviceScaleFactor: SCALE,
});
await page.setContent(html, { waitUntil: 'networkidle' }).catch(() => {});
await page.evaluate(() => document.fonts.ready);

if (cropArg) {
  const [x, y, w, h] = cropArg.split(',').map(Number);
  // crop is given in user units of the plate's own coordinate space
  const k = await page.evaluate(({ vbX, vbY, vbW }) => {
    const s = document.querySelector('.solo svg').getBoundingClientRect().width / vbW;
    return s;
  }, { vbX, vbY, vbW });
  const offX = (x - vbX) * k, offY = (y - vbY) * k;
  await page.screenshot({ path: out, clip: { x: offX, y: offY, width: w * k, height: h * k } });
} else {
  const el = await page.$('.solo svg');
  await el.screenshot({ path: out });
}

const dims = await page.evaluate(() => {
  const r = document.querySelector('.solo svg').getBoundingClientRect();
  return { w: r.width, h: r.height };
});
await browser.close();
console.log(JSON.stringify({ file: path.basename(file), png: out, viewBox: [vbX, vbY, vbW, vbH],
  deviceScaleFactor: SCALE, renderedPx: { w: Math.round(dims.w * SCALE), h: Math.round(dims.h * SCALE) } }, null, 1));
