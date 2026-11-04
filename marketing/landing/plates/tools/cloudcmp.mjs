#!/usr/bin/env node
/**
 * cloudcmp.mjs — settle a factual dispute: what is the ACTUAL rendered bbox and
 * stroked length of the old vs new cloud path? (Arc endpoints are NOT the bbox:
 * a near-semicircular arc bulges far past its junction points.)
 */
import { chromium } from 'playwright';
import fs from 'fs';

const OLD = 'M674,132 A26,26 0 0 1 670.10,80.29 A26,26 0 1 1 721.90,80.29 A26,26 0 0 1 718,132 Z';
const plate = fs.readFileSync('/Users/scrimwiggins/visionbridge-plates/v3/plate3-modes.svg', 'utf8');
const inst = /<g id="p3-mode-instance">([\s\S]*?)<\/g>/.exec(plate);
const NEW = /<path d="([^"]+)"/.exec(inst[1])[1];

const svg = `<svg viewBox="0 0 1120 272" role="img" aria-label="cloud comparison">
<g id="gA"><path d="${OLD}" fill="none" stroke="#101820" stroke-width="1.5"/></g>
<g id="gB"><path d="${NEW}" fill="none" stroke="#101820" stroke-width="1.5"/></g>
</svg>`;

const html = `<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0}svg{width:1120px}</style></head><body>${svg}</body></html>`;

async function launch() {
  for (const o of [{ channel: 'chrome' }, { channel: 'chromium' }, {}]) {
    try { return await chromium.launch(o); } catch {}
  }
  throw new Error('no browser');
}
const browser = await launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 800 } });
await page.setContent(html, { waitUntil: 'load' });
const out = await page.evaluate(() => {
  const r = {};
  for (const [name, id] of [['OLD (batch-1)', 'gA'], ['NEW (batch-2 "fix")', 'gB']]) {
    const p = document.querySelector('#' + id + ' path');
    const b = p.getBBox();
    r[name] = { path: p.getAttribute('d'),
      bbox: { x: +b.x.toFixed(2), y: +b.y.toFixed(2), w: +b.width.toFixed(2), h: +b.height.toFixed(2) },
      boxArea: +(b.width * b.height).toFixed(0),
      strokedLength: +p.getTotalLength().toFixed(2),
      inkLengthXStroke: +(p.getTotalLength() * 1.5).toFixed(0),
      bottom: +(b.y + b.height).toFixed(2) };
  }
  return r;
});
await browser.close();
console.log(JSON.stringify(out, null, 1));
const a = out['OLD (batch-1)'], b = out['NEW (batch-2 "fix")'];
console.log('\n--- verdict ---');
console.log(`OLD box ${a.bbox.w}x${a.bbox.h} (area ${a.boxArea})  len ${a.strokedLength}`);
console.log(`NEW box ${b.bbox.w}x${b.bbox.h} (area ${b.boxArea})  len ${b.strokedLength}`);
console.log(`OLD was ${a.bbox.w <= 60 ? 'GENUINELY ~52 wide (my defect claim was RIGHT)' : 'ALREADY ~' + a.bbox.w + ' wide (my claim was WRONG)'}`);
console.log(`shared baseline y=132: OLD bottom ${a.bottom} / NEW bottom ${b.bottom}`);
