#!/usr/bin/env node
/**
 * glyphbox.mjs — measure the rendered bounding box of named glyph groups in a plate.
 * Verifies claims like "all four panel glyphs now carry equal optical weight".
 *
 * Usage: node glyphbox.mjs <plate.svg> <group-id> [<group-id> ...]
 * Renders the fragment in the real page CSS at 1440 and reports each group's
 * bbox in USER UNITS plus its box area, so parity can be checked as arithmetic.
 */
import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

const [file, ...groups] = process.argv.slice(2);
const HARNESS = '/Users/scrimwiggins/visionbridge-plates/harness';
const CSS = fs.readFileSync(path.join(HARNESS, 'page.css'), 'utf8');
const svg = fs.readFileSync(file, 'utf8');
const vbW = Number(/viewBox="[\d.\-]+\s+[\d.\-]+\s+([\d.]+)/.exec(svg)[1]);

const html = `<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>${CSS}</style></head><body><section class="wrap"><div class="diagram">${svg}</div></section></body></html>`;

async function launch() {
  for (const o of [{ channel: 'chrome' }, { channel: 'chromium' }, {}]) {
    try { return await chromium.launch(o); } catch {}
  }
  throw new Error('no browser');
}

const browser = await launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1200 }, deviceScaleFactor: 2 });
await page.setContent(html, { waitUntil: 'networkidle' }).catch(() => {});
await page.evaluate(() => document.fonts.ready);

const out = await page.evaluate(({ vbW, groups }) => {
  const svgEl = document.querySelector('.diagram svg');
  const scale = svgEl.getBoundingClientRect().width / vbW;
  const res = [];
  // per-group: bbox of the group's own geometry only (exclude its <text> children)
  for (const gid of groups) {
    const g = svgEl.querySelector('#' + CSS_escape(gid)) || svgEl.querySelector('g[id="' + gid + '"]');
    if (!g) { res.push({ gid, error: 'not found' }); continue; }
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, n = 0;
    for (const el of g.querySelectorAll('path,rect,circle,line,polyline,polygon')) {
      if (!el.getBBox) continue;
      // the panel border (rule #c8d0d6) is the group's CONTAINER, not its glyph:
      // measuring it would report 240x216 for every panel and prove nothing.
      const st = el.getAttribute('stroke') || '', fl = el.getAttribute('fill') || '';
      if (st !== '#101820' && fl !== '#101820') continue;
      let b; try { b = el.getBBox(); } catch { continue; }
      x0 = Math.min(x0, b.x); y0 = Math.min(y0, b.y);
      x1 = Math.max(x1, b.x + b.width); y1 = Math.max(y1, b.y + b.height); n++;
    }
    res.push({ gid, shapes: n,
      bbox: { x: +x0.toFixed(1), y: +y0.toFixed(1), w: +(x1 - x0).toFixed(1), h: +(y1 - y0).toFixed(1) },
      boxArea: +((x1 - x0) * (y1 - y0)).toFixed(0),
      bottom: +y1.toFixed(1), cx: +((x0 + x1) / 2).toFixed(1) });
  }
  return res;
  function CSS_escape(s) { return s.replace(/([^\w-])/g, '\\$1'); }
}, { vbW, groups });

await browser.close();
console.log(JSON.stringify(out, null, 1));

const ok = out.filter(g => !g.error);
if (ok.length > 1) {
  const areas = ok.map(g => g.boxArea);
  const widths = ok.map(g => g.bbox.w);
  const bottoms = ok.map(g => g.bottom);
  const cx = ok.map(g => g.cx);
  console.log('\n--- parity ---');
  console.log('widths  :', widths.join(', '), ' spread =', (Math.max(...widths) - Math.min(...widths)).toFixed(1));
  console.log('areas   :', areas.join(', '), ' max/min ratio =', (Math.max(...areas) / Math.min(...areas)).toFixed(2));
  console.log('bottoms :', bottoms.join(', '), ' spread =', (Math.max(...bottoms) - Math.min(...bottoms)).toFixed(1));
  console.log('centers :', cx.join(', '), ' spread =', (Math.max(...cx) - Math.min(...cx)).toFixed(1));
}
