#!/usr/bin/env node
/**
 * pagecheck.mjs — whole-assembled-page verification.
 * Loads a URL at three widths and reports:
 *   - horizontal overflow (px)
 *   - text-vs-text collisions between LEAF text nodes (Range rects), whole page
 *   - image load status (broken <img> would silently ruin the page)
 *   - svg[role=img] census + duplicate DOM ids (real DOM, not regex)
 *   - console / page errors
 * Usage: node pagecheck.mjs <url>
 */
import { chromium } from 'playwright';

const url = process.argv[2];
if (!url) { console.error('usage: node pagecheck.mjs <url>'); process.exit(2); }

async function launch() {
  for (const o of [{ channel: 'chrome' }, { channel: 'chromium' }, {}]) {
    try { return await chromium.launch(o); } catch {}
  }
  throw new Error('no browser');
}

const browser = await launch();
let hardFail = 0;
for (const width of [1440, 768, 390]) {
  const page = await browser.newPage({ viewport: { width, height: 1000 }, deviceScaleFactor: 2 });
  const errs = [];
  page.on('pageerror', e => errs.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
  await page.goto(url, { waitUntil: 'networkidle', timeout: 45000 }).catch(() => {});
  await page.evaluate(() => document.fonts.ready);

  const r = await page.evaluate(() => {
    const de = document.documentElement;
    const overflow = de.scrollWidth - de.clientWidth;

    // leaf text nodes → client rects via Range
    const leaves = [];
    const walk = (n) => {
      for (const c of n.childNodes) {
        if (c.nodeType === 3 && c.textContent.trim()) leaves.push(c);
        else if (c.nodeType === 1) walk(c);
      }
    };
    walk(document.body);
    const boxes = [];
    for (const t of leaves) {
      const rg = document.createRange(); rg.selectNodeContents(t);
      const b = rg.getBoundingClientRect();
      if (b.width > 0.5 && b.height > 0.5)
        boxes.push({ t: t.textContent.replace(/\s+/g, ' ').trim().slice(0, 38),
                     x: b.left, y: b.top, w: b.width, h: b.height,
                     svg: !!t.parentElement.closest('svg'), blk: blockOf(t.parentElement) });
    }
    // Two inline runs on the SAME line (e.g. <b>foo</b> bar) have bounding rects that
    // necessarily overlap by a full line-height. That is typography, not a collision.
    // Only compare text belonging to DIFFERENT block containers.
    const blkId = new Map(); let nextId = 0;
    const bid = (el) => { if (!blkId.has(el)) blkId.set(el, nextId++); return blkId.get(el); };
    function blockOf(el) {
      let n = el;
      while (n && n !== document.body) {
        const d = getComputedStyle(n).display;
        if (['block', 'list-item', 'flex', 'grid', 'table-cell', 'flow-root'].includes(d)) return n;
        n = n.parentElement;
      }
      return document.body;
    }
    const collisions = [];
    for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
      const A = boxes[i], B = boxes[j];
      if (A.svg !== B.svg) continue;              // SVG text lives in its own unit space
      if (bid(A.blk) === bid(B.blk)) continue;    // same block container → inline siblings
      const ox = Math.min(A.x + A.w, B.x + B.w) - Math.max(A.x, B.x);
      const oy = Math.min(A.y + A.h, B.y + B.h) - Math.max(A.y, B.y);
      if (ox > 2 && oy > 2 && ox * oy > 8)
        collisions.push({ a: A.t, b: B.t, px: [+ox.toFixed(1), +oy.toFixed(1)] });
    }

    const imgs = [...document.images].map(i => ({ src: i.getAttribute('src'),
      ok: i.complete && i.naturalWidth > 0, nat: i.naturalWidth + 'x' + i.naturalHeight }));
    const svgRole = document.querySelectorAll('svg[role="img"]').length;
    const svgAny = document.querySelectorAll('svg').length;
    // rendered size of each plate + the effective size of a 12.5px mono label inside it:
    // at a 390px viewport a 1120-unit plate drawn at 342px renders 12.5px text as 3.8px.
    const plates = [...document.querySelectorAll('svg[role="img"]')].map(s => {
      const r = s.getBoundingClientRect();
      const vbW = (s.getAttribute('viewBox') || '').trim().split(/\s+/)[2] || 1120;
      return { w: Math.round(r.width), h: Math.round(r.height),
               scale: +(r.width / vbW).toFixed(3),
               monoPx: +(12.5 * r.width / vbW).toFixed(1) };
    });
    const ids = [...document.querySelectorAll('[id]')].map(e => e.id);
    const dupIds = [...new Set(ids.filter((v, i) => ids.indexOf(v) !== i))];
    const ariaMissing = [...document.querySelectorAll('svg[role="img"]')]
      .filter(s => !s.getAttribute('aria-label')).length;
    return { overflow, textNodes: boxes.length, collisions, imgs, svgRole, svgAny, plates, dupIds, ariaMissing };
  });

  const bad = r.overflow !== 0 || r.collisions.length > 0 || r.imgs.some(i => !i.ok) ||
              r.dupIds.length > 0 || r.ariaMissing > 0 || r.svgRole !== 4 || r.svgAny !== 4;
  if (bad) hardFail++;

  console.log(`===== ${width}px =====`);
  console.log(`  horizontal overflow.. ${r.overflow} px`);
  console.log(`  text collisions...... ${r.collisions.length}  (${r.textNodes} leaf text boxes tested)`);
  r.collisions.slice(0, 6).forEach(c => console.log(`      ${JSON.stringify(c.a)} x ${JSON.stringify(c.b)} overlap ${c.px} px`));
  console.log(`  images............... ${r.imgs.filter(i => i.ok).length}/${r.imgs.length} loaded`);
  r.imgs.filter(i => !i.ok).forEach(i => console.log(`      BROKEN ${i.src} (natural ${i.nat})`));
  console.log(`  svg role=img......... ${r.svgRole}  (total svg ${r.svgAny})   aria missing ${r.ariaMissing}`);
  const pl = r.plates[0];
  if (pl) console.log(`  plate render......... ${pl.w}x${pl.h}px  scale ${pl.scale}  -> 12.5px mono renders at ${pl.monoPx}px`);
  console.log(`  duplicate DOM ids.... ${r.dupIds.length ? r.dupIds.join(', ') : 'none'}`);
  console.log(`  errors............... ${errs.length ? errs.slice(0, 4).join(' | ') : 'none'}`);
  console.log(`  VERDICT.............. ${bad ? 'FAIL' : 'PASS'}`);
  console.log();
  await page.close();
}
await browser.close();
console.log(hardFail === 0 ? 'ALL WIDTHS PASS' : `FAILED at ${hardFail} width(s)`);
process.exit(hardFail === 0 ? 0 : 1);
