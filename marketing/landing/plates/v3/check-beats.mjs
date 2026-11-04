#!/usr/bin/env node
/**
 * check-beats.mjs — HTML-component gate for the VisionBridge v3 card furniture.
 *
 * Unlike harness/measure.mjs (which gates one SVG plate against its viewBox) this
 * gates two HTML components in the real page context:
 *
 *   A. .beats    the numbered step row
 *   B. .readout  the key/value band under the hero
 *
 * It renders the components in the REAL page CSS (harness/page.css), at a given
 * viewport width, and reports hard numbers:
 *
 *   horizontalOverflowPx   page scrollWidth - clientWidth, plus any component box
 *                          whose rect leaves the viewport      (hard: must be 0)
 *   textCollisions         pairwise text-line-rect overlaps, using Range.getClientRects
 *                          so it measures rendered line boxes, not element boxes
 *                                                             (hard: must be 0)
 *   cardHeightDelta        tallest card - shortest card in .beats (hard: must be 0)
 *   bandHeightDelta        per row-band version of the same number
 *   internalAlignment      per band: delta of ordinal top / h3 top / p top /
 *                          last-line bottom across that band's cards
 *   baselineDelta          per readout row: |first baseline of key - of value|
 *   columnAlignment        per readout row: x of the key cell, x of the value cell
 *   redInk                 every computed color inside the two components that is
 *                          one of the two red tokens      (hard for this author: 0)
 *   contrastMin            every text color against the paper ground
 *   shape                  "4-up" / "2-up" / "1-up", read as the number of distinct
 *                          row bands the four cards occupy
 *
 * Usage:
 *   node check-beats.mjs [--variant v3|v2|both] [--widths 1440,768,390] [--png]
 *
 * Run it from /Users/scrimwiggins/visionbridge-plates/harness so node resolves
 * playwright, or from anywhere once the node_modules symlink exists beside it.
 * Chromium is launched as { channel: 'chrome' } — the bundled build is missing.
 *
 * v2 = baseline/beats-cards.html + the readout band sliced out of the live
 *      site/index-v2.html, styled by harness/page.css.  Printed first, so the
 *      before/after numbers sit next to each other.
 */
import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

const ROOT = '/Users/scrimwiggins/visionbridge-plates';
const HARNESS = path.join(ROOT, 'harness');
const V3 = path.join(ROOT, 'v3');
const SHOTS = path.join(ROOT, 'shots');
const LIVE = '/Users/scrimwiggins/.hermes/workspace/visionbridge-demo/site/index-v2.html';

const argv = process.argv.slice(2);
const getFlag = (n, d) => { const i = argv.indexOf(n); return i === -1 ? d : argv[i + 1]; };
const VARIANT = getFlag('--variant', 'both');
const WIDTHS = getFlag('--widths', '1440,768,390').split(',').map(Number);
const WANT_PNG = argv.includes('--png');

const read = (p) => fs.readFileSync(p, 'utf8');
const sliceBetween = (src, a, b) => {
  const i = src.indexOf(a); if (i === -1) throw new Error('missing ' + a);
  const j = src.indexOf(b, i); if (j === -1) throw new Error('missing ' + b);
  return src.slice(i, j);
};

const PAGE_CSS = read(path.join(HARNESS, 'page.css'));
const V3_CSS = read(path.join(V3, 'beats.css'));
const BEATS_V3 = read(path.join(V3, 'beats-cards.html'));
const READOUT_V3 = read(path.join(V3, 'readout-band.html'));
const BEATS_V2 = sliceBetween(read(path.join(ROOT, 'baseline', 'beats-cards.html')), '<div class="beats">', '</section>');
const READOUT_V2 = sliceBetween(read(LIVE), '<div class="wrap readout">', '\n  </div>') + '\n  </div>';

const FONTS = '<link rel="preconnect" href="https://fonts.googleapis.com">'
  + '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
  + '<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..700;1,9..144,300..600'
  + '&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">';

const PLACEHOLDER_SVG = '<svg viewBox="0 0 1120 350" role="img" aria-label="placeholder for plate 1">'
  + '<rect x="28" y="28" width="1064" height="294" fill="none" stroke="#c8d0d6"/>'
  + '<text x="28" y="24" class="svg-mono">placeholder — plate 1 is another agent\'s file</text></svg>';

function buildHtml(variant) {
  const css = PAGE_CSS + (variant === 'v3' ? '\n' + V3_CSS : '');
  const readout = variant === 'v3' ? READOUT_V3 : READOUT_V2;
  const beats = variant === 'v3' ? BEATS_V3 : BEATS_V2;
  return `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
${FONTS}
<style>${css}</style></head><body>
${readout}
<section class="wrap" id="how">
  <h2>What the system actually does</h2>
  <p class="big" style="max-width:70ch">Inputs come in from wherever you already have eyes.</p>
  <div class="diagram">${PLACEHOLDER_SVG}</div>
${beats}
</section>
</body></html>`;
}
async function measure(browser, variant, width) {
  const page = await browser.newPage({ viewport: { width, height: 1400 }, deviceScaleFactor: 1 });
  await page.setContent(buildHtml(variant), { waitUntil: 'networkidle' }).catch(() => {});
  await page.evaluate(() => document.fonts.ready).catch(() => {});

  const data = await page.evaluate((W) => {
    const r2 = (n) => Math.round(n * 100) / 100;
    const uniq = (a) => [...new Set(a.map((v) => Math.round(v * 10) / 10))];

    /* ---------- fonts actually in use ---------- */
    const fontsReady = {
      plexMono: document.fonts.check('13px "IBM Plex Mono"'),
      plexSans: document.fonts.check('15px "IBM Plex Sans"'),
      fraunces: document.fonts.check('18px Fraunces'),
      loaded: [...document.fonts].filter((f) => f.status === 'loaded').map((f) => f.family).filter((v, i, a) => a.indexOf(v) === i),
    };

    /* ---------- render helpers ---------- */
    const lines = (root) => {
      const out = [];
      const walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      let n;
      while ((n = walk.nextNode())) {
        if (!n.nodeValue || !n.nodeValue.trim()) continue;
        const rg = document.createRange();
        rg.selectNodeContents(n);
        for (const r of rg.getClientRects()) {
          if (r.width < 0.5 || r.height < 0.5) continue;
          out.push({
            t: n.nodeValue.replace(/\s+/g, ' ').trim().slice(0, 46),
            host: (n.parentElement.getAttribute('class') || n.parentElement.tagName).split(' ')[0].toLowerCase(),
            x: r.left, y: r.top, w: r.width, h: r.height,
          });
        }
      }
      return out;
    };
    const collisions = (rs) => {
      const c = [];
      for (let i = 0; i < rs.length; i++) {
        for (let j = i + 1; j < rs.length; j++) {
          const a = rs[i], b = rs[j];
          const ox = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
          const oy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
          if (ox > 0 && oy > 0 && ox * oy > 2) {
            c.push({ a: a.t, b: b.t, aHost: a.host, bHost: b.host, overlapPx: [r2(ox), r2(oy)], area: r2(ox * oy) });
          }
        }
      }
      return c;
    };
    const ctx = document.createElement('canvas').getContext('2d');
    const fmCache = new Map();
    const fm = (cs) => {
      const key = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
      if (!fmCache.has(key)) {
        ctx.font = key;
        const m = ctx.measureText('Hxg');
        const size = parseFloat(cs.fontSize);
        fmCache.set(key, {
          asc: m.fontBoundingBoxAscent || size * 0.78,
          desc: m.fontBoundingBoxDescent || size * 0.22,
        });
      }
      return fmCache.get(key);
    };
    const firstBaseline = (el) => {
      const cs = getComputedStyle(el);
      const r = el.getBoundingClientRect();
      const { asc, desc } = fm(cs);
      const lh = parseFloat(cs.lineHeight) || (asc + desc);
      return r.top + parseFloat(cs.paddingTop) + (lh - (asc + desc)) / 2 + asc;
    };
    /* exact baseline: a zero-size inline-block's bottom edge sits ON the
       baseline of the line box it is inserted into. */
    const baselineOf = (el) => {
      if (!el) return null;
      const probe = document.createElement('span');
      probe.setAttribute('style', 'display:inline-block;width:0;height:0;padding:0;margin:0;vertical-align:baseline');
      el.insertBefore(probe, el.firstChild);
      const y = probe.getBoundingClientRect().bottom;
      probe.remove();
      return y;
    };

    /* ---------- page overflow ---------- */
    const de = document.documentElement;
    const scrollers = [];
    for (const el of document.querySelectorAll('.beats, .readout, .beats *, .readout *')) {
      const r = el.getBoundingClientRect();
      if (r.width === 0 && r.height === 0) continue;
      if (r.right > W + 0.5 || r.left < -0.5) {
        scrollers.push({ tag: el.tagName, cls: el.getAttribute('class') || '', left: r2(r.left), right: r2(r.right) });
      }
    }

    /* ---------- component A: .beats ---------- */
    const beatsEl = document.querySelector('.beats');
    const beatsRects = beatsEl ? lines(beatsEl) : [];
    const cards = beatsEl ? [...beatsEl.querySelectorAll(':scope > *')].map((el, i) => {
      const r = el.getBoundingClientRect();
      const cs = getComputedStyle(el);
      const t = lines(el);
      const g = (sel) => { const e = el.querySelector(sel); return e ? e.getBoundingClientRect() : null; };
      const h3r = g('h3'), pr = g('p'), nr = g('.n');
      const pRects = t.filter((x) => x.host === 'p');
      const before = getComputedStyle(el, '::before');
      const borderLeftW = parseFloat(cs.borderLeftWidth) || 0;
      const hasSep = before.display !== 'none' && before.content !== 'none' && before.content !== '';
      const sepX = hasSep ? r2(r.left + parseFloat(before.left)) : null;
      const pText = el.querySelector('p') ? el.querySelector('p').textContent.replace(/\s+/g, ' ').trim() : '';
      const pEl = el.querySelector('p'), h3El = el.querySelector('h3'), nEl = el.querySelector('.n');
      const box = (e) => (e ? { top: r2(e.getBoundingClientRect().top), h: r2(e.getBoundingClientRect().height) } : null);
      return {
        i: i + 1,
        ordinal: nEl ? nEl.textContent.trim() : '',
        title: h3El ? h3El.textContent.trim() : '',
        top: r2(r.top), left: r2(r.left), right: r2(r.right), bottom: r2(r.bottom), height: r2(r.height),
        contentWidth: r2(r.width - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight)),
        ordinalTop: nr ? r2(nr.top) : null,
        h3Top: h3r ? r2(h3r.top) : null,
        pTop: pr ? r2(pr.top) : null,
        pBox: box(pEl), h3Box: box(h3El),
        pLineHeight: pEl ? getComputedStyle(pEl).lineHeight : null,
        h3Lines: h3El ? uniq(t.filter((x) => x.host === 'h3').map((x) => x.y)).length : 0,
        pChars: pText.length,
        pLines: uniq(pRects.map((x) => x.y)).length,
        ordinalBaseline: baselineOf(nEl),
        h3Baseline: baselineOf(h3El),
        pBaseline: baselineOf(pEl),
        firstInkTop: t.length ? r2(Math.min(...t.map((x) => x.y))) : null,
        lastInkBottom: t.length ? r2(Math.max(...t.map((x) => x.y + x.h))) : null,
        sepX, sepWidth: hasSep ? parseFloat(before.width) : null,
        sepLeftGap: hasSep ? r2(parseFloat(before.left)) : null,
        borderLeftX: borderLeftW > 0 ? r2(r.left) : null,
      };
    }) : [];

    const heights = cards.map((c) => c.height);
    const cardHeightDelta = heights.length ? r2(Math.max(...heights) - Math.min(...heights)) : null;
    for (let k = 0; k < cards.length; k++) {
      const c = cards[k];
      c.inkFloorInset = c.lastInkBottom === null ? null : r2(c.bottom - c.lastInkBottom);
      if (c.sepX === null || !cards[k - 1]) { c.sepLeftWhitespace = null; c.sepRightWhitespace = null; continue; }
      c.sepLeftWhitespace = r2(c.sepX - cards[k - 1].right);
      c.sepRightWhitespace = r2(c.left - (c.sepX + (c.sepWidth || 1)));
    }

    const bandOf = (c) => cards.filter((o) => Math.abs(o.top - c.top) < 1.5).map((o) => o.i);
    const seenB = [];
    const bands = [];
    for (const c of cards) {
      const key = bandOf(c).join(',');
      if (seenB.includes(key)) continue;
      seenB.push(key);
      const members = cards.filter((o) => bandOf(c).includes(o.i));
      const span = (f) => {
        const v = members.map(f).filter((x) => x !== null && x !== undefined);
        return v.length ? r2(Math.max(...v) - Math.min(...v)) : null;
      };
      const rules = members.map((m) => m.sepX !== null || m.borderLeftX !== null ? m.i : null).filter((v) => v !== null);
      bands.push({
        cards: members.map((m) => m.i),
        rulesOnCards: rules,
        rulesExpected: members.length - 1,
        rulesMatch: rules.length === members.length - 1,
        cardHeightDelta: span((m) => m.height),
        ordinalTopDelta: span((m) => m.ordinalTop),
        h3TopDelta: span((m) => m.h3Top),
        pTopDelta: span((m) => m.pTop),
        ordinalBaselineDelta: span((m) => m.ordinalBaseline),
        h3BaselineDelta: span((m) => m.h3Baseline),
        pBaselineDelta: span((m) => m.pBaseline),
        lastInkBottomDelta: span((m) => m.lastInkBottom),
      });
    }
    const shape = bands.length === 4 ? '1-up' : bands.length === 2 ? '2-up' : bands.length === 1 ? '4-up' : `${bands.length} bands`;
    const ruleMismatches = bands.filter((b) => !b.rulesMatch).length;

    /* ---------- component B: .readout ---------- */
    const roEl = document.querySelector('.readout');
    const roRects = roEl ? lines(roEl) : [];
    const rows = roEl ? [...roEl.querySelectorAll(':scope > div')].map((row) => {
      const b = row.querySelector('b'), s = row.querySelector('span');
      const br = b.getBoundingClientRect(), sr = s.getBoundingClientRect();
      const bcs = getComputedStyle(b), scs = getComputedStyle(s);
      const fam = (cs) => cs.fontFamily.split(',')[0].replace(/["']/g, '');
      const sRects = lines(s);
      const stacked = sr.top >= br.bottom - 0.5;
      const kb = baselineOf(b), vb = baselineOf(s);
      return {
        key: b.textContent.trim(),
        group: row.classList.contains('ro-measured') ? 'measured' : 'architecture',
        keyLeft: r2(br.left), keyRight: r2(br.right), keyWidth: r2(br.width),
        valLeft: r2(sr.left), valTop: r2(sr.top),
        stackedPair: stacked,
        keyFace: fam(bcs), valFace: fam(scs),
        keySize: bcs.fontSize, valSize: scs.fontSize,
        keyColor: bcs.color, valColor: scs.color,
        keyBaseline: r2(kb), valBaseline: r2(vb),
        baselineDelta: stacked ? null : r2(Math.abs(kb - vb)),
        valLines: uniq(sRects.map((x) => x.y)).length,
      };
    }) : [];
    const heads = roEl ? [...roEl.querySelectorAll('.ro-head')].map((h) => h.textContent.trim()) : [];
    const roBox = roEl ? roEl.getBoundingClientRect() : null;
    const refWrap = document.querySelector('section.wrap');
    const refBox = refWrap ? refWrap.getBoundingClientRect() : null;
    const bandLeft = roBox ? r2(roBox.left) : null;
    const bodyLines = rows.map((r) => r.baselineDelta).filter((v) => v !== null);
    const colSpan = (f) => {
      const v = rows.map(f);
      return v.length ? r2(Math.max(...v) - Math.min(...v)) : null;
    };

    /* ---------- token discipline ---------- */
    const RED = ['rgb(200, 54, 26)', 'rgb(156, 42, 19)'];
    const redInk = [];
    for (const el of document.querySelectorAll('.beats *, .readout *')) {
      const cs = getComputedStyle(el);
      for (const prop of ['color', 'backgroundColor', 'borderTopColor', 'borderLeftColor', 'outlineColor']) {
        if (RED.includes(cs[prop])) {
          redInk.push({ tag: el.tagName, cls: el.getAttribute('class') || '', prop, value: cs[prop] });
        }
      }
    }
    const lum = (c) => {
      const m = /rgba?\((\d+),\s*(\d+),\s*(\d+)/.exec(c);
      if (!m) return null;
      const [r, g, b] = [+m[1], +m[2], +m[3]].map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); });
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    const contrast = [];
    for (const el of document.querySelectorAll('.beats .n, .beats h3, .beats p, .readout b, .readout span, .readout .ro-head')) {
      const cs = getComputedStyle(el);
      const la = lum(cs.color), lb = lum('rgb(238, 241, 243)');
      if (la === null || lb === null) continue;
      const ratio = (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
      contrast.push({ t: el.textContent.replace(/\s+/g, ' ').trim().slice(0, 34), size: cs.fontSize, color: cs.color, ratio: r2(ratio) });
    }

    return {
      fontsReady,
      horizontalOverflowPx: Math.max(0, de.scrollWidth - de.clientWidth),
      scrollWidth: de.scrollWidth, clientWidth: de.clientWidth,
      boxesOutsideViewport: scrollers,
      beats: {
        shape, cardCount: cards.length, rowsBandCount: bands.length,
        cardHeightDelta, bands, cards, ruleMismatches,
        textCollisions: collisions(beatsRects),
        textLineCount: beatsRects.length,
      },
      readout: {
        rowCount: rows.length, heads, rows,
        bandLeft, bandWidth: roBox ? r2(roBox.width) : null,
        sectionLeft: refBox ? r2(refBox.left) : null,
        bandLeftDelta: (bandLeft !== null && refBox) ? r2(Math.abs(bandLeft - refBox.left)) : null,
        keyLeftDelta: colSpan((r) => r.keyLeft),
        valLeftDelta: colSpan((r) => r.valLeft),
        maxBaselineDelta: bodyLines.length ? Math.max(...bodyLines) : null,
        stackedRows: rows.filter((r) => r.stackedPair).map((r) => r.key),
        measuredRows: rows.filter((r) => r.group === 'measured').map((r) => r.key),
        architectureRows: rows.filter((r) => r.group === 'architecture').map((r) => r.key),
        monoValues: rows.filter((r) => /Mono/.test(r.valFace)).length,
        sansValues: rows.filter((r) => /Sans/.test(r.valFace)).length,
        textCollisions: collisions(roRects),
        textLineCount: roRects.length,
      },
      redInk,
      redInkCount: redInk.length,
      contrastMin: contrast.length ? Math.min(...contrast.map((c) => c.ratio)) : null,
      contrastUnder45: contrast.filter((c) => c.ratio < 4.5),
    };
  }, width);

  if (WANT_PNG) {
    fs.mkdirSync(SHOTS, { recursive: true });
    const p = path.join(SHOTS, `beats-v3-${variant}-${width}.png`);
    await page.screenshot({ path: p, fullPage: true });
    data.png = p;
  }
  await page.close();
  return data;
}

async function launch() {
  const tries = [{ channel: 'chrome' }, { channel: 'chromium' }, {}];
  let last;
  for (const opt of tries) { try { return await chromium.launch(opt); } catch (e) { last = e; } }
  throw last;
}

const browser = await launch();
const pwPkg = JSON.parse(read(path.join(HARNESS, 'node_modules', 'playwright', 'package.json')));
const out = { playwright: pwPkg.version, results: {} };
const variants = VARIANT === 'both' ? ['v2', 'v3'] : [VARIANT];
for (const v of variants) {
  out.results[v] = {};
  for (const w of WIDTHS) out.results[v][w] = await measure(browser, v, w);
}
await browser.close();

/* ---------------------------------------------------------------- reporting */
const HARD = (d) => ({
  horizontalOverflowPx: d.horizontalOverflowPx,
  boxesOutsideViewport: d.boxesOutsideViewport.length,
  beatsCollisions: d.beats.textCollisions.length,
  readoutCollisions: d.readout.textCollisions.length,
  cardHeightDelta: d.beats.cardHeightDelta,
  separatorRuleMismatches: d.beats.ruleMismatches,
  redInk: d.redInkCount,
});

function reportCompact(v, w, d) {
  const L = [];
  const line = (k, val) => L.push('  ' + k.padEnd(26, '.') + ' ' + val);
  const b = d.beats, r = d.readout;
  L.push('');
  L.push(`===== ${v} @ ${w}px ` + '='.repeat(Math.max(4, 68 - String(w).length)));
  line('page horizontal overflow', `${d.horizontalOverflowPx} px   (scrollWidth ${d.scrollWidth} vs clientWidth ${d.clientWidth})`);
  line('component boxes outside', String(d.boxesOutsideViewport.length));
  line('fonts used', `Plex Mono ${d.fontsReady.plexMono ? 'ok' : 'MISSING'}, Plex Sans ${d.fontsReady.plexSans ? 'ok' : 'MISSING'}, Fraunces ${d.fontsReady.fraunces ? 'ok' : 'MISSING'}`);
  line('.beats shape', `${b.shape}  (${b.cardCount} cards in ${b.rowsBandCount} row band${b.rowsBandCount > 1 ? 's' : ''})`);
  line('.beats card heights', b.cards.map((c) => c.height.toFixed(2)).join('  '));
  line('tallest - shortest', `${b.cardHeightDelta} px`);
  line('.beats text collisions', `${b.textCollisions.length}   (${b.textLineCount} rendered line boxes tested)`);
  for (const band of b.bands) {
    line(`band [card ${band.cards.join(',')}]`, `height Δ ${band.cardHeightDelta} | rules ${band.rulesOnCards.length}/${band.rulesExpected} on card(s) ${band.rulesOnCards.join(',') || '-'} | top Δ ord ${band.ordinalTopDelta} h3 ${band.h3TopDelta} p ${band.pTopDelta}`);
    line('  ...baselines', `ordinal ${band.ordinalBaselineDelta} | h3 ${band.h3BaselineDelta} | body ${band.pBaselineDelta} | last-line-bottom ${band.lastInkBottomDelta}`);
  }
  for (const c of b.cards) {
    line(`card ${c.ordinal} "${c.title}"`, `h ${c.height.toFixed(2)}  textcol ${c.contentWidth.toFixed(1)}  title ${c.h3Lines} line(s)  body ${c.pChars}ch/${c.pLines} lines (lh ${c.pLineHeight})  lastInk ${c.lastInkBottom}  floor inset ${c.inkFloorInset}`
      + (c.sepX !== null ? `  sep hairline x=${c.sepX} (whitespace ${c.sepLeftWhitespace}/${c.sepRightWhitespace})` : '')
      + (c.borderLeftX !== null ? `  sep border x=${c.borderLeftX}` : ''));
  }
  line('.readout text collisions', `${r.textCollisions.length}   (${r.textLineCount} rendered line boxes tested)`);
  line('.readout rows', `${r.rowCount}   (${r.measuredRows.length} measured / ${r.architectureRows.length} architecture)`);
  line('.readout group heads', r.heads.join(' | '));
  line('.readout band left / width', `${r.bandLeft} / ${r.bandWidth}   (section.wrap left ${r.sectionLeft}, delta ${r.bandLeftDelta})`);
  line('key column', `x = ${r.rows.length ? r.rows[0].keyLeft.toFixed(1) : '-'}  width ${r.rows.length ? r.rows[0].keyWidth.toFixed(1) : '-'}  (row-to-row spread ${r.keyLeftDelta} px)`);
  line('value column', `x = ${r.rows.length ? r.rows[0].valLeft.toFixed(1) : '-'}  (row-to-row spread ${r.valLeftDelta} px)`);
  line('key/value baseline Δ', r.maxBaselineDelta === null ? 'n/a (pairs stacked)' : `${r.maxBaselineDelta} px (max over ${r.rowCount - r.stackedRows.length} side-by-side rows; stacked: ${r.stackedRows.length ? r.stackedRows.join(', ') : 'none'})`);
  line('value faces', `${r.monoValues} IBM Plex Mono (measured), ${r.sansValues} IBM Plex Sans (architecture)`);
  for (const row of r.rows) {
    line(`  ${row.key}`, `${row.group.padEnd(12)} key ${row.keySize} ${row.keyFace}  value ${row.valSize} ${row.valFace}  baselineΔ ${row.baselineDelta === null ? 'stacked' : row.baselineDelta}  value lines ${row.valLines}`);
  }
  line('red ink in components', `${d.redInkCount}` + (d.redInkCount ? '  ' + JSON.stringify(d.redInk.slice(0, 4)) : ''));
  line('contrast min on paper', `${d.contrastMin}` + (d.contrastUnder45.length ? `  UNDER 4.5: ${JSON.stringify(d.contrastUnder45)}` : '  (0 texts under 4.5)'));
  return L.join('\n');
}

console.log(`playwright ${out.playwright} · chromium channel=chrome · preview = harness/page.css` + (VARIANT === 'v3' ? ' + v3/beats.css' : ' (v2 baseline)') + ' + fragment');
for (const v of variants) for (const w of WIDTHS) console.log(reportCompact(v, w, out.results[v][w]));

console.log('\n===== HARD COUNTERS ' + '='.repeat(52));
for (const v of variants) for (const w of WIDTHS) {
  const h = HARD(out.results[v][w]);
  console.log(`${v} @ ${String(w).padStart(4)}px  overflow=${h.horizontalOverflowPx}px  outOfBox=${h.boxesOutsideViewport}`
    + `  beatsCollisions=${h.beatsCollisions}  readoutCollisions=${h.readoutCollisions}`
    + `  cardHeightDelta=${h.cardHeightDelta}  separatorRules=${h.separatorRuleMismatches}  redInk=${h.redInk}  shape=${out.results[v][w].beats.shape}`);
}
const fail = [];
for (const v of variants) for (const w of WIDTHS) {
  const h = HARD(out.results[v][w]);
  for (const [k, val] of Object.entries(h)) if (val !== 0) fail.push(`${v}@${w} ${k}=${val}`);
}
console.log(fail.length ? `\nNON-ZERO: ${fail.join(', ')}` : '\nall hard counters 0');
if (argv.includes('--json')) console.log('\n===== FULL JSON ' + '='.repeat(58) + '\n' + JSON.stringify(out, null, 1));

