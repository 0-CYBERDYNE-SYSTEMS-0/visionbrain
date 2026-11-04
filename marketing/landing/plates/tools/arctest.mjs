#!/usr/bin/env node
/**
 * arctest.mjs — measure the FILLED AREA of candidate lens paths (fixed: no image-load race).
 *
 * The v3 lens paths use the same circle centre for both arcs, so the arcs coincide and the
 * shape encloses no area. This measures that, then finds the flag combination that yields a real
 * intersection lens — decided by measurement, not by reasoning about arc flags.
 */
import { chromium } from 'playwright';

const CASES = {
  L1: { S: [403.88, 259.54], E: [408.12, 142.46], r: 150, d: 276.18 },
  L2: { S: [685.91, 254.59], E: [682.09, 147.41], r: 150, d: 280.18 },
};
const expected = (r, d) => 2 * r * r * Math.acos(d / (2 * r)) - (d / 2) * Math.sqrt(4 * r * r - d * d);

async function launch() {
  for (const o of [{ channel: 'chrome' }, { channel: 'chromium' }, {}]) {
    try { return await chromium.launch(o); } catch {}
  }
  throw new Error('no browser');
}

const browser = await launch();
try {
  const page = await browser.newPage({ viewport: { width: 1200, height: 480 } });

  console.log('correct intersection-lens area for these circle pairs:');
  for (const [k, c] of Object.entries(CASES)) console.log(`  ${k}: ${expected(c.r, c.d).toFixed(0)} u^2`);

  for (const [name, c] of Object.entries(CASES)) {
    const exp = expected(c.r, c.d);
    console.log(`\n${name}  (correct = ${exp.toFixed(0)} u^2)`);
    for (const la of [0, 1]) for (const sw of [0, 1]) {
      const d = `M${c.S[0]} ${c.S[1]} A${c.r} ${c.r} 0 0 0 ${c.E[0]} ${c.E[1]} `
              + `A${c.r} ${c.r} 0 ${la} ${sw} ${c.S[0]} ${c.S[1]} Z`;
      const area = await page.evaluate(async ({ d }) => {
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="420" viewBox="0 0 1120 420">`
                  + `<path d="${d}" fill="#ff0000" stroke="none"/></svg>`;
        const cv = document.createElement('canvas'); cv.width = 1120; cv.height = 420;
        const ctx = cv.getContext('2d');
        const img = new Image();
        await new Promise((res) => {
          const done = () => res();
          img.onload = done; img.onerror = done;
          setTimeout(done, 4000);
          img.src = 'data:image/svg+xml;base64,' + btoa(svg);
        });
        ctx.drawImage(img, 0, 0);
        const px = ctx.getImageData(0, 0, 1120, 420).data;
        let n = 0;
        for (let i = 3; i < px.length; i += 4) if (px[i] > 128) n++;
        return n;
      }, { d });
      const verdict = area < exp * 0.2 ? 'DEGENERATE (no enclosed area)'
        : Math.abs(area - exp) / exp < 0.15 ? 'CORRECT lens' : 'wrong shape';
      console.log(`  arc2 large-arc=${la} sweep=${sw}   area ${String(area).padStart(6)} u^2   ${verdict}`);
    }
  }
} finally {
  await browser.close();
}
