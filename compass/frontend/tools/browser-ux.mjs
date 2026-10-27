import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

/**
 * Browser acceptance for the Compass UX refresh, run against the built SPA
 * (frontend/dist/compass), not the dev server, so it checks the artifact that
 * actually ships.
 *
 * A browser is optional: with no Chromium and no CHROMIUM_PATH the suite prints
 * SKIP and exits 0, so CI without a browser does not fail. Install one with
 * `npx playwright install chromium`, or point CHROMIUM_PATH at any Chromium
 * build (Edge/Chrome). PLAYWRIGHT_PATH lets a host without the npm dependency
 * reuse another node_modules (e.g. \\wsl$\...\node_modules\playwright-core\index.js).
 */

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DIST = process.env.UX_DIST || path.resolve(HERE, '../../../frontend/dist/compass');
const REFRESH = ['P03', 'P04', 'P05', 'P06', 'P10', 'P11', 'P12', 'P13', 'P16', 'P17', 'P19', 'P26', 'P27', 'D04'];
const WIDTHS = [1440, 1024, 760, 390, 320];
const SHOTS = process.env.UX_SCREENSHOTS_DIR || path.join(os.tmpdir(), 'razuma-ux-screens');
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
};

if (!fs.existsSync(path.join(DIST, 'index.html'))) {
  console.log(`SKIP: no build at ${DIST} (run make build_compass)`);
  process.exit(0);
}

const serve = () =>
  new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const wanted = path.join(DIST, decodeURIComponent(req.url.split('?')[0]));
      const file = wanted.startsWith(DIST) && fs.existsSync(wanted) && fs.statSync(wanted).isFile()
        ? wanted
        : path.join(DIST, 'index.html');
      res.writeHead(200, { 'content-type': MIME[path.extname(file)] || 'application/octet-stream' });
      fs.createReadStream(file).pipe(res);
    });
    server.listen(0, '127.0.0.1', () => resolve(server));
  });

const loadPlaywright = async () => {
  const spec = process.env.PLAYWRIGHT_PATH ? pathToFileURL(process.env.PLAYWRIGHT_PATH).href : 'playwright';
  return (await import(spec)).default;
};

const main = async () => {
  let playwright;
  try {
    playwright = await loadPlaywright();
  } catch {
    console.log('SKIP: playwright is not installed (npm ci, or set PLAYWRIGHT_PATH)');
    return 0;
  }
  // A launch failure is a real failure, not a skip: a broken CHROMIUM_PATH, a
  // sandbox violation and a missing browser must not look like "no browser".
  const browser = await playwright.chromium
    .launch({ executablePath: process.env.CHROMIUM_PATH || undefined, headless: true, args: ['--no-sandbox'] })
    .catch((error) => {
      throw new Error(
        `cannot launch Chromium: ${error.message}\n` +
          'Install one (npx playwright install chromium) or set CHROMIUM_PATH to a Chromium/Edge executable.',
      );
    });

  const server = await serve();
  const base = `http://127.0.0.1:${server.address().port}/`;
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const problems = [];
  page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
  page.on('console', (m) => m.type() === 'error' && problems.push(`console.error: ${m.text()}`));

  const open = async (id) => {
    await page.goto(`${base}#/screens/${id}`, { waitUntil: 'networkidle' });
    await page.locator('h1').first().waitFor();
  };

  fs.mkdirSync(SHOTS, { recursive: true });
  const measured = { renders: 0 };
  const overflow = [];
  const narrowTargets = [];

  for (const width of WIDTHS) {
    await page.setViewportSize({ width, height: width <= 390 ? 844 : 900 });
    for (const id of REFRESH) {
      await open(id);
      const over = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
      if (over > 0) overflow.push(`${id} @${width}: +${over}px`);
      measured.renders++;
      assert.equal(await page.locator('.rz-refresh').count(), 1, `${id} @${width}: expected one refresh root`);
      if (width === 390) {
        const small = await page.evaluate(() =>
          [...document.querySelectorAll('.rz-main button, .rz-main a')]
            .filter((el) => el.getClientRects().length)
            .map((el) => ({ el, r: el.getBoundingClientRect() }))
            .filter(({ r }) => r.height < 40 && r.width > 0)
            .map(({ el, r }) => `${el.className || el.tagName} ${Math.round(r.height)}px "${el.textContent.trim().slice(0, 28)}"`),
        );
        for (const item of small) narrowTargets.push(`${id}: ${item}`);
        await page.screenshot({ path: path.join(SHOTS, `${id}-390.png`) });
      }
    }
  }
  assert.deepEqual(overflow, [], `horizontal overflow:\n  ${overflow.join('\n  ')}`);
  assert.deepEqual(narrowTargets, [], `tap targets under 40px at 390:\n  ${narrowTargets.join('\n  ')}`);

  /* Both record entry points must be reachable without scrolling on a phone. */
  await page.setViewportSize({ width: 390, height: 844 });
  for (const id of ['P03', 'P04']) {
    await open(id);
    for (const name of ['Записать аудио', 'Записать видео']) {
      const box = await page.getByRole('link', { name, exact: true }).boundingBox();
      assert.ok(box && box.y >= 0 && box.y + box.height <= 844, `${id}: "${name}" is not in the first phone screen`);
    }
  }

  /* The slider must update its number while dragging, without losing focus. */
  await open('P04');
  const mood = page.locator('input[data-range="mood"]');
  await mood.focus();
  const before = await mood.inputValue();
  await mood.press('ArrowRight');
  const after = await mood.inputValue();
  assert.notEqual(before, after, 'arrow key did not move the mood slider');
  assert.equal((await page.locator('#rz-mood-value').textContent()).trim(), after, 'the slider output is stale');
  assert.ok(await mood.evaluate((el) => el === document.activeElement), 'the slider lost focus on input');

  /* The diary keeps what was typed after leaving the step and coming back. */
  const text = `Проверка сохранения ${Date.now()}`;
  await page.locator('textarea[data-bind="form.diaryText"]').fill(text);
  await page.goto(`${base}#/screens/P05`, { waitUntil: 'networkidle' });
  await page.locator('h1').first().waitFor();
  await open('P04');
  assert.equal(await page.locator('textarea[data-bind="form.diaryText"]').inputValue(), text, 'the diary lost its text');

  /* The refreshed home links into the untouched P08 uploader. */
  await open('P03');
  await page.locator('[data-action="nav:P08"]').first().click();
  await page.waitForURL('**/screens/P08');
  await page.getByRole('button', { name: 'Выбрать файл', exact: true }).waitFor();
  assert.equal(await page.locator('.rz-refresh').count(), 0, 'P08 must stay outside the refresh');
  assert.equal(await page.locator('.rz-analyzer').count(), 1, 'P08 lost the analyzer');
  assert.ok(!(await page.getByRole('button', { name: 'Экстренная помощь' }).count()), 'P08 must not show the refresh header button');
  await page.screenshot({ path: path.join(SHOTS, 'P08-after-entry.png') });

  assert.deepEqual(problems, [], `page problems:\n  ${problems.join('\n  ')}`);
  await browser.close();
  server.close();

  console.log(
    `PASS: ${REFRESH.length} refreshed screens x ${WIDTHS.length} widths (${measured.renders} renders) - no overflow, ` +
      'tap targets >= 40px at 390, record entry points above the fold, live slider with focus kept, ' +
      'form persistence, P08 untouched.',
  );
  console.log(`Screenshots: ${SHOTS}`);
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
