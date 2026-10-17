// Usage: node check_frontend.cjs PLAYWRIGHT_PACKAGE BASE_URL [VIDEO1 VIDEO2]
// Exercises browser behavior with controlled API responses, then checks live history/uploads.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.argv[2]);
const base = process.argv[3];
const videos = process.argv.slice(4);
const out = path.resolve('tmp/frontend-checks');
fs.mkdirSync(out, { recursive: true });

(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    const requests = [];
    const job = (id, verdict = 'no_violation') => ({
      job_id: id, status: 'done', verdict, violations: verdict === 'violation' ? [{
        type: 'SAFE_LEFT_OPEN', title: 'Сейф оставлен открытым', timecode_sec: 2,
        description: 'В кадре виден открытый сейф. Требуется проверка сотрудником.',
        law_short: 'Правила кассовых операций № 231', law_url: 'https://adilet.zan.kz/rus/docs/V1900019680',
        rule_section: 'п. 80', law_about: 'Порядок хранения наличных денег и ключей.',
        evidence_frames: [{ timestamp_sec: 2, image_base64: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6CnQAAAAASUVORK5CYII=' }],
      }] : [],
    });
    await page.route('**/ui/**', async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      requests.push({ path: url.pathname, method: request.method(), body: request.postData() || '' });
      assert.equal(request.headers()['x-api-key'], undefined);
      let data;
      if (url.pathname === '/ui/history') data = { jobs: [
        { job_id: 'past', filename: 'История.mov', status: 'done', verdict: 'violation', created_at: 1792240000 },
        { job_id: 'failed', filename: 'Ошибка.mov', status: 'failed', error: 'Файл повреждён', created_at: 1792240000 },
      ] };
      else if (url.pathname === '/ui/analyze/batch') data = { batch_id: 'batch', jobs: [
        { job_id: 'batch-good', filename: 'good.mov', status: 'processing' },
        { job_id: null, filename: 'bad.mov', status: 'rejected', error: 'Файл повреждён' },
      ] };
      else if (url.pathname === '/ui/analyze') data = { job_id: 'single', status: 'processing' };
      else if (url.pathname === '/ui/jobs/failed') data = { job_id: 'failed', status: 'failed', error: 'Файл повреждён' };
      else data = job(url.pathname.split('/').pop(), url.pathname.endsWith('/past') ? 'violation' : 'no_violation');
      await route.fulfill({ json: data });
    });
    await page.route('**/media/**', (route) => route.fulfill({ status: 404, body: '' }));
    await page.goto(base);
    await page.getByRole('button', { name: /История проверок/ }).click();
    await page.locator('.history-item').first().click();
    await page.locator('.verdict').waitFor();
    assert.equal(await page.locator('.law a').getAttribute('href'), 'https://adilet.zan.kz/rus/docs/V1900019680');
    assert.match(await page.locator('.law').innerText(), /п\. 80/);
    assert.match(await page.locator('.law').innerText(), /Порядок хранения/);
    await page.locator('.history-item').first().click();
    assert.equal(await page.locator('.task-card').count(), 1, 'Reopening history duplicated the task');
    await page.locator('.history-item').nth(1).click();
    await page.locator('.error').waitFor();
    assert.match(await page.locator('.error').innerText(), /Файл повреждён/);
    await page.getByRole('button', { name: 'Очистить список' }).click();
    await page.locator('input[type=file]').setInputFiles([
      { name: 'good.mov', mimeType: 'video/quicktime', buffer: Buffer.from('test') },
      { name: 'bad.mov', mimeType: 'video/quicktime', buffer: Buffer.from('test') },
    ]);
    await page.locator('.verdict').waitFor();
    assert.equal(await page.locator('.verdict h2').innerText(), 'Нарушение не обнаружено');
    assert.equal(requests.filter((r) => r.path === '/ui/analyze/batch').length, 1);
    assert.equal(requests.filter((r) => r.path === '/ui/analyze').length, 0);
    const batch = requests.find((r) => r.path === '/ui/analyze/batch');
    assert.equal((batch.body.match(/name="files"/g) || []).length, 2);
    await page.locator('.task-card').nth(1).click();
    await page.locator('.error').waitFor();
    assert.match(await page.locator('.error').innerText(), /Файл повреждён/);
    await page.getByRole('button', { name: 'Очистить список' }).click();
    await page.locator('input[type=file]').setInputFiles([
      { name: 'single.mov', mimeType: 'video/quicktime', buffer: Buffer.from('test') },
    ]);
    await page.locator('.verdict').waitFor();
    assert.equal(requests.filter((r) => r.path === '/ui/analyze').length, 1);
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    await page.screenshot({ path: path.join(out, 'mobile.png'), fullPage: true });
    assert.deepEqual(errors, []);
    console.log('PASS: history reopen/deduplication, failed history, law details, single upload, partial batch rejection, mobile layout, no browser key.');
    await page.close();

    const live = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const liveErrors = [];
    live.on('pageerror', (error) => liveErrors.push(error.message));
    await live.goto(base);
    const response = await live.request.get(new URL('/ui/history?limit=100', base).href);
    assert.equal(response.status(), 200);
    const history = (await response.json()).jobs;
    const selected = history.findIndex((item) => item.status === 'done' && item.verdict === 'violation');
    assert(selected >= 0, 'No completed live result to verify');
    await live.getByRole('button', { name: /История проверок/ }).click();
    await live.locator('.history-item').nth(selected).click();
    await live.locator('.verdict').waitFor();
    await live.waitForFunction(() => document.querySelector('video')?.readyState >= 1, null, { timeout: 30000 });
    assert(await live.locator('.evidence-frame').count() > 0);
    assert(await live.locator('.law a').count() > 0);
    await live.locator('.evidence-frame').first().click();
    await live.waitForFunction(() => {
      const video = document.querySelector('video');
      return video && !video.seeking && video.readyState >= 2 && video.currentTime > 0;
    }, null, { timeout: 30000 });
    await live.screenshot({ path: path.join(out, 'live-history.png'), fullPage: true });
    console.log('PASS: live history, result, preview metadata, evidence frames, law links and seeking.');

    if (videos.length) {
      await live.getByRole('button', { name: 'Очистить список' }).click();
      const uploads = [];
      live.on('request', (request) => {
        if (request.method() === 'POST' && request.url().includes('/ui/analyze')) uploads.push(request.url());
      });
      const started = Date.now();
      await live.locator('input[type=file]').setInputFiles(videos);
      await live.waitForFunction((count) => {
        const cards = [...document.querySelectorAll('.task-card')];
        return cards.length === count && cards.every((card) => !card.querySelector('.status-processing'));
      }, videos.length, { timeout: 300000 });
      assert.equal(await live.locator('.task-card .status-failed').count(), 0, 'Live upload failed');
      assert.equal(uploads.length, 1, 'Files were not submitted as one batch');
      if (videos.length > 1) assert(uploads[0].endsWith('/ui/analyze/batch'));
      await live.screenshot({ path: path.join(out, 'live-batch.png'), fullPage: true });
      console.log(`PASS: ${videos.length} live videos completed in ${Math.round((Date.now() - started) / 1000)}s using one upload request.`);
    }
    assert.deepEqual(liveErrors, []);
    await live.close();
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
