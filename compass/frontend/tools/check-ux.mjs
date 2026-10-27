import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createEngine } from '../src/prototype/engine.js';

/**
 * Acceptance suite for the Compass UX refresh.
 *
 * Deliberately written as invariants over every screen, not as a list of
 * expected sentences: widening the refresh to an unlisted screen, breaking an
 * interaction or leaking unescaped input fails here. The screen list is the
 * only duplicated fact and it is checked against the engine itself.
 */

const screens = JSON.parse(fs.readFileSync(new URL('../src/data/screens.json', import.meta.url)));
const REFRESH = ['P03', 'P04', 'P05', 'P06', 'P10', 'P11', 'P12', 'P13', 'P16', 'P17', 'P19', 'P26', 'P27', 'D04'];

function mount(initial = 'P03') {
  const box = { html: '', page: initial };
  box.e = createEngine((html, page) => {
    box.html = html;
    box.page = page;
  }, initial);
  return box;
}

function eachScreen(box, fn) {
  for (const role of ['patient', 'doctor']) {
    box.e.action(`role:${role}`);
    for (const s of screens) {
      box.e.navigate(s.id);
      assert.equal(box.page, s.id, `navigate(${s.id}) landed on ${box.page}`);
      fn(s, role, box.html);
    }
  }
}

/* 1. The refresh wrapper appears exactly on the listed screens, exactly once. */
{
  const box = mount();
  let wrapped = 0;
  eachScreen(box, (s, role, html) => {
    const count = (html.match(/class="rz-refresh rz-screen-/g) || []).length;
    if (REFRESH.includes(s.id)) {
      assert.equal(count, 1, `${s.id} (${role}): expected one refresh wrapper, got ${count}`);
      assert.ok(html.includes(`rz-screen-${s.id}`), `${s.id}: wrapper is not tagged with the screen id`);
      wrapped++;
    } else {
      assert.equal(count, 0, `${s.id} (${role}): refresh wrapper leaked onto an unlisted screen`);
    }
  });
  assert.equal(wrapped, REFRESH.length * 2, `wrapper count ${wrapped} != ${REFRESH.length} screens x 2 roles`);
  box.e.action('catalog');
  assert.ok(!box.html.includes('rz-refresh'), 'the screen catalog must never be wrapped');
  box.e.dispose();
}

/* 2. P08 keeps the real uploader and is left outside the refresh entirely. */
{
  const box = mount();
  box.e.navigate('P08');
  assert.ok(box.html.includes('data-react="depression-analyzer"'), 'P08 must mount the React analyzer');
  assert.ok(!box.html.includes('rz-refresh'), 'P08 must not be wrapped by the refresh');
  for (const block of ['rz-welcome', 'rz-media-panel', 'rz-record-btn', 'rz-analyze-link']) {
    assert.ok(!box.html.includes(block), `P08 markup picked up the refresh block ${block}`);
  }
  assert.ok(!box.html.includes('type="file"'), 'the engine must not re-implement the uploader');
  assert.ok(!box.html.includes('Записать аудио'), 'P08 must not render the diary record buttons');
  box.e.dispose();
}

/* 3. The emergency entry point exists on refreshed patient screens only.
   Navigation inside the engine also switches the role by screen prefix, so a
   doctor screen reached from the patient side is already the doctor's view. */
{
  const box = mount();
  box.e.action('role:patient');
  for (const s of screens) {
    box.e.navigate(s.id);
    assert.equal(
      box.html.includes('Экстренная помощь'),
      REFRESH.includes(s.id) && s.id[0] === 'P',
      `${s.id}: emergency header button must follow the refresh list and the patient role`,
    );
  }
  box.e.navigate('P03');
  assert.ok(box.html.includes('data-action="nav:P22"'), 'P03: emergency button must open the urgent-help screen');
  box.e.action('role:doctor');
  box.e.navigate('D04');
  assert.ok(!box.html.includes('Экстренная помощь'), 'doctor screens must not borrow the patient emergency button');
  box.e.dispose();
}

/* 4. Record entry points on the refreshed home and diary. */
{
  const box = mount();
  for (const id of ['P03', 'P04']) {
    box.e.navigate(id);
    const buttons = box.html.match(/<a href="#\/capture" class="rz-record-btn"/g) || [];
    assert.equal(buttons.length, 2, `${id}: expected audio and video entry points`);
    assert.ok(box.html.includes('aria-describedby="rz-format-hint"'), `${id}: record buttons lost the format hint`);
    assert.ok(box.html.includes('data-action="nav:P08"'), `${id}: missing the link into P08`);
  }
  box.e.dispose();
}

/* 5. No dead controls: every action emitted on any screen is handled. */
{
  const box = mount();
  const emitted = new Map();
  eachScreen(box, (s, role, html) => {
    for (const m of html.matchAll(/data-action="([^"]+)"/g)) {
      if (!emitted.has(m[1])) emitted.set(m[1], s.id);
    }
  });
  assert.ok(emitted.size > 60, `suspiciously few interactions collected: ${emitted.size}`);
  const broken = [];
  for (const [action, id] of emitted) {
    box.e.navigate(id);
    try {
      box.e.action(action);
    } catch (error) {
      broken.push(`${action} (${id}): ${error.message}`);
    }
  }
  box.e.dispose();
  assert.deepEqual(broken, [], `unhandled interactions:\n  ${broken.join('\n  ')}`);
}

/* 6. Hostile diary input survives a round trip escaped, never as markup. */
{
  const box = mount();
  const hostile = {
    diaryText: '<script>alert(1)</script> "кавычки" & <img src=x onerror=alert(2)>',
    sleep: '7" onfocus="alert(3)"',
    mood: 9,
    anxiety: 1,
  };
  box.e.navigate('P04');
  for (const [key, value] of Object.entries(hostile)) box.e.bind(`form.${key}`, value);
  box.e.navigate('P05');
  box.e.navigate('P04');
  assert.ok(box.html.includes('&lt;script&gt;alert(1)&lt;/script&gt;'), 'diary text is not escaped');
  assert.ok(box.html.includes('&amp; &lt;img src=x onerror=alert(2)&gt;'), 'mixed hostile text is not escaped');
  assert.ok(box.html.includes('7&quot; onfocus=&quot;alert(3)&quot;'), 'sleep value is not escaped');
  assert.ok(!/<script>alert\(1\)/i.test(box.html), 'raw <script> reached the page');
  assert.ok(!/<img src=x/i.test(box.html), 'raw <img> from user input reached the page');
  assert.ok(box.html.includes('id="rz-mood-value">9</output>'), 'mood is not reflected in the live output');
  assert.ok(box.html.includes('id="rz-anxiety-value">1</output>'), 'anxiety is not reflected in the live output');
  box.e.dispose();
}

/* 7. Skipping a PHQ-9 item is not a zero answer and blocks the score. */
{
  const box = mount();
  box.e.navigate('P16');
  box.e.action('qskip');
  assert.ok(box.html.includes('Этот вопрос останется без ответа'), 'skipping must ask for confirmation');
  assert.ok(box.html.includes('role="dialog"'), 'the confirmation must be a dialog');
  box.e.action('qskipconfirm');
  for (let i = 1; i < 9; i++) {
    box.e.action('answer:0');
    box.e.action('qnext');
  }
  assert.equal(box.page, 'P19', 'an incomplete questionnaire must land on P19');
  assert.ok(box.html.includes('Не заполнены вопросы: 1.'), 'P19 must name the skipped question');
  assert.ok(box.html.includes('Суммарный балл пока не рассчитывается'), 'P19 must not present a score');
  box.e.dispose();
}

/* 8. A complete questionnaire produces the real sum, not a placeholder. */
{
  const box = mount();
  box.e.navigate('P16');
  for (let i = 0; i < 9; i++) {
    box.e.action('answer:1');
    if (box.page === 'P21') break;
    box.e.action('qnext');
  }
  // Item 9 is the safety item: answering it positively routes to the safety flow.
  assert.equal(box.page, 'P21', 'a positive safety item must escalate immediately');
  assert.ok(box.html.includes('Есть ли сейчас непосредственная опасность'), 'P21 must ask the direct safety question');
  assert.ok(!box.html.includes('из 27'), 'P21 must not present a questionnaire score next to the safety question');
  box.e.action('safety:no');
  assert.match(box.html, /rz-metric">9 <span>из 27/, 'the PHQ-9 sum must be 9 of 27');
  box.e.dispose();
}

/* 9. An active safety event outranks the clinical summary on D04. */
{
  const box = mount();
  box.e.action('role:doctor');
  box.e.navigate('D04');
  assert.ok(box.html.includes('Что изменилось'), 'D04 lost the pre-visit summary');
  box.e.action('role:patient');
  box.e.navigate('P16');
  for (let i = 0; i < 9; i++) {
    box.e.action('answer:1');
    if (box.page === 'P21') break;
    box.e.action('qnext');
  }
  box.e.action('safety:yes');
  assert.equal(box.page, 'P22', 'a confirmed danger must reach the emergency instructions');
  box.e.action('role:doctor');
  box.e.navigate('D04');
  const alert = box.html.indexOf('Требуется немедленная оценка безопасности');
  const summary = box.html.indexOf('rz-clinical-summary');
  assert.ok(alert > -1, 'D04 must show the active safety event');
  assert.ok(alert < summary, 'the safety event must be rendered above the summary');
  box.e.dispose();
}

/* 10. Delivery screens describe the demo, never a real delivery. */
{
  const box = mount();
  box.e.navigate('P10');
  assert.ok(box.html.includes('данные не передаются на сервер или врачу'), 'P10 must state that nothing is sent');
  box.e.control('variant', 'error');
  assert.ok(box.html.includes('Не удалось отправить'), 'the error variant must show the failure');
  assert.ok(!box.html.includes('rz-dot done'), 'a failed send must not mark any stage as done');
  box.e.control('variant', 'default');
  box.e.navigate('P11');
  assert.ok(box.html.includes('Реальная передача врачу не подключена'), 'P11 must state that delivery is not wired');
  for (const s of screens) {
    box.e.navigate(s.id);
    assert.ok(!box.html.includes('врач просмотрел'), `${s.id}: claims the doctor reviewed the record`);
  }
  box.e.dispose();
}

/* 11. Refreshed screens stay self-contained: no remote assets, no frames. */
{
  const box = mount();
  box.e.action('role:patient');
  for (const id of REFRESH) {
    box.e.navigate(id);
    assert.ok(
      !/<(img|iframe|video|audio|object|embed|link)\b|src="https?:|url\(https?:|@import/iu.test(box.html),
      `${id}: loads an external or embedded asset`,
    );
  }
  box.e.dispose();
}

/* 12. The stylesheet keeps the scope promise the whole "P08 is untouched" claim
   rests on: every selector of the refresh block is prefixed. Without this, a
   single unprefixed rule restyles the uploader and no other test notices. */
{
  const css = fs.readFileSync(new URL('../src/styles.css', import.meta.url), 'utf8');
  const MARK = '/* UX refresh, 2026-10-26.';
  const at = css.indexOf(MARK);
  assert.ok(at > 0, 'the refresh block marker is missing from src/styles.css');
  const PREFIX = '#razuma-ux-root .rz-refresh';
  const offenders = [];
  let checked = 0;
  for (const raw of css.slice(at + MARK.length).replace(/\/\*[\s\S]*?\*\//g, '').split('\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('/*') || line.startsWith('*') || line === '}' || line.startsWith('@media')) continue;
    const brace = line.indexOf('{');
    const selector = (brace >= 0 ? line.slice(0, brace) : line.includes(';') || line.startsWith('--') ? '' : line)
      .replace(/,$/, '')
      .trim();
    if (!selector) continue; // a declaration inside a block
    checked++;
    if (!selector.startsWith(PREFIX)) offenders.push(selector);
  }
  assert.ok(checked > 100, `stylesheet scan looks wrong: only ${checked} selector lines`);
  assert.deepEqual(offenders, [], `refresh rules escaping the ${PREFIX} scope:\n  ${offenders.join('\n  ')}`);
}

console.log(
  `PASS: ${screens.length} screens - refresh scoping, P08 isolation, emergency header, ` +
    `dead-control scan, escaping, PHQ-9 skip vs sum, safety priority, delivery honesty, self-contained screens.`,
);
