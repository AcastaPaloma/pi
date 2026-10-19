import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { annotations } from '../article.js';
import { steps, lanes, task, subtasks } from '../inference-content.js';
import { extraNodes } from '../results-runs.js';
import { translator, sentencePairs, plain } from '../i18n-core.mjs';
const read = name => JSON.parse(fs.readFileSync(new URL(`../${name}`, import.meta.url), 'utf8'));
const catalogs = ['ui','article-a','article-b','inference','results'].map(name => read(`locales/${name}.zh.json`));
const catalog = Object.assign({}, ...catalogs);
function strings(value) { return typeof value === 'string' ? [value] : value && typeof value === 'object' ? Object.values(value).flatMap(strings) : []; }
test('published training prose and inference blocks have exact, nonempty Chinese pairs', () => {
  const sources = [...strings(annotations), ...steps.flatMap(s => [s.title, ...(s.time ? [s.time] : []), ...strings(s.text)]), ...Object.values(lanes).map(l => l.title), task, ...subtasks];
  for (const source of sources) assert.ok(catalog[source]?.trim(), `Missing translation: ${source}`);
  const copy = read('text.json');
  function visit(node) { if (node.text) assert.ok(catalog[node.text]?.trim(), `Missing rich-text fragment: ${node.text}`); for (const child of node.children || []) visit(child); }
  Object.values(copy.sections).flat().forEach(visit);
  for (const node of extraNodes) for (const key of ['name','short','when','changed','result','next']) if (node[key]) assert.ok(catalog[node[key]], `Missing live result: ${node[key]}`);
});
test('language switching keeps unknown tokens, formulas and all English source intact', () => {
  const en = translator(catalog, 'en'), zh = translator(catalog, 'zh');
  for (const [source, target] of Object.entries(catalog)) { assert.equal(en(source), source); assert.equal(zh(source), target); }
  assert.equal(zh('x_control_lr5e6_s0'), 'x_control_lr5e6_s0');
  assert.equal(zh('Done in 4.2 s'), '4.2 秒完成');
});
test('sentence alignment falls back to the full paragraph when sentence counts differ', () => {
  assert.deepEqual(sentencePairs('One. Two.', '一。二。'), [{en:'One. ',zh:'一。'},{en:'Two.',zh:'二。'}]);
  assert.deepEqual(sentencePairs('One. Two.', '合成一句。'), [{en:'One. Two.',zh:'合成一句。'}]);
  assert.equal(plain('**π0.7** reads images.'), 'π0.7 reads images.');
});

test('sentence boundaries preserve bold delimiters and both exact source strings', () => {
  const en = '**Camera frames.** One image from each camera.', zh = '**摄像头图像。** 每个摄像头各一张。';
  const pairs = sentencePairs(en, zh);
  assert.equal(pairs.map(p => p.en).join(''), en);
  assert.equal(pairs.map(p => p.zh).join(''), zh);
  for (const pair of pairs) for (const text of Object.values(pair)) assert.equal((text.match(/\*\*/g) || []).length % 2, 0);
});
