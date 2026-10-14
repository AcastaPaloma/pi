import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { steps, details, annotations } from '../content.js';
import { buildReadingEntries, readingIndex, placeholder } from '../reading.mjs';

const entries = buildReadingEntries(steps, details, annotations);

test('every canvas destination has one reading passage in step/component order', () => {
  assert.equal(entries.length, steps.length + details.length);
  assert.equal(new Set(entries.map(e => e.id)).size, entries.length);
  for (const [index, step] of steps.entries()) {
    assert.deepEqual(entries.filter(e => e.stepIndex === index).map(e => e.id), [step.id, ...step.details]);
  }
});

test('author prose appears verbatim and lorem ipsum fills only unwritten destinations', async () => {
  const supplied = JSON.parse(await readFile(new URL('../author-copy.json', import.meta.url), 'utf8'));
  for (const entry of entries) {
    if (supplied[entry.id]) {
      assert.equal(entry.isPlaceholder, false);
      assert.deepEqual(entry.paragraphs, supplied[entry.id]);
    } else {
      assert.equal(entry.isPlaceholder, true);
      assert.ok(entry.paragraphs.every(p => placeholder.includes(p)));
    }
  }
  assert.equal(entries.filter(e => !e.isPlaceholder).length, Object.keys(supplied).length);
});

test('reading line remains deterministic for long passages, boundaries, and reverse scrolling', () => {
  const tops = [32, 420, 1600, 1880];
  assert.equal(readingIndex(tops, 0, 100), 0);
  assert.equal(readingIndex(tops, 320, 100), 1);
  assert.equal(readingIndex(tops, 1250, 100), 1);
  assert.equal(readingIndex(tops, 1500, 100), 2);
  assert.equal(readingIndex(tops, 1900, 100), 3);
  assert.equal(readingIndex(tops, 250, 100), 0);
  for (const [index, top] of tops.entries()) assert.equal(readingIndex(tops, Math.max(0, top - 100 + 4), 100), index);
});

test('subgoal policy is segmented from observation to cross-entropy', () => {
  assert.deepEqual(steps.filter(s => s.macro === 'subgoal-policy').map(s => s.id),
    ['subgoal-policy', 'sg-observation', 'sg-vision', 'sg-text', 'sg-context', 'sg-next-subtask', 'sg-prediction']);
  assert.ok(entries.some(e => e.id === 'sg-cross-entropy' && e.detail === 'sg-cross-entropy'));
});
