import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, statSync } from 'node:fs';
import { clips, models, tasks } from '../results-content.js';

const here = new URL('..', import.meta.url);

test('every clip names a known task and model and has sane numbers', () => {
  const ids = new Set();
  for (const clip of clips) {
    assert.ok(!ids.has(clip.id), `duplicate clip id ${clip.id}`);
    ids.add(clip.id);
    assert.ok(tasks[clip.task], `${clip.id}: unknown task ${clip.task}`);
    assert.ok(models[clip.model], `${clip.id}: unknown model ${clip.model}`);
    assert.ok(Number.isInteger(clip.rate) && clip.rate >= 0 && clip.rate <= 30, `${clip.id}: rate out of 30`);
    assert.ok(clip.layout >= 1 && clip.layout <= 30, `${clip.id}: layout is one of the 30 test layouts`);
    assert.equal(typeof clip.success, 'boolean');
  }
});

test('every clip video and poster exists and is small enough to stream', () => {
  for (const clip of clips) {
    for (const path of [clip.video, clip.poster]) {
      const file = new URL(path, here);
      assert.ok(existsSync(file), `${clip.id}: missing ${path}`);
      assert.ok(statSync(file).size < 8e6, `${clip.id}: ${path} is over 8 MB`);
    }
    assert.match(clip.video, /^assets\/results\/[\w.-]+\.mp4$/);
    assert.match(clip.poster, /^assets\/results\/[\w.-]+\.jpg$/);
  }
});
