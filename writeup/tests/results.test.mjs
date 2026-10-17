import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, statSync } from 'node:fs';
import { nodes, tasks } from '../results-content.js';

const here = new URL('..', import.meta.url);

test('the experiment tree is well formed', () => {
  const ids = new Set();
  assert.equal(nodes[0].depth, 0, 'the tree starts on the main line');
  for (const node of nodes) {
    assert.ok(!ids.has(node.id), `duplicate node id ${node.id}`);
    ids.add(node.id);
    assert.match(node.id, /^[a-z0-9-]+$/, `${node.id}: ids are used in URLs`);
    assert.ok(['main', 'dead', 'running'].includes(node.kind), `${node.id}: unknown kind ${node.kind}`);
    assert.ok([0, 1].includes(node.depth), `${node.id}: depth is 0 or 1`);
    for (const key of ['name', 'when', 'changed', 'result', 'next']) assert.ok(node[key], `${node.id}: missing ${key}`);
    if (node.score) {
      assert.ok(Number.isInteger(node.score.value) && node.score.value >= 0, `${node.id}: score value`);
      assert.ok(node.score.value <= node.score.of, `${node.id}: score above its maximum`);
    }
  }
  assert.ok(nodes.some(n => n.kind === 'main' && n.clips.length), 'some kept experiment has a video (the start page)');
});

test('every clip names a known task, has sane numbers, and its files exist and stream', () => {
  for (const node of nodes) {
    for (const clip of node.clips) {
      const name = `${node.id}/${clip.video}`;
      assert.ok(tasks[clip.task], `${name}: unknown task ${clip.task}`);
      assert.equal(typeof clip.success, 'boolean', `${name}: success`);
      assert.ok(clip.layout >= 1 && clip.layout <= 30, `${name}: layout is one of the 30 test layouts`);
      if (clip.rate !== undefined) assert.ok(Number.isInteger(clip.rate) && clip.rate <= (clip.of || 30), `${name}: rate`);
      assert.match(clip.video, /^assets\/results\/[\w.-]+\.mp4$/);
      assert.match(clip.poster, /^assets\/results\/[\w.-]+\.jpg$/);
      for (const path of [clip.video, clip.poster]) {
        const file = new URL(path, here);
        assert.ok(existsSync(file), `${node.id}: missing ${path}`);
        assert.ok(statSync(file).size < 8e6, `${node.id}: ${path} is over 8 MB`);
      }
    }
  }
});
