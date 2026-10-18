import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, statSync } from 'node:fs';
import { nodes, tasks, lossPlan } from '../results-content.js';
import { loss } from '../results-loss.js';
import { runs, updated } from '../results-runs.js';

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
    for (const run of node.runs || []) assert.match(run, /^x_[a-z0-9_]+$/, `${node.id}: run id ${run}`);
    assert.ok(!(node.runs && node.score), `${node.id}: a node with runs takes its score from results-runs.js`);
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

test('the generated run data is well formed and its clips exist', () => {
  assert.equal(typeof updated, 'string');
  for (const [id, run] of Object.entries(runs)) {
    for (const score of run.scores) {
      assert.equal(score.tasks.length, 4, `${id}: four task scores`);
      assert.equal(score.tasks.reduce((a, b) => a + b, 0), score.total, `${id}: tasks add up to the total`);
      assert.ok(score.total <= 120 && Number.isInteger(score.step), `${id}: score`);
    }
    for (const clip of run.clips) {
      assert.ok(tasks[clip.task], `${id}: unknown task ${clip.task}`);
      for (const path of [clip.video, clip.poster]) {
        const file = new URL(path, here);
        assert.ok(existsSync(file), `${id}: missing ${path}`);
        assert.ok(statSync(file).size < 8e6, `${id}: ${path} is over 8 MB`);
      }
    }
  }
});

test('loss curves: the plan names real nodes and runs, and every curve is sound and small', () => {
  const ids = new Set(nodes.map(n => n.id));
  for (const [id, plan] of Object.entries(lossPlan)) {
    assert.ok(ids.has(id), `lossPlan names unknown node ${id}`);
    for (const [run, label] of [...(plan.lines || []), ...(plan.ref || [])]) {
      assert.ok(typeof label === 'string' && label, `${id}: ${run} needs a label`);
      assert.ok(loss[run] || (runs[run] && runs[run].state !== 'done'), `${id}: no loss curve for finished run ${run}`);
    }
  }
  for (const node of nodes) {
    const plan = lossPlan[node.id] || {};
    assert.ok(plan.note || plan.lines || node.runs, `${node.id}: the Loss view needs curves or a note`);
  }
  for (const [run, curve] of Object.entries(loss)) {
    assert.ok(curve.s.length >= 3 && curve.s.length <= 400, `${run}: ${curve.s.length} points`);
    assert.equal(curve.f.length, curve.s.length, `${run}: flow loss length`);
    if (curve.c) assert.equal(curve.c.length, curve.s.length, `${run}: FAST loss length`);
    for (let i = 1; i < curve.s.length; i++) assert.ok(curve.s[i] > curve.s[i - 1], `${run}: steps must increase`);
    assert.ok(curve.f.every(v => Number.isFinite(v) && v >= 0), `${run}: finite, non-negative losses`);
  }
  assert.ok(statSync(new URL('results-loss.js', here)).size < 1e6, 'results-loss.js stays under 1 MB');
});
