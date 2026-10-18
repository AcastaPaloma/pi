import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, statSync } from 'node:fs';
import { nodes, tasks, lossPlan, treeRoots } from '../results-content.js';
import { lineageParents } from '../results-tree.js';
import { loss } from '../results-loss.js';
import * as gen from '../results-runs.js';
const { runs, updated } = gen;

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

test('queue-derived nodes and replicates attach to real nodes', () => {
  const ids = new Set(nodes.map(n => n.id));
  for (const n of gen.extraNodes || []) {
    assert.match(n.id, /^[a-z0-9-]+$/, `${n.id}: ids are used in URLs`);
    assert.ok(!ids.has(n.id), `${n.id}: collides with a hand-written node`);
    assert.ok(n.name && n.changed && n.when, `${n.id}: needs a title, a note and a date`);
    assert.ok(ids.has(n.parent), `${n.id}: parent ${n.parent} is not a node`);
    for (const run of n.runs) assert.ok(runs[run], `${n.id}: run ${run} missing from the generated data`);
  }
  for (const [parent, list] of Object.entries(gen.replicates || {})) {
    assert.ok(ids.has(parent), `replicates attach to an existing node, not ${parent}`);
    for (const run of list) assert.match(run, /^x_[a-z0-9_]+$/);
  }
});

test('lighting-and-colour rows name real models with sane totals', () => {
  const robust = gen.robust || { rows: [], pending: [] };
  const known = new Set(nodes.flatMap(n => [...(n.runs || []), ...((lossPlan[n.id] || {}).lines || []).map(l => l[0])]));
  for (const id of Object.values(gen.replicates || {}).flat()) known.add(id);
  for (const n of gen.extraNodes || []) n.runs.forEach(r => known.add(r));
  assert.ok(nodes.some(n => n.robust), 'the tree has the lighting-and-colour node');
  for (const r of robust.rows) {
    assert.ok(known.has(r.run), `robust row ${r.run} is not shown by any node`);
    assert.equal(r.tasks.length, 4);
    assert.equal(r.total, r.tasks.reduce((a, b) => a + b, 0));
    assert.ok(r.total <= 120 && r.tasks.every(t => t >= 0 && t <= 30));
    if (r.plain !== null && r.plain !== undefined) assert.ok(r.plain >= 0 && r.plain <= 120);
  }
});

test('every node hangs from a parent in the lineage tree', () => {
  const extra = (gen.extraNodes || []).map(n => ({ ...n, depth: 1, clips: [] }));
  const all = [...nodes, ...extra];
  const parents = lineageParents(all, gen.lineage || {}, lossPlan, treeRoots);
  const ids = new Set([...all.map(n => n.id), ...treeRoots.map(r => r.id)]);
  for (const n of all) {
    const p = parents.get(n.id);
    assert.ok(p, `${n.id}: no lineage parent (add 'from' or a resume record)`);
    assert.ok(ids.has(p.id) && p.id !== n.id, `${n.id}: parent ${p.id} does not resolve`);
  }
  const seen = new Set();  // no cycles: walking up from any node reaches a root
  for (const n of all) {
    let id = n.id, hops = 0;
    while (parents.get(id) && hops++ < all.length) id = parents.get(id).id;
    assert.ok(treeRoots.some(r => r.id === id), `${n.id}: walks up to ${id}, not a starting point`);
    seen.add(id);
  }
});
