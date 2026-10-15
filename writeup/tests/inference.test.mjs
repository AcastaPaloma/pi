import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { drawing, steps, lanes, subtasks } from '../inference-content.js';

const BEATS = ['establish', 'capture-full', 'capture-quick', 'subtask', 'subgoal', 'subgoal-slow', 'subgoal-land', 'chunk', 'chunk-overlap', 'execute', 'execute-close', 'splice'];

test('every inference block has text, a known lane and beat, and bounds inside the drawing', () => {
  assert.equal(new Set(steps.map(s => s.id)).size, steps.length);
  for (const step of steps) {
    const [x, y, w, h] = step.bounds;
    assert.ok(w > 0 && h > 0 && x >= 0 && y >= 0 && x + w <= drawing.width + 1 && y + h <= drawing.height + 1, step.id);
    assert.ok(step.lane === null || Object.hasOwn(lanes, step.lane), step.id);
    assert.ok(BEATS.includes(step.stage.beat), step.id);
    assert.ok(step.stage.dwell > 0 && step.text.length, step.id);
  }
});

test('within one loop the robot only moves forward, and chunks cover the motion they plan', () => {
  let tau = 0, chunk = null;
  for (const step of steps) {
    const s = step.stage;
    assert.ok(s.tau >= tau - 1e-9, `${step.id} moves backwards`);
    tau = s.tau;
    if (s.chunk) chunk = s.chunk;
    if (s.splice) chunk = steps.find(x => x.stage.pending).stage.pending;
    if (chunk && ['execute', 'execute-close', 'splice'].includes(s.beat)) assert.ok(s.tau <= chunk[1] + 1e-9, `${step.id} runs past its chunk`);
  }
  assert.equal(steps.at(-1).stage.tau, 2, 'a loop completes two subtasks');
  assert.ok(subtasks.length % 2 === 0, 'loops pair subtasks');
});

test('the vendored SO-101 asset is well formed', async () => {
  const meta = JSON.parse(await readFile(new URL('../assets/so101/so101.json', import.meta.url)));
  const bin = await readFile(new URL('../assets/so101/so101.bin', import.meta.url));
  for (const [name, m] of Object.entries(meta.meshes)) {
    for (const [key, size] of [['position', 6], ['normal', 3], ['index', m.wideIndex ? 12 : 6]]) {
      const [offset, length] = m[key];
      assert.equal(offset % 4, 0, `${name} ${key} alignment`);
      assert.equal(length, (key === 'index' ? m.triangles : m.vertices) * size, `${name} ${key} length`);
      assert.ok(offset + length <= bin.length, `${name} ${key} range`);
    }
  }
});

test('every scripted waypoint is reachable by the SO-101 within 2 mm', async () => {
  const dir = new URL('../', import.meta.url);
  const THREE = await import(new URL('vendor/three.module.min.js', dir));
  const { kinematics: K } = await import(new URL('robot-stage.js', dir));
  const fetcher = async path => { const b = await readFile(new URL(path, dir)); return { ok: true, json: async () => JSON.parse(b), arrayBuffer: async () => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) }; };
  const asset = await K.loadSO101('assets/so101/', fetcher), materials = K.materialsFor(), scene = new THREE.Scene();
  const arms = { left: K.buildArm(asset, 'left', materials), right: K.buildArm(asset, 'right', materials) };
  scene.add(arms.left.root, arms.right.root);
  K.solveScripts(arms);
  for (const [i, script] of K.SCRIPTS.entries()) script.errors.forEach((e, j) => assert.ok(e < .002, `script ${i} waypoint ${j}: ${(e * 1000).toFixed(1)} mm`));
});
