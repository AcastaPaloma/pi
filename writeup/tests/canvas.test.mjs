import test from 'node:test';
import assert from 'node:assert/strict';
import { fitBounds, zoomAt, constrainCamera } from '../camera.mjs';
import { artwork, steps, details, annotations } from '../content.js';

test('every section fits desktop and phone viewports without being cut off', () => {
  for (const [width, height] of [[1080, 670], [620, 480], [390, 252], [320, 210]]) {
    for (const { bounds: [x, y, w, h] } of steps) {
      const c = fitBounds([x, y, w, h], width, height, 17);
      assert.ok(x * c.scale + c.x >= 16.99);
      assert.ok(y * c.scale + c.y >= 16.99);
      assert.ok((x + w) * c.scale + c.x <= width - 16.99);
      assert.ok((y + h) * c.scale + c.y <= height - 16.99);
    }
  }
});

test('zooming preserves the point under the pointer', () => {
  const camera = { x: -249, y: 53, scale: .63 }, point = { x: 417, y: 229 };
  for (const scale of [.12, .5, 1, 3]) {
    const next = zoomAt(camera, point, scale);
    assert.ok(Math.abs((point.x - camera.x) / camera.scale - (point.x - next.x) / next.scale) < 1e-8);
    assert.ok(Math.abs((point.y - camera.y) / camera.scale - (point.y - next.y) / next.scale) < 1e-8);
  }
});

test('extreme panning cannot lose the drawing entirely', () => {
  for (const sign of [-1, 1]) {
    const c = constrainCamera({ x: sign * 1e8, y: sign * 1e8, scale: .4 }, 900, 600, artwork.width, artwork.height);
    assert.ok(c.x < 900 && c.x + artwork.width * c.scale > 0);
    assert.ok(c.y < 600 && c.y + artwork.height * c.scale > 0);
  }
});

test('route and inspection references are unique, valid, and inside the artwork', () => {
  assert.equal(new Set(steps.map(s => s.id)).size, steps.length);
  assert.equal(new Set(details.map(d => d.id)).size, details.length);
  for (const item of [...steps, ...details]) {
    const [x, y, w, h] = item.bounds;
    assert.ok(w > 0 && h > 0 && x >= 0 && y >= 0, item.id);
    assert.ok(x + w <= artwork.width && y + h <= artwork.height, item.id);
  }
  for (const step of steps) for (const id of step.details) assert.ok(details.some(d => d.id === id), id);
  for (const detail of details) assert.ok(steps.some(s => s.id === detail.step), detail.id);
  assert.deepEqual(steps.filter(s => s.macro === 'ingestion').map(s => s.id),
    ['context', 'observation', 'subgoals', 'text', 'proprioception', 'together']);
});

test('chapters follow the reading order: start, inputs, model, losses, prompt writers', () => {
  const order = [...new Set(steps.map(s => s.macro))];
  assert.deepEqual(order, ['start', 'ingestion', 'decision', 'training', 'subgoal-policy', 'world-model']);
  assert.deepEqual(steps.filter(s => s.macro === 'start').map(s => s.id), ['preface', 'map', 'notation']);
  assert.deepEqual(steps.filter(s => s.macro === 'world-model').map(s => s.id), ['world-model']);
});

test('decision walkthrough follows the authored macro-to-layer sequence', () => {
  assert.deepEqual(steps.filter(s => ['decision', 'training'].includes(s.macro)).map(s => s.id),
    ['decision', 'vlm', 'flow', 'l1', 'l2', 'l33', 'l34', 'transformer',
      'eam-loss', 'eam-projection', 'flow-loss', 'vlm-loss', 'fast-input',
      'fast-transformer', 'token-probabilities', 'cross-entropy']);
  const vlm = steps.find(s => s.id === 'vlm'), flow = steps.find(s => s.id === 'flow');
  assert.deepEqual(vlm.frame.slice(1), flow.frame.slice(1), 'upper panels share camera height and scale');
  for (const step of steps) for (const id of step.details) {
    assert.equal(details.find(d => d.id === id).step, step.id, 'TOC detail belongs to its stop');
  }
});


test('every destination has written prose that keeps the supplied transcript verbatim and in order', async () => {
  const { readFile } = await import('node:fs/promises');
  const supplied = JSON.parse(await readFile(new URL('../author-copy.json', import.meta.url), 'utf8'));
  for (const item of [...steps, ...details]) assert.ok(annotations[item.id]?.length, item.id);
  assert.equal(Object.keys(annotations).length, steps.length + details.length);
  for (const [id, paragraphs] of Object.entries(supplied)) {
    const body = annotations[id], at = body.indexOf(paragraphs[0]);
    assert.ok(at >= 0, id);
    assert.deepEqual(body.slice(at, at + paragraphs.length), paragraphs, id);
  }
  // The preface is reserved for the author; the world model notes its unfinished diagram.
  const reserved = Object.entries(annotations).filter(([, blocks]) => blocks.some(b => b.placeholder)).map(([id]) => id);
  assert.deepEqual(reserved.sort(), ['preface', 'world-model']);
  assert.ok(annotations.preface.every(b => b.placeholder));
});

test('reading routes point at real sections', () => {
  const ids = new Set([...steps, ...details].map(x => x.id));
  const routed = steps.filter(s => s.routes);
  assert.ok(routed.length);
  for (const step of routed) for (const route of step.routes) {
    assert.ok(route.title && route.note && route.ids.length, route.title);
    for (const id of route.ids) assert.ok(ids.has(id), id);
  }
});

test('composed SVG layers and all component bounds fit the extended canvas', async () => {
  const { readFile } = await import('node:fs/promises');
  for (const layer of artwork.layers) {
    const source = await readFile(new URL('../' + layer.src, import.meta.url), 'utf8');
    assert.match(source, /<svg[\s>]/);
    assert.doesNotMatch(source, /<script[\s>]/);
  }
  for (const step of steps) for (const id of step.details) {
    assert.equal(details.find(item => item.id === id)?.step, step.id);
  }
});
