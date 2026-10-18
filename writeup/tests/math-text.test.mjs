import test from 'node:test';
import assert from 'node:assert/strict';
import { segment, compileReferences } from '../math-text.mjs';
import { annotations } from '../article.js';
import { references, steps, details } from '../content.js';

const math = text => segment(text).filter(part => part.type === 'math').map(part => part.source);
const prose = () => Object.values(annotations).flat().flatMap(block => typeof block === 'string' ? [block] : block.list ?? [block.note ?? block.placeholder ?? '']);

test('every passage splits into segments that join back to its exact source', () => {
  const refs = compileReferences(references);
  for (const text of prose()) assert.equal(segment(text, refs).map(part => part.source).join(''), text);
});

test('subscripted names, shapes, equations, and layers are math', () => {
  assert.deepEqual(math('Q_t is the state now, and Q_{t−5} … Q_{t−1} are earlier'), ['Q_t', 'Q_{t−5} … Q_{t−1}']);
  assert.deepEqual(math('shaped [B, 6, d_q], where d_q is the number'), ['[B, 6, d_q]', 'd_q']);
  assert.deepEqual(math('they form one tensor, X_ctx, shown'), ['X_ctx']);
  assert.deepEqual(math('Linear(d_q, 2560) is applied: [B, 6, d_q] → [B, 6, 2560].'), ['Linear(d_q, 2560)', '[B, 6, d_q] → [B, 6, 2560]']);
  assert.deepEqual(math('The noisy input is A_τ = (1 − τ)·A* + τ·ε, with shape'), ['A_τ = (1 − τ)·A* + τ·ε']);
  assert.deepEqual(math('We flatten the 14 × 14 × 3 = 588 values'), ['14 × 14 × 3 = 588']);
  assert.deepEqual(math('Loss = (−ln 0.80 − ln 0.60) / 2 ≈ 0.367. In'), ['Loss = (−ln 0.80 − ln 0.60) / 2 ≈ 0.367']);
  assert.deepEqual(math('each A ← A − 0.2 · (predicted velocity), until'), ['A ← A − 0.2 · (predicted velocity)']);
  assert.deepEqual(math('l_j = −log p(z_j | X_ctx, z_{<j}): the'), ['l_j = −log p(z_j | X_ctx, z_{<j})']);
});

test('ordinary prose stays text', () => {
  for (const text of [
    'A training example takes a 2 × 2 kernel', 'From Option B: stop the flow gradients.',
    'It’s the model’s turn, e.g. a robot', 'π0.7 has about 5 billion parameters on one H100 GPU',
    'two 6-joint arms with a 1-DoF gripper', 'the drawing labels the step “Linear”', 'a log-softmax, which turns'
  ]) assert.deepEqual(math(text).filter(source => !/×/.test(source)), [], text);
  assert.deepEqual(math('takes a 2 × 2 kernel'), ['2 × 2']);
});

test('cross-references point at real passages, once per passage and never at their own', () => {
  const ids = new Set([...steps, ...details].map(item => item.id));
  for (const ref of references) assert.ok(ids.has(ref.target), ref.target);
  const refs = compileReferences(references);
  const links = text => segment(text, refs, new Set(['flow-loss'])).filter(part => part.type === 'ref').map(part => part.target);
  assert.deepEqual(links('(see the VLM loss) and again the VLM loss, but not the flow loss'), ['vlm-loss']);
  assert.deepEqual(links('Covered in the Ingestion, Decision model, and Training losses chapters.'), ['context', 'decision', 'eam-loss']);
  assert.deepEqual(links('The decision model and the world model.'), []);
});
