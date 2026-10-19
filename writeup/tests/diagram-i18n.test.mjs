import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { translateArtwork } from '../diagram-i18n.mjs';
import { translationPosition, HOVER_DELAY } from '../translation-ui.mjs';
const catalog = Object.assign({}, ...['diagram-training','diagram-other'].map(name => JSON.parse(fs.readFileSync(new URL(`../locales/${name}.zh.json`, import.meta.url), 'utf8'))));
const decode = s => s.replace(/&#(x[0-9a-f]+|\d+);/gi, (_, n) => String.fromCodePoint(n[0] === 'x' ? parseInt(n.slice(1),16) : Number(n))).replace(/&(amp|lt|gt|quot|apos);/g, (_, n) => ({amp:'&',lt:'<',gt:'>',quot:'"',apos:"'"})[n]);
const sourceLabels = new Set(['full-backbone','decision-model','inference','subgoal-generation','loss-vlm','loss-eam','backbone-overview','world-model-placeholder'].flatMap(name => [...fs.readFileSync(new URL(`../assets/${name}.svg`, import.meta.url), 'utf8').matchAll(/<text\b[^>]*?(?:\/>|>([\s\S]*?)<\/text>)/g)].map(m => decode(m[1] || ''))));
test('diagram translations match exact source labels across all eight drawings', () => {
  assert.ok(Object.keys(catalog).length > 200);
  for (const [en, zh] of Object.entries(catalog)) { assert.ok(sourceLabels.has(en), `Stale label: ${en}`); assert.ok(zh.trim()); }
  for (const label of ['Low-Level Planning: One Forward Pass (two, really)', 'Action Expert Model', 'Decoder', 'Encoder']) assert.ok(catalog[label]);
});
class Text {
  constructor(text) { this.textContent = text; this.dataset = {}; this.attrs = {x:'24', y:'16', 'font-family':'Excalifont', fill:'#1e1e1e'}; }
  hasAttribute(name) { return name === 'data-original-text' ? this.dataset.originalText !== undefined : name in this.attrs; }
  getAttribute(name) { return this.attrs[name] ?? null; }
  setAttribute(name, value) { this.attrs[name] = value; }
  removeAttribute(name) { delete this.attrs[name]; }
  getComputedTextLength() { return Number(this.attrs.textLength) || [...this.textContent].length * 10; }
}
test('SVG translation fits the original slot, preserves geometry and restores exact English', () => {
  const label = new Text('Test'), formula = new Text('[B, 50, d]');
  const root = {dataset:{}, querySelectorAll:() => [label, formula]};
  translateArtwork(root, {'Test':'一条较长的中文标签'}, 'zh');
  assert.equal(label.textContent, '一条较长的中文标签');
  assert.equal(label.getComputedTextLength(), 40);
  assert.equal(label.attrs.x, '24'); assert.equal(label.attrs.y, '16'); assert.equal(label.attrs.fill, '#1e1e1e');
  assert.equal(formula.textContent, '[B, 50, d]'); assert.equal(formula.attrs['font-family'], 'Excalifont');
  translateArtwork(root, {'Test':'一条较长的中文标签'}, 'en');
  assert.equal(label.textContent, 'Test'); assert.equal(label.attrs['font-family'], 'Excalifont'); assert.ok(!('textLength' in label.attrs));
});
test('compact translation stays in the diagram pane on desktop and stacked mobile layouts', () => {
  assert.ok(HOVER_DELAY >= 800);
  const desktop = translationPosition({anchor:{left:0,bottom:686}, width:300,height:100,viewportWidth:1280,viewportHeight:720});
  assert.ok(desktop.left + 300 < 640); assert.ok(desktop.top + 100 < 686);
  const mobile = translationPosition({anchor:{left:0,bottom:250}, width:300,height:100,viewportWidth:390,viewportHeight:640});
  assert.ok(mobile.left >= 8 && mobile.left + 300 < 390); assert.ok(mobile.top + 100 < 250);
});
