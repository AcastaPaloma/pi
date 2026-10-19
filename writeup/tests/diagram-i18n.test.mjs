import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { translateArtwork } from '../diagram-i18n.mjs';
import { translationPosition, translationLine, HOVER_DELAY } from '../translation-ui.mjs';
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
test('popup stays beside the source line and flips below when the top is crowded', () => {
  assert.ok(HOVER_DELAY >= 800);
  const source = {left:700,right:920,top:180,bottom:204};
  const desktop = translationPosition({anchor:source, point:{x:800,y:190},bounds:{left:640,right:1280}, width:320,height:80,viewportWidth:1280,viewportHeight:720});
  assert.equal(desktop.placement,'above'); assert.ok(desktop.top + 80 < source.top);
  assert.ok(desktop.left >= 640 && desktop.left + 320 < 1280);
  const top = translationPosition({anchor:{...source,top:20,bottom:44},width:320,height:80,viewportWidth:1280,viewportHeight:720});
  assert.equal(top.placement,'below'); assert.ok(top.top > 44);
  const expanded = translationPosition({anchor:source,width:380,height:300,viewportWidth:1280,viewportHeight:720,preferredPlacement:'above'});
  assert.equal(expanded.placement,'above'); assert.ok(expanded.top >= 10); assert.equal(expanded.maxHeight,161);
  const mobile = translationPosition({anchor:{left:20,right:350,top:520,bottom:544},point:{x:370,y:530},bounds:{left:0,right:390},width:366,height:110,viewportWidth:390,viewportHeight:640});
  assert.ok(mobile.left >= 10 && mobile.left + 366 <= 380); assert.ok(mobile.top + 110 < 520);
});
test('wrapped sentences use the line under the pointer, with a keyboard fallback', () => {
  const lines = [{left:700,right:1200,top:200,bottom:224,width:500,height:24},{left:700,right:940,top:238,bottom:262,width:240,height:24}];
  assert.equal(translationLine(lines,{x:810,y:250},null),lines[1]);
  assert.equal(translationLine(lines,null,null),lines[0]);
});
