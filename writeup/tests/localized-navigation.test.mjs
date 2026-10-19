import test from 'node:test';
import assert from 'node:assert/strict';
import { compileReferences, segment } from '../math-text.mjs';

test('Chinese references link inside adjacent Chinese prose and retain chapter-only matching', () => {
  const refs = compileReferences([
    {text:'VLM 损失',target:'vlm-loss',locale:'zh'},
    {text:'高层策略',target:'subgoal-policy',chapter:true,locale:'zh'},
  ]);
  const links = text => segment(text,refs).filter(p=>p.type==='ref').map(p=>p.target);
  assert.deepEqual(links('参见VLM 损失部分；再次提及VLM 损失。'),['vlm-loss']);
  assert.deepEqual(links('详见“高层策略”章节。'),['subgoal-policy']);
  assert.deepEqual(links('高层策略会写出子任务。'),[]);
});

import { firstSentence, localizeReferences } from '../i18n-core.mjs';
import { references } from '../content.js';
test('localized aliases keep their upstream destinations and technical math unchanged', () => {
  const compiled = compileReferences(localizeReferences(references, 'zh'));
  const parts = segment('详见“输入、决策模型”和“训练损失”章节。张量 [B, 50, d]。', compiled);
  assert.deepEqual(parts.filter(p=>p.type==='ref').map(p=>p.target), ['context','decision','eam-loss']);
  assert.deepEqual(parts.filter(p=>p.type==='math').map(p=>p.source), ['[B, 50, d]']);
  assert.equal(parts.map(p=>p.source).join(''),'详见“输入、决策模型”和“训练损失”章节。张量 [B, 50, d]。');
});
test('result summaries stop at Chinese punctuation without cutting decimal model names', () => {
  assert.equal(firstSentence('我们从 π0.5 开始训练。然后测试双臂。'), '我们从 π0.5 开始训练。');
  assert.equal(firstSentence('We started from π0.5. Then tested both arms.'), 'We started from π0.5.');
});
