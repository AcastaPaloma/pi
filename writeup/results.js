import { nodes, tasks } from './results-content.js?v=20261017-x2';

const $ = id => document.getElementById(id);
const video = $('video'), tree = $('tree');
let current = null, clipIndex = 0;

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

const outcome = c => c.success ? `Done in ${c.seconds.toFixed(1)} s` : `Not finished in ${Math.round(c.seconds)} s`;
const clipName = c => (c.label ? `${c.label} · ` : '') + tasks[c.task].short;

const items = new Map();
for (const node of nodes) {
  const li = el('li', 'node');
  li.dataset.kind = node.kind;
  li.dataset.depth = String(node.depth);
  li.id = node.id;
  const head = el('button', 'node-head');
  head.type = 'button';
  head.setAttribute('aria-expanded', 'false');
  head.append(el('span', 'node-when', node.when + (node.kind === 'running' ? ' · running' : node.kind === 'dead' ? ' · dropped' : '')),
    el('span', 'node-name', node.name));
  const score = el('span', 'node-score');
  if (node.score) {
    const bar = el('span', 'bar'), fill = el('i');
    fill.style.width = `${Math.max(2, 100 * node.score.value / node.score.of)}%`;
    bar.append(fill);
    const num = el('span');
    num.append(el('b', '', String(node.score.value)), ` / ${node.score.of}`);
    score.append(bar, num);
  }
  if (node.clips.length) score.append(el('span', 'node-clips', `▶ ${node.clips.length}`));
  head.append(score);
  head.addEventListener('click', () => select(node, 0, true));
  li.append(head);
  items.set(node.id, li);
  tree.append(li);
}

function detail(node) {
  const box = el('div', 'node-detail');
  const para = (label, text) => { const p = el('p'); p.append(el('span', '', label), text); return p; };
  box.append(para('What changed', node.changed), para('Result', node.result));
  if (node.score?.label) box.append(el('p', 'score-note', `Score shown: ${node.score.label}, ${node.score.value} of ${node.score.of}.`));
  box.append(para(node.kind === 'main' ? 'Next' : node.kind === 'running' ? 'Now' : 'Why we moved on', node.next));
  if (node.clips.length > 1) {
    const chips = el('div', 'chips');
    chips.setAttribute('role', 'group');
    chips.setAttribute('aria-label', 'Videos for this experiment');
    node.clips.forEach((c, i) => {
      const chip = el('button', 'chip');
      chip.type = 'button';
      chip.setAttribute('aria-pressed', String(i === clipIndex));
      chip.append(el('i', c.success ? '' : 'is-miss'), clipName(c));
      chip.addEventListener('click', () => select(node, i, true));
      chips.append(chip);
    });
    box.append(chips);
  }
  return box;
}

function show(node) {
  const c = node.clips[clipIndex];
  $('stage-empty').hidden = Boolean(c);
  $('clip-card').hidden = !c;
  video.hidden = !c;
  if (!c) {
    video.pause();
    video.removeAttribute('src');
    video.load();
    $('empty-title').textContent = node.name;
    return;
  }
  const task = tasks[c.task];
  $('clip-label').textContent = `${node.name}${c.label ? ' · ' + c.label : ''}`;
  $('clip-title').textContent = task.name;
  const line = $('clip-outcome');
  line.textContent = '';
  line.append(el('i', `outcome-mark${c.success ? '' : ' is-miss'}`), outcome(c),
    el('span', 'clip-goal', c.rate !== undefined ? ` · succeeds in ${c.rate} of ${c.of || 30} random layouts` : ` · layout ${c.layout} of 30`));
  video.poster = c.poster;
  video.src = c.video;
  video.setAttribute('aria-label', `${node.name}: ${task.name}. ${task.goal} ${outcome(c)}.`);
  video.play().catch(() => {});
}

function select(node, index, userChose) {
  current = node;
  clipIndex = Math.min(index, Math.max(0, node.clips.length - 1));
  for (const [id, li] of items) {
    const on = id === node.id;
    li.setAttribute('aria-current', String(on));
    li.querySelector('.node-head').setAttribute('aria-expanded', String(on));
    li.querySelector('.node-detail')?.remove();
    if (on) li.append(detail(node));
  }
  show(node);
  $('counter').textContent = `${nodes.indexOf(node) + 1} of ${nodes.length}`;
  if (userChose) history.replaceState(null, '', `#${node.id}${clipIndex ? '/' + clipIndex : ''}`);
}

// Start at the hash, else the best model so far (the latest kept experiment with a video).
const [hashId, hashClip] = location.hash.slice(1).split('/');
const start = nodes.find(n => n.id === hashId) || [...nodes].reverse().find(n => n.kind === 'main' && n.clips.length) || nodes[0];
select(start, Number(hashClip) || 0, false);
if (hashId) items.get(start.id).scrollIntoView({ block: 'center' });

addEventListener('keydown', e => {
  if (e.target === video || e.target.closest('input, textarea, [contenteditable], a') || e.metaKey || e.ctrlKey) return;
  const i = nodes.indexOf(current);
  const go = (node, k) => { e.preventDefault(); select(node, k, true); items.get(node.id).scrollIntoView({ block: 'nearest' }); };
  if (e.key === 'ArrowDown' && i < nodes.length - 1) go(nodes[i + 1], 0);
  if (e.key === 'ArrowUp' && i > 0) go(nodes[i - 1], 0);
  if (e.key === 'ArrowRight' && clipIndex < current.clips.length - 1) go(current, clipIndex + 1);
  if (e.key === 'ArrowLeft' && clipIndex > 0) go(current, clipIndex - 1);
});
