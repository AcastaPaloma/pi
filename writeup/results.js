import { clips, models, tasks } from './results-content.js?v=20261017-x1';

const $ = id => document.getElementById(id);
const video = $('video'), list = $('clip-list');
const order = Object.keys(tasks);
const sorted = [...clips].sort((a, b) => order.indexOf(a.task) - order.indexOf(b.task));

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function outcome(clip) {
  return clip.success ? `Done in ${clip.seconds.toFixed(1)} s` : `Not finished in ${clip.seconds.toFixed(0)} s`;
}

const buttons = new Map();
let group = null;
for (const clip of sorted) {
  if (clip.task !== group) {
    group = clip.task;
    list.append(el('div', 'clip-group', tasks[group].name));
  }
  const task = tasks[clip.task], model = models[clip.model];
  const button = el('button', 'clip-item');
  button.type = 'button';
  button.setAttribute('role', 'listitem');
  const thumb = el('img');
  thumb.src = clip.poster;
  thumb.alt = '';
  thumb.loading = 'lazy';
  const text = el('span', 'clip-text');
  text.append(el('span', 'clip-name', model.name),
    el('span', 'clip-meta', `${task.setup} · layout ${clip.layout} of 30 · ${outcome(clip)}`),
    el('span', 'clip-rate', `Succeeds in ${clip.rate} of 30 random layouts`));
  button.append(thumb, text);
  button.addEventListener('click', () => select(clip, true));
  buttons.set(clip.id, button);
  list.append(button);
}

const modelList = $('model-list');
for (const model of Object.values(models)) {
  const dt = el('dt', '', model.name);
  dt.append(el('span', '', model.run));
  modelList.append(dt, el('dd', '', `${model.about} Succeeds in ${model.score} of 120 test trials (30 per task).`));
}
const best = models.best;
$('scoreline').innerHTML = '';
$('scoreline').append('The best model so far succeeds in ', el('strong', '', `${best.score} of 120`),
  ' random trials, 30 per task, up from ', el('strong', '', `${models.previous.score}`),
  ' for the previous best. The clips are added as training goes on.');

function select(clip, userChose) {
  const task = tasks[clip.task], model = models[clip.model];
  for (const [id, button] of buttons) button.setAttribute('aria-current', String(id === clip.id));
  $('clip-label').textContent = `${task.setup} · ${model.name}`;
  $('clip-title').textContent = task.name;
  const outcomeLine = $('clip-outcome');
  outcomeLine.textContent = '';
  outcomeLine.append(el('i', `outcome-mark${clip.success ? '' : ' is-miss'}`), outcome(clip), el('span', 'clip-goal', ` · ${task.goal}`));
  $('counter').textContent = `${sorted.indexOf(clip) + 1} of ${sorted.length} clips`;
  video.poster = clip.poster;
  video.src = clip.video;
  video.setAttribute('aria-label', `${model.name}: ${task.name}. ${task.goal} ${outcome(clip)}.`);
  video.play().catch(() => {});
  if (userChose) history.replaceState(null, '', `#${clip.id}`);
}

const fromHash = sorted.find(c => c.id === location.hash.slice(1));
if (sorted.length) select(fromHash || sorted[0], false);
addEventListener('keydown', e => {
  if (e.target === video || e.target.closest('input, textarea, [contenteditable], a, button') || !sorted.length) return;
  const current = sorted.findIndex(c => buttons.get(c.id).getAttribute('aria-current') === 'true');
  if (e.key === 'ArrowDown' || e.key === 'ArrowRight') { e.preventDefault(); select(sorted[Math.min(sorted.length - 1, current + 1)], true); }
  if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') { e.preventDefault(); select(sorted[Math.max(0, current - 1)], true); }
});
