import { t, inlineBilingual, localizeArtwork } from './i18n.mjs?v=20261019-zh3';
import { drawing, steps, lanes, task, subtasks } from './inference-content.js?v=20261015-i1';
import { clamp, fitBounds, zoomAt, constrainCamera } from './camera.mjs';
import { readingIndex } from './reading.mjs?v=20261017-r2';
import { addSplitter } from './splitter.mjs?v=20261018-s1';

const $ = id => document.getElementById(id);
const reader = $('reader'), viewport = $('viewport'), world = $('world');
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const svgNS = 'http://www.w3.org/2000/svg';
const allBounds = [0, 0, drawing.width, drawing.height];
const LOOP_START = 1; // looping skips the overview and restarts at startup
let index = 0, cycle = 0, overview = false, detached = false, ready = false;
let camera = { x: 0, y: 0, scale: 1 }, animation = 0;
let tops = [], scrollFrame = 0, jumpTarget = null, jumpTimer = 0;
let playing = !reduced.matches, autoTimer = 0, autoStart = 0, stage = null;
world.style.width = `${drawing.width}px`; world.style.height = `${drawing.height}px`;

function make(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = t(text);
  if (className) el.className = className;
  return el;
}
function inline(el, text) {
  return inlineBilingual(el, text, make);
}
function block(item) {
  if (typeof item === 'string') return inline(make('p'), item);
  if (item.list) { const ul = make('ul'); for (const li of item.list) ul.append(inline(make('li'), li)); return ul; }
  if (item.note) { const q = make('blockquote', undefined, 'drawing-note'), p = inline(make('p'), item.note); p.prepend(make('strong', 'On the drawing: ')); q.append(p); return q; }
  return make('p', String(item));
}
function position(el, [x, y, w, h]) { Object.assign(el.style, { left: `${x}px`, top: `${y}px`, width: `${w}px`, height: `${h}px` }); }

// ---- Article ----
function buildArticle() {
  const fragment = document.createDocumentFragment();
  steps.forEach((step, i) => {
    const section = make('section', undefined, 'reading-block step-block');
    section.id = `read-${step.id}`; section.dataset.index = i;
    const heading = make('h2', step.title); heading.id = `title-${step.id}`;
    section.setAttribute('aria-labelledby', heading.id);
    const meta = make('p', undefined, 'step-meta');
    meta.append(make('span', String(i).padStart(2, '0'), 'step-number'));
    if (step.lane) { const chip = make('span', lanes[step.lane].title, 'lane-chip'); chip.dataset.lane = step.lane; meta.append(chip); }
    if (step.time) meta.append(make('span', step.time, 'step-time'));
    const body = make('div', undefined, 'passage-body');
    body.append(...step.text.map(block));
    section.append(meta, heading, body);
    fragment.append(section);
  });
  $('article').replaceChildren(fragment);
}

// ---- Timeline canvas ----
function paint() { world.style.transform = `translate(${camera.x}px, ${camera.y}px) scale(${camera.scale})`; }
function setCamera(next, animate) {
  cancelAnimationFrame(animation);
  const end = constrainCamera(next, viewport.clientWidth, viewport.clientHeight, drawing.width, drawing.height);
  if (!animate || reduced.matches) { camera = end; paint(); return; }
  const start = { ...camera }, t0 = performance.now(), dur = 520;
  const tick = t => {
    const k = Math.min(1, (t - t0) / dur), e = 1 - (1 - k) ** 4;
    camera = { x: start.x + (end.x - start.x) * e, y: start.y + (end.y - start.y) * e, scale: start.scale + (end.scale - start.scale) * e };
    paint(); if (k < 1) animation = requestAnimationFrame(tick);
  };
  animation = requestAnimationFrame(tick);
}
function fit(animate = true) {
  detached = false; renderFocus();
  const pad = viewport.clientWidth < 500 ? 14 : 28, top = 44;
  const bounds = overview ? allBounds : steps[index].bounds;
  const f = fitBounds(bounds, viewport.clientWidth, viewport.clientHeight - top, pad, overview ? 1 : 1.1);
  f.y += top; setCamera(f, animate);
}
function renderFocus() {
  const bounds = steps[index].bounds, focused = !overview && index > 0;
  world.classList.toggle('focused', focused);
  $('lifted').hidden = !focused;
  if (focused) { position($('lifted'), bounds); $('lifted').querySelector('svg')?.setAttribute('viewBox', bounds.join(' ')); }
  for (const button of $('hotspots').children) button.classList.toggle('is-current', Number(button.dataset.index) === index);
  $('start').firstChild.textContent = overview || detached ? 'Back to this block ' : 'Whole timeline ';
}
async function loadDrawing() {
  try {
    const response = await fetch(drawing.src);
    if (!response.ok) throw new Error(`Timeline: ${response.status}`);
    const parsed = new DOMParser().parseFromString(await response.text(), 'image/svg+xml');
    if (parsed.querySelector('parsererror')) throw new Error('Invalid SVG');
    const source = parsed.documentElement;
    const svg = document.createElementNS(svgNS, 'svg'); svg.setAttribute('viewBox', allBounds.join(' ')); svg.setAttribute('aria-hidden', 'true');
    const defs = document.createElementNS(svgNS, 'defs'), group = document.createElementNS(svgNS, 'g');
    group.id = 'inference-drawing';
    // Excalidraw exports carry their own fonts and defs; keep them, scoped under one group.
    for (const child of [...source.children]) (child.localName === 'defs' ? defs : group).append(document.importNode(child, true));
    defs.append(group); svg.append(defs);
    const use = document.createElementNS(svgNS, 'use'); use.setAttribute('href', '#inference-drawing'); svg.append(use);
    $('artwork').replaceChildren(svg);
    const focus = document.createElementNS(svgNS, 'svg'); focus.setAttribute('preserveAspectRatio', 'none'); focus.append(use.cloneNode());
    $('lifted').replaceChildren(focus);
    await document.fonts.ready; localizeArtwork(svg);
    $('loading').hidden = true; renderFocus(); fit(false);
  } catch (error) { console.error(error); $('loading').textContent = 'The timeline drawing could not load.'; }
}
function buildHotspots() {
  steps.forEach((step, i) => {
    if (!i) return;
    const button = make('button', undefined, 'hotspot'); button.type = 'button'; button.dataset.index = i; button.tabIndex = -1;
    button.setAttribute('aria-label', `Go to ${step.title}`); position(button, step.bounds);
    button.addEventListener('click', () => { if (moved) return; pause(); jumpTo(i); });
    $('hotspots').append(button);
  });
}

// ---- Stage state for a block ----
function stageSpec(i) {
  const step = steps[i], st = step.stage, base = 2 * cycle;
  const subtaskOf = c => c == null ? null : subtasks[(base + (c === 'B' ? 1 : 0)) % subtasks.length];
  let active = null, pending = null;
  for (let j = 0; j <= i; j++) {
    const s = steps[j].stage;
    if (s.chunk) { active = { t0: base + s.chunk[0], t1: base + s.chunk[1], animate: j === i }; pending = null; }
    if (s.pending) pending = { t0: base + s.pending[0], t1: base + s.pending[1], animate: j === i };
    if (s.splice && pending) { active = { ...pending, animate: false, skip: 6 }; pending = null; }
  }
  return {
    beat: st.beat, T: base + st.tau, dwell: st.dwell, caption: t(subtaskOf(st.caption)), typing: Boolean(st.type),
    goal: st.goal ? [(base + (st.goal[0] === 'B' ? 1 : 0)) % subtasks.length, st.goal[1], st.goal[2]] : null,
    active, pending, time: step.time, lane: step.lane, laneTitle: step.lane ? lanes[step.lane].title : null
  };
}
function loopLabel() {
  const a = subtasks[(2 * cycle) % subtasks.length], b = subtasks[(2 * cycle + 1) % subtasks.length];
  $('cycle').textContent = `Loop ${cycle + 1} · ${a} → ${b}`;
}

// ---- Navigation ----
function activate(i, { history = 'replace', force = false } = {}) {
  if (i === index && !force) return;
  index = i; overview = false;
  for (const section of $('article').children) section.classList.toggle('is-current', Number(section.dataset.index) === index);
  $('counter').textContent = `${String(index).padStart(2, '0')} / ${String(steps.length - 1).padStart(2, '0')}`;
  renderFocus(); fit(true); loopLabel();
  stage?.play(stageSpec(index));
  const hash = `#inf-${steps[index].id}`;
  if (history !== 'none' && hash !== location.hash) window.history[history === 'push' ? 'pushState' : 'replaceState'](null, '', hash);
  $('announcement').textContent = steps[index].title;
  schedule();
}
function measure() {
  const rect = reader.getBoundingClientRect();
  tops = steps.map(step => $(`title-${step.id}`).closest('section').getBoundingClientRect().top - rect.top + reader.scrollTop);
}
const anchor = () => Math.min(150, reader.clientHeight * .24);
function sync() { if (!ready || jumpTarget != null) return; activate(readingIndex(tops, reader.scrollTop, anchor())); }
function jumpTo(i, { animate = true, history = 'push' } = {}) {
  clearTimeout(jumpTimer);
  activate(i, { history, force: true });
  measure();
  jumpTarget = i;
  reader.scrollTo({ top: Math.max(0, tops[i] - anchor() + 4), behavior: animate && !reduced.matches ? 'smooth' : 'instant' });
  jumpTimer = setTimeout(() => { jumpTarget = null; }, 900);
}
function next() {
  if (index < steps.length - 1) jumpTo(index + 1);
  else { cycle++; jumpTo(LOOP_START); }
}
function prev() {
  if (index > LOOP_START || (index === LOOP_START && cycle === 0)) jumpTo(Math.max(0, index - 1));
  else if (index === LOOP_START) { cycle--; jumpTo(steps.length - 1); }
}

// ---- Autoplay ----
function schedule() {
  clearTimeout(autoTimer); cancelAnimationFrame(progressFrame);
  $('play').setAttribute('aria-pressed', String(playing));
  $('play').firstChild.textContent = playing ? 'Pause' : 'Play';
  if (!playing) { $('progress').style.transform = 'scaleX(0)'; return; }
  const dwell = steps[index].stage.dwell * 1000 * (index === 0 ? 1.8 : 1);
  autoStart = performance.now();
  autoTimer = setTimeout(next, dwell);
  const tick = t => { $('progress').style.transform = `scaleX(${clamp((t - autoStart) / dwell, 0, 1)})`; progressFrame = requestAnimationFrame(tick); };
  progressFrame = requestAnimationFrame(tick);
}
let progressFrame = 0;
function pause() { if (playing) { playing = false; schedule(); } }

// ---- Dragging and zooming the timeline ----
// Moving the drawing by hand pauses the walkthrough; "Back to this block" refits.
const pointers = new Map();
let gesture = null, moved = false;
const minScale = () => fitBounds(allBounds, viewport.clientWidth, viewport.clientHeight, 0).scale * .6;
function moveByHand() { if (!detached) { detached = true; pause(); renderFocus(); } }
function zoom(factor, point = { x: viewport.clientWidth / 2, y: viewport.clientHeight / 2 }, animate = false) {
  moveByHand(); setCamera(zoomAt(camera, point, clamp(camera.scale * factor, minScale(), 8)), animate);
}
function localPoint(event) { const rect = viewport.getBoundingClientRect(); return { x: event.clientX - rect.left, y: event.clientY - rect.top }; }
function resetGesture() {
  const [a, b] = pointers.values();
  if (!a) gesture = null;
  else if (!b) gesture = { type: 'pan', point: a, camera: { ...camera } };
  else gesture = { type: 'pinch', center: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, distance: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)), camera: { ...camera } };
}
viewport.addEventListener('pointerdown', event => {
  if (event.button !== 0) return;
  cancelAnimationFrame(animation);
  if (!pointers.size) moved = false;
  pointers.set(event.pointerId, localPoint(event)); resetGesture();
  // A press on a block stays a click until it becomes a drag.
  if (!event.target.closest('button')) viewport.setPointerCapture(event.pointerId);
});
viewport.addEventListener('pointermove', event => {
  if (!pointers.has(event.pointerId) || !gesture) return;
  const point = localPoint(event); pointers.set(event.pointerId, point);
  if (gesture.type === 'pan') {
    const dx = point.x - gesture.point.x, dy = point.y - gesture.point.y;
    if (!moved && Math.hypot(dx, dy) < 5) return;
    moved = true; viewport.setPointerCapture(event.pointerId);
    setCamera({ ...gesture.camera, x: gesture.camera.x + dx, y: gesture.camera.y + dy });
  } else {
    const [a, b] = pointers.values(), center = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    const next = zoomAt(gesture.camera, gesture.center, clamp(gesture.camera.scale * Math.hypot(a.x - b.x, a.y - b.y) / gesture.distance, minScale(), 8));
    next.x += center.x - gesture.center.x; next.y += center.y - gesture.center.y;
    moved = true; setCamera(next);
  }
  viewport.classList.add('dragging'); moveByHand();
});
function pointerEnd(event) {
  if (!pointers.has(event.pointerId)) return;
  pointers.delete(event.pointerId);
  if (viewport.hasPointerCapture(event.pointerId)) viewport.releasePointerCapture(event.pointerId);
  resetGesture(); if (!pointers.size) viewport.classList.remove('dragging');
  // The click that ends a drag sees moved=true; later clicks must not.
  if (!pointers.size) setTimeout(() => { moved = false; }, 0);
}
window.addEventListener('pointerup', pointerEnd);
window.addEventListener('pointercancel', pointerEnd);
window.addEventListener('blur', () => { pointers.clear(); gesture = null; moved = false; viewport.classList.remove('dragging'); });
viewport.addEventListener('wheel', event => {
  event.preventDefault();
  const factor = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? viewport.clientHeight : 1;
  if (event.ctrlKey || event.metaKey) zoom(Math.exp(-event.deltaY * factor * .008), localPoint(event));
  else { moveByHand(); setCamera({ ...camera, x: camera.x - event.deltaX * factor, y: camera.y - event.deltaY * factor }); }
}, { passive: false });
viewport.addEventListener('dblclick', event => { if (!event.target.closest('button')) zoom(event.shiftKey ? .65 : 1.5, localPoint(event), true); });
viewport.addEventListener('scroll', () => { if (viewport.scrollTop || viewport.scrollLeft) viewport.scrollTo(0, 0); });

// ---- Events ----
$('prev').addEventListener('click', () => { pause(); prev(); });
$('next').addEventListener('click', () => { pause(); next(); });
$('play').addEventListener('click', () => { playing = !playing; if (playing && index === steps.length - 1) next(); else schedule(); });
$('start').addEventListener('click', () => { overview = !overview && !detached; fit(true); });
reader.addEventListener('scroll', () => {
  if (jumpTarget != null) { clearTimeout(jumpTimer); jumpTimer = setTimeout(() => { jumpTarget = null; }, 140); return; }
  if (!scrollFrame) scrollFrame = requestAnimationFrame(() => { scrollFrame = 0; sync(); });
}, { passive: true });
for (const type of ['wheel', 'touchstart']) reader.addEventListener(type, () => { jumpTarget = null; pause(); }, { passive: true });
document.querySelector('.skip-link').addEventListener('click', e => { e.preventDefault(); reader.focus({ preventScroll: true }); });
document.addEventListener('keydown', event => {
  if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey || event.target.closest('input,textarea,select')) return;
  if (event.key === 'ArrowRight') { event.preventDefault(); pause(); next(); }
  else if (event.key === 'ArrowLeft') { event.preventDefault(); pause(); prev(); }
  else if (event.key === ' ' && !event.target.closest('button,a')) { event.preventDefault(); playing = !playing; schedule(); }
  else if (event.key === 'Escape' && (overview || detached)) { overview = false; fit(true); }
  else if (event.key === '+' || event.key === '=') { event.preventDefault(); zoom(1.2); }
  else if (event.key === '-') { event.preventDefault(); zoom(1 / 1.2); }
});
window.addEventListener('hashchange', () => { const i = steps.findIndex(s => `#inf-${s.id}` === location.hash); if (i >= 0 && i !== index) { pause(); jumpTo(i, { history: 'none' }); } });
new ResizeObserver(() => { measure(); if (detached) setCamera(camera); else fit(false); }).observe(viewport);
new ResizeObserver(measure).observe(reader);
document.addEventListener('visibilitychange', () => { if (document.hidden) { clearTimeout(autoTimer); cancelAnimationFrame(progressFrame); } else schedule(); });

// ---- Start ----
addSplitter(document.querySelector('.workspace'), { name: 'inference', label: 'Resize the visuals and the text' });
addSplitter(document.querySelector('.visual-column'), { name: 'inference-stage', label: 'Resize the 3D view and the timeline' });
buildArticle(); buildHotspots(); loopLabel();
const initial = Math.max(0, steps.findIndex(s => `#inf-${s.id}` === location.hash));
if (initial > 0) playing = false;
ready = true; measure();
jumpTo(initial, { animate: false, history: 'none' });
loadDrawing();
import('./robot-stage.js?v=20261015-i1').then(m => m.createStage($('stage'), { task: t(task), subtasks: subtasks.map(t) })).then(created => {
  stage = created; $('stage-loading').hidden = true;
  if (new URLSearchParams(location.search).has('debug')) window.__stage = stage;
  stage.play(stageSpec(index)); schedule();
}).catch(error => {
  console.error(error);
  $('stage-loading').textContent = 'The 3D view needs WebGL, which this browser could not start. The timeline and text still work.';
});
