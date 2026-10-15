import { artwork, macros, steps, details, annotations } from './content.js?v=20261014-r2';
import { clamp, fitBounds, zoomAt, constrainCamera } from './camera.mjs';
import { buildReadingEntries, readingIndex } from './reading.mjs?v=20261014-r2';

import { validateCopy, renderNodes } from './rich-text.mjs';

const $ = id => document.getElementById(id);
const viewport = $('viewport'), world = $('world'), reader = $('reader');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const detailById = new Map(details.map(item => [item.id, item]));
const entries = buildReadingEntries(steps, details, annotations);
const entryById = new Map(entries.map(item => [item.id, item]));
const allBounds = [0, 0, artwork.width, artwork.height];
const svgNS = 'http://www.w3.org/2000/svg';
let ready = false;
let stepIndex = 0, pinned = null, preview = null, activeId = steps[0].id;
let camera = { x: 0, y: 0, scale: 1 }, animation = 0, frame = 0;
let hoverTimer = 0, leaveTimer = 0, suppressHoverUntil = 0, moved = false;
let activeGesture = null, detached = false, overview = false;
let scrollFrame = 0, jumpTarget = null, jumpTimer = 0, tops = [];
// Component notes are folded under their main section until opened. Only open
// passages take part in reading-position tracking.
const expanded = new Set();
let visible = entries;
const pointers = new Map();
world.style.width = `${artwork.width}px`;
world.style.height = `${artwork.height}px`;
function make(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function position(element, bounds) {
  const [x, y, width, height] = bounds;
  Object.assign(element.style, { left: `${x}px`, top: `${y}px`, width: `${width}px`, height: `${height}px` });
}
function currentStep() { return steps[stepIndex]; }
function currentBounds() { return detailById.get(pinned)?.bounds ?? currentStep()?.frame ?? currentStep()?.bounds ?? allBounds; }
function clearTimers() { clearTimeout(hoverTimer); clearTimeout(leaveTimer); }
function paintCamera() { world.style.transform = `translate(${camera.x}px, ${camera.y}px) scale(${camera.scale})`; }
function requestPaint() { if (!frame) frame = requestAnimationFrame(() => { frame = 0; paintCamera(); }); }
function minScale() { return Math.min(.12, fitBounds(allBounds, viewport.clientWidth, viewport.clientHeight, 12).scale); }
function setCamera(next, animate = false) {
  cancelAnimationFrame(animation);
  const end = constrainCamera(next, viewport.clientWidth, viewport.clientHeight, artwork.width, artwork.height);
  if (!animate || reducedMotion.matches) { camera = end; requestPaint(); return; }
  const start = { ...camera }, started = performance.now(), duration = 340;
  suppressHoverUntil = started + duration + 80;
  function tick(now) {
    const t = Math.min(1, (now - started) / duration), ease = 1 - (1 - t) ** 4;
    camera = { x: start.x + (end.x - start.x) * ease, y: start.y + (end.y - start.y) * ease, scale: start.scale + (end.scale - start.scale) * ease };
    paintCamera();
    if (t < 1) animation = requestAnimationFrame(tick);
  }
  animation = requestAnimationFrame(tick);
}
function fit(animate = true) {
  // Leave breathing room around details; the walkthrough button lives above them.
  const padding = viewport.clientWidth < 500 ? 18 : 36;
  const topInset = 48;
  const fitted = fitBounds(currentBounds(), viewport.clientWidth, viewport.clientHeight - topInset, padding, currentStep()?.maxScale ?? (pinned ? 3.5 : 3));
  fitted.y += topInset;
  setCamera(fitted, animate);
}
function manuallyMoved() {
  detached = true;
  $('start').firstChild.textContent = 'Resume reading ';
  suppressHoverUntil = performance.now() + 180; clearTimers();
  if (preview) { preview = null; renderSelection(); }
}
function zoom(factor, point = { x: viewport.clientWidth / 2, y: viewport.clientHeight / 2 }, animate = false) {
  manuallyMoved(); setCamera(zoomAt(camera, point, clamp(camera.scale * factor, minScale(), 8)), animate);
}

function writeRoute(mode = 'replace') {
  const hash = `#${currentStep().id}${pinned ? `/${pinned}` : ''}`;
  if (hash !== location.hash) history[mode === 'push' ? 'pushState' : 'replaceState'](null, '', `${location.pathname}${location.search}${hash}`);
}
function activate(id, { historyMode = 'replace', animate = true, force = false } = {}) {
  const entry = entryById.get(id); if (!entry) return;
  const changed = activeId !== id;
  if (!changed && !detached && !overview && !force) return;
  activeId = id; stepIndex = entry.stepIndex; pinned = entry.detail;
  preview = null; detached = false; overview = false; clearTimers();
  render(); fit(animate);
  if (historyMode !== 'none') writeRoute(historyMode);
  if (changed) $('announcement').textContent = entry.title;
}
function isShown(entry) { return !entry.detail || expanded.has(steps[entry.stepIndex].id); }
function measureReading() {
  const rect = reader.getBoundingClientRect();
  visible = entries.filter(isShown);
  tops = visible.map(entry => $(`title-${entry.id}`).getBoundingClientRect().top - rect.top + reader.scrollTop);
}
function readingAnchor() { return Math.min(150, reader.clientHeight * .24); }
function syncReading() {
  if (!ready || jumpTarget) return;
  const entry = visible[readingIndex(tops, reader.scrollTop, readingAnchor())];
  if (entry) activate(entry.id);
}
function scheduleReading() {
  if (!scrollFrame) scrollFrame = requestAnimationFrame(() => { scrollFrame = 0; syncReading(); });
}
function cancelJump() {
  clearTimeout(jumpTimer); jumpTarget = null;
}
function finishJump() {
  if (!jumpTarget) return;
  cancelJump(); scheduleReading();
}
function jumpTo(id, { historyMode = 'push', animate = true } = {}) {
  if (!entryById.has(id)) return;
  closeContents(); cancelJump();
  const target = entryById.get(id);
  if (target.detail) setExpanded(steps[target.stepIndex].id, true);
  activate(id, { historyMode, animate, force: true });
  measureReading();
  const index = visible.findIndex(entry => entry.id === id);
  // Align to the same line the scroll observer uses, so a click and the next
  // wheel movement agree on which passage owns the camera.
  const top = Math.max(0, tops[index] - readingAnchor() + 4);
  jumpTarget = id;
  reader.scrollTo({ top, behavior: animate && !reducedMotion.matches ? 'smooth' : 'instant' });
  jumpTimer = setTimeout(finishJump, 850);
  if (historyMode === 'push') reader.focus({ preventScroll: true });
}
function go(index, detail) {
  const step = steps[clamp(index, 0, steps.length - 1)];
  jumpTo(detail && step.details.includes(detail) ? detail : step.id);
}
function readRoute() {
  const [id, detail] = location.hash.slice(1).split('/');
  const step = steps.find(item => item.id === id) ?? steps[0];
  jumpTo(detail && step.details.includes(detail) ? detail : step.id, { historyMode: 'none', animate: false });
}
function inline(element, text) {
  // Only **bold** is interpreted; everything else stays literal text.
  text.split(/\*\*(.+?)\*\*/).forEach((part, index) => {
    if (part) element.append(index % 2 ? make('strong', part) : document.createTextNode(part));
  });
  return element;
}
function flowEquation() {
  const ns = 'http://www.w3.org/1998/Math/MathML';
  const math = document.createElementNS(ns, 'math'); math.setAttribute('display', 'block');
  const sub = document.createElementNS(ns, 'msub');
  const letter = document.createElementNS(ns, 'mi'); letter.textContent = 'ℒ';
  const label = document.createElementNS(ns, 'mtext'); label.textContent = 'flow';
  const period = document.createElementNS(ns, 'mo'); period.textContent = '.';
  sub.append(letter, label); math.append(sub, period);
  const holder = make('div'); holder.dataset.equation = 'flow-loss'; holder.contentEditable = 'false'; holder.append(math); return holder;
}
function paragraph(block) {
  if (block === String.raw`\[\mathcal L_{\rm flow}.\]`) return flowEquation();
  if (typeof block === 'string') return inline(make('p'), block);
  if (block.list) {
    const list = make('ul');
    for (const item of block.list) list.append(inline(make('li'), item));
    return list;
  }
  if (block.note) {
    const note = make('blockquote', undefined, 'drawing-note');
    const text = inline(make('p'), block.note);
    text.prepend(make('strong', 'On the drawing: '));
    note.append(text); return note;
  }
  if (block.placeholder) return inline(make('p', undefined, 'placeholder-note'), block.placeholder);
  return make('p', String(block));
}
function previewRegion(id) { preview = id; renderSelection(); }
function setExpanded(stepId, open) {
  if (open) expanded.add(stepId); else expanded.delete(stepId);
  const group = $(`components-${stepId}`), toggle = $(`toggle-${stepId}`);
  if (!group) return;
  group.hidden = !open;
  toggle.setAttribute('aria-expanded', String(open));
  toggle.querySelector('.toggle-label').textContent = `${open ? 'Hide' : 'Open'} the ${group.children.length} component notes`;
  $('expand-all').checked = steps.every(step => !step.details.length || expanded.has(step.id));
}
function toggleGroup(stepId) {
  const open = !expanded.has(stepId);
  setExpanded(stepId, open);
  if (!open && entryById.get(activeId)?.detail && steps[entryById.get(activeId).stepIndex].id === stepId) activate(stepId, { force: true });
  measureReading(); scheduleReading();
}
export function expandAll(open = true) {
  for (const step of steps) if (step.details.length) setExpanded(step.id, open);
  measureReading(); scheduleReading();
}
function componentChips(step) {
  // Each chip names one drawn component. Hover or focus outlines its exact
  // region on the canvas; activation opens the notes and moves the camera there.
  const row = make('div', undefined, 'component-chips');
  for (const id of step.details) {
    const chip = make('button', entryById.get(id).title, 'component-chip');
    chip.type = 'button'; chip.dataset.target = id;
    chip.addEventListener('pointerenter', () => previewRegion(id));
    chip.addEventListener('focus', () => previewRegion(id));
    chip.addEventListener('pointerleave', () => previewRegion(null));
    chip.addEventListener('blur', () => previewRegion(null));
    chip.addEventListener('click', () => { preview = null; jumpTo(id); });
    row.append(chip);
  }
  return row;
}
function routeList(step) {
  const nav = make('nav', undefined, 'reading-routes'); nav.setAttribute('aria-label', 'Reading routes');
  nav.append(make('h3', 'Reading routes'));
  for (const route of step.routes) {
    const item = make('div', undefined, 'route');
    item.append(make('p', route.title, 'route-title'), make('p', route.note, 'route-note'));
    const links = make('ol', undefined, 'route-links');
    for (const id of route.ids) {
      const link = make('button', entryById.get(id).title, 'route-link'); link.type = 'button'; link.dataset.target = id;
      link.addEventListener('pointerenter', () => previewRegion(id));
      link.addEventListener('pointerleave', () => previewRegion(null));
      link.addEventListener('click', () => { preview = null; jumpTo(id); });
      const li = make('li'); li.append(link); links.append(li);
    }
    item.append(links); nav.append(item);
  }
  return nav;
}
function passage(entry) {
  const block = make('section', undefined, `reading-block${entry.detail ? '' : ' is-major'}`);
  block.id = `read-${entry.id}`; block.dataset.entry = entry.id;
  block.dataset.placeholder = String(entry.isPlaceholder);
  const heading = make(entry.detail ? 'h3' : 'h2', entry.title);
  heading.id = `title-${entry.id}`; block.setAttribute('aria-labelledby', heading.id);
  const body = make('div', undefined, 'passage-body'); body.dataset.section = entry.id;
  body.append(...entry.paragraphs.map(paragraph));
  block.append(heading, body);
  return block;
}
function buildArticle() {
  const fragment = document.createDocumentFragment();
  for (const step of steps) {
    const major = passage(entryById.get(step.id));
    if (step.routes) major.append(routeList(step));
    fragment.append(major);
    if (!step.details.length) continue;
    const toggle = make('button', undefined, 'components-toggle');
    toggle.type = 'button'; toggle.id = `toggle-${step.id}`;
    toggle.setAttribute('aria-controls', `components-${step.id}`); toggle.setAttribute('aria-expanded', 'false');
    toggle.append(make('span', '', 'toggle-label'), make('span', '', 'toggle-mark'));
    toggle.addEventListener('click', () => toggleGroup(step.id));
    const group = make('div', undefined, 'component-group');
    group.id = `components-${step.id}`; group.hidden = true;
    for (const id of step.details) group.append(passage(entryById.get(id)));
    const fold = make('div', undefined, 'component-fold');
    fold.append(componentChips(step), toggle);
    major.append(fold);
    fragment.append(group);
  }
  $('article').replaceChildren(fragment);
  for (const step of steps) if (step.details.length) setExpanded(step.id, false);
}
function buildContents() {
  $('expand-all').addEventListener('change', event => expandAll(event.target.checked));
  for (const macro of macros) {
    $('toc').append(make('h2', macro.title, 'toc-macro'));
    steps.forEach((step, index) => {
      if (step.macro !== macro.id) return;
      const group = make('div', undefined, 'toc-group');
      for (const id of [step.id, ...step.details]) {
        const entry = entryById.get(id), button = make('button', undefined, `toc-link${entry.detail ? ' toc-detail' : ''}`);
        button.type = 'button'; button.dataset.entry = id;
        if (!entry.detail) button.append(make('span', String(index + 1).padStart(2, '0'), 'toc-number'));
        button.append(make('span', entry.title)); button.addEventListener('click', () => jumpTo(id));
        group.append(button);
      }
      $('toc').append(group);
    });
  }
}
function closeContents() {
  if ($('contents-menu').contains(document.activeElement)) $('contents-toggle').focus({ preventScroll: true });
  $('contents-menu').hidden = true; $('contents-toggle').setAttribute('aria-expanded', 'false');
}
function buildHotspots() {
  // Coarse regions sit behind the finer targets. Every canvas target has a
  // matching heading and contents entry, including regions outside the current view.
  for (const item of [...steps, ...details]) {
    const button = make('button', undefined, 'hotspot');
    button.type = 'button'; button.tabIndex = -1; button.dataset.entry = item.id;
    button.setAttribute('aria-label', `Read ${item.title}`);
    position(button, item.bounds);
    button.addEventListener('pointerenter', event => {
      clearTimers();
      if (event.pointerType === 'touch' || pointers.size || performance.now() < suppressHoverUntil) return;
      hoverTimer = setTimeout(() => { preview = item.id; renderSelection(); }, 140);
    });
    button.addEventListener('pointerleave', () => { clearTimers(); preview = null; renderSelection(); });
    button.addEventListener('click', () => {
      if (moved) return;
      jumpTo(item.id); reader.focus({ preventScroll: true });
    });
    $('hotspots').append(button);
  }
}
function renderSelection() {
  const item = entryById.get(preview ?? (pinned ? activeId : null));
  $('detail-outline').hidden = !item || overview;
  $('detail-outline').classList.toggle('is-preview', Boolean(preview));
  if (item) position($('detail-outline'), item.bounds);
  // Mirror the canvas hover in the text: the chip or route link naming this region lights up.
  for (const link of $('article').querySelectorAll('[data-target]')) link.classList.toggle('is-linked', link.dataset.target === preview);
}
function render() {
  world.classList.toggle('focused', !overview);
  $('lifted').hidden = overview;
  if (!overview) {
    const bounds = entryById.get(activeId).bounds;
    position($('lifted'), bounds);
    $('lifted').querySelector('svg')?.setAttribute('viewBox', bounds.join(' '));
  }
  $('start').firstChild.textContent = detached || overview ? 'Resume reading ' : 'Whole canvas ';
  $('chapter-label').textContent = macros.find(macro => macro.id === currentStep().macro).title;
  for (const block of $('article').querySelectorAll('.reading-block')) block.classList.toggle('is-current', block.dataset.entry === activeId);
  for (const link of $('toc').querySelectorAll('button')) {
    if (link.dataset.entry === activeId) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  }
  for (const button of $('hotspots').children) button.classList.toggle('is-available', currentStep().details.includes(button.dataset.entry));
  renderSelection();
}
$('contents-toggle').addEventListener('click', () => {
  const open = $('contents-menu').hidden;
  $('contents-menu').hidden = !open; $('contents-toggle').setAttribute('aria-expanded', String(open));
  if (open) {
    const current = $('toc').querySelector('[aria-current="location"]');
    if (current) {
      const a = current.getBoundingClientRect(), b = $('contents-menu').getBoundingClientRect();
      $('contents-menu').scrollTop += a.top - b.top - 24;
    }
  }
});
document.addEventListener('pointerdown', event => {
  if (!event.target.closest('#contents-menu,#contents-toggle')) closeContents();
});
$('start').addEventListener('click', () => {
  if (detached || overview) { cancelJump(); activate(activeId, { force: true }); }
  else { overview = true; detached = true; preview = null; render(); setCamera(fitBounds(allBounds, viewport.clientWidth, viewport.clientHeight, 30), true); }
});
document.querySelector('.skip-link').addEventListener('click', event => { event.preventDefault(); reader.focus({ preventScroll: true }); });
reader.addEventListener('pointerenter', () => { if (detached || overview) { cancelJump(); syncReading(); } });
reader.addEventListener('focusin', () => { if (detached || overview) { cancelJump(); syncReading(); } });
reader.addEventListener('scroll', () => {
  if (jumpTarget) { clearTimeout(jumpTimer); jumpTimer = setTimeout(finishJump, 120); }
  else scheduleReading();
}, { passive: true });
reader.addEventListener('scrollend', finishJump);
reader.addEventListener('wheel', () => { cancelJump(); scheduleReading(); }, { passive: true });
reader.addEventListener('touchstart', () => { cancelJump(); scheduleReading(); }, { passive: true });
reader.addEventListener('pointerdown', event => { if (event.target.closest('[contenteditable="true"]')) return; cancelJump(); scheduleReading(); });
window.addEventListener('hashchange', readRoute);
window.addEventListener('popstate', readRoute);
document.addEventListener('keydown', event => {
  if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey || event.target.closest('input,textarea,select,[contenteditable="true"]')) return;
  if (event.key === 'Escape') {
    event.preventDefault();
    if (!$('contents-menu').hidden) closeContents();
    else if (detached || overview) activate(activeId, { force: true });
    else if (pinned) go(stepIndex);
    return;
  }
  if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
    event.preventDefault(); if (!event.repeat) go(stepIndex + (event.key === 'ArrowLeft' ? -1 : 1));
  } else if ([' ', 'PageDown', 'PageUp', 'ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key) && !event.target.closest('button,a,#contents-menu')) {
    event.preventDefault(); cancelJump();
    const direction = ['PageUp', 'ArrowUp'].includes(event.key) || (event.key === ' ' && event.shiftKey) ? -1 : 1;
    const amount = event.key.startsWith('Arrow') ? 56 : reader.clientHeight * .72;
    if (event.key === 'Home' || event.key === 'End') reader.scrollTo({ top: event.key === 'Home' ? 0 : reader.scrollHeight, behavior: 'instant' });
    else reader.scrollBy({ top: amount * direction, behavior: reducedMotion.matches ? 'instant' : 'smooth' });
    scheduleReading();
  } else if (event.key === '+' || event.key === '=') { event.preventDefault(); zoom(1.2); }
  else if (event.key === '-') { event.preventDefault(); zoom(1 / 1.2); }
  else if (event.key.toLowerCase() === 'f') { event.preventDefault(); activate(activeId, { force: true }); }
});
function localPoint(event) { const rect = viewport.getBoundingClientRect(); return { x: event.clientX - rect.left, y: event.clientY - rect.top }; }
function resetGesture() {
  const values = [...pointers.values()];
  if (!values.length) { activeGesture = null; return; }
  if (values.length === 1) activeGesture = { type: 'pan', point: values[0], camera: { ...camera } };
  else {
    const [a, b] = values;
    activeGesture = { type: 'pinch', center: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, distance: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)), camera: { ...camera } };
  }
}
viewport.addEventListener('pointerdown', event => {
  if (event.button !== 0) return;
  cancelAnimationFrame(animation); clearTimers();
  if (!pointers.size) moved = false;
  pointers.set(event.pointerId, localPoint(event)); resetGesture();
  // Preserve click targeting on hotspots until this becomes a drag.
  if (!event.target.closest('button')) viewport.setPointerCapture(event.pointerId);
});
viewport.addEventListener('pointermove', event => {
  if (!pointers.has(event.pointerId) || !activeGesture) return;
  const point = localPoint(event); pointers.set(event.pointerId, point);
  const start = activeGesture;
  if (start.type === 'pan') {
    const dx = point.x - start.point.x, dy = point.y - start.point.y;
    if (!moved && Math.hypot(dx, dy) < 5) return;
    moved = true; viewport.setPointerCapture(event.pointerId);
    setCamera({ ...start.camera, x: start.camera.x + dx, y: start.camera.y + dy });
  } else {
    const [a, b] = [...pointers.values()]; if (!b) return;
    const center = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    const scale = clamp(start.camera.scale * Math.hypot(a.x - b.x, a.y - b.y) / start.distance, minScale(), 8);
    const next = zoomAt(start.camera, start.center, scale);
    next.x += center.x - start.center.x; next.y += center.y - start.center.y;
    moved = true; setCamera(next);
  }
  viewport.classList.add('dragging'); manuallyMoved();
});
function pointerEnd(event) {
  if (!pointers.has(event.pointerId)) return;
  pointers.delete(event.pointerId);
  if (viewport.hasPointerCapture(event.pointerId)) viewport.releasePointerCapture(event.pointerId);
  resetGesture(); if (!pointers.size) viewport.classList.remove('dragging');
  // The ensuing click sees moved=true; keyboard and later clicks must not inherit it.
  if (!pointers.size) setTimeout(() => { moved = false; }, 0);
}
window.addEventListener('pointerup', pointerEnd);
window.addEventListener('pointercancel', pointerEnd);
window.addEventListener('blur', () => { pointers.clear(); activeGesture = null; moved = false; viewport.classList.remove('dragging'); });
viewport.addEventListener('wheel', event => {
  event.preventDefault();
  const factor = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? viewport.clientHeight : 1;
  if (event.ctrlKey || event.metaKey) zoom(Math.exp(-event.deltaY * factor * .008), localPoint(event));
  else {
    cancelJump(); manuallyMoved();
    setCamera({ ...camera, x: camera.x - event.deltaX * factor, y: camera.y - event.deltaY * factor });

  }
}, { passive: false });
viewport.addEventListener('dblclick', event => { if (!event.target.closest('button')) zoom(event.shiftKey ? .65 : 1.5, localPoint(event), true); });
// A camera owns this coordinate system. Native focus scrolling must never
// create an untracked second translation (also covers older overflow:clip support).
viewport.addEventListener('scroll', () => {
  if (viewport.scrollTop || viewport.scrollLeft) viewport.scrollTo(0, 0);
});
const layoutObserver = new ResizeObserver(() => {
  measureReading();
  if (!detached && !overview) { syncReading(); fit(false); }
});
layoutObserver.observe(viewport); layoutObserver.observe(reader);


async function loadArtwork() {
  $('loading').hidden = false; $('asset-error').hidden = true;
  try {
    const sources = await Promise.all(artwork.layers.map(async layer => {
      const response = await fetch(layer.src);
      if (!response.ok) throw new Error(`Artwork: ${response.status}`);
      const parsed = new DOMParser().parseFromString(await response.text(), 'image/svg+xml');
      if (parsed.querySelector('parsererror')) throw new Error('Invalid SVG');
      return { layer, source: parsed.documentElement };
    }));
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('viewBox', allBounds.join(' ')); svg.setAttribute('aria-hidden', 'true');
    const defs = document.createElementNS(svgNS, 'defs');
    const drawing = document.createElementNS(svgNS, 'g');
    drawing.id = 'pi-drawing'; drawing.setAttribute('transform', `scale(${artwork.width / artwork.nativeWidth})`);
    for (const { layer, source } of sources) {
      // Excalidraw exports can reuse mask/image IDs. Scope each before combining them.
      const ids = new Map([...source.querySelectorAll('[id]')].map(node => [node.id, `${layer.id}-${node.id}`]));
      for (const node of source.querySelectorAll('*')) {
        for (const attr of [...node.attributes]) {
          let value = attr.value;
          if (attr.name === 'id') value = ids.get(value);
          else if (value.startsWith('#') && ids.has(value.slice(1))) value = `#${ids.get(value.slice(1))}`;
          else value = value.replace(/url\(#([^)]*)\)/g, (match, id) => ids.has(id) ? `url(#${ids.get(id)})` : match);
          node.setAttributeNS(attr.namespaceURI, attr.name, value);
        }
      }
      const placed = document.createElementNS(svgNS, 'g');
      placed.setAttribute('transform', `translate(${layer.x} ${layer.y}) scale(${layer.scale})`);
      const content = document.createElementNS(svgNS, 'g');
      if (layer.clip) {
        const clip = document.createElementNS(svgNS, 'clipPath'); clip.id = `${layer.id}-crop`;
        const path = document.createElementNS(svgNS, 'path'); path.setAttribute('d', layer.clip);
        clip.append(path); defs.append(clip); content.setAttribute('clip-path', `url(#${clip.id})`);
      }
      for (const child of [...source.children]) {
        if (child.localName === 'defs') for (const def of child.children) defs.append(document.importNode(def, true));
        else content.append(document.importNode(child, true));
      }
      placed.append(content); drawing.append(placed);
    }
    defs.append(drawing); svg.append(defs);
    const use = document.createElementNS(svgNS, 'use'); use.setAttribute('href', '#pi-drawing'); svg.append(use);
    $('artwork').replaceChildren(svg);
    const focus = document.createElementNS(svgNS, 'svg');
    focus.setAttribute('preserveAspectRatio', 'none'); focus.append(use.cloneNode());
    $('lifted').replaceChildren(focus);
    await document.fonts.ready;
    measureReading(); render(); fit(false); $('loading').hidden = true;
  } catch (error) {
    console.error(error); $('loading').hidden = true; $('asset-error').hidden = false;
  }
}
$('retry').addEventListener('click', loadArtwork);
export function applyCopy(doc) {
  validateCopy(doc, new Set(entries.map(entry => entry.id)));
  for (const entry of entries) {
    const body = document.querySelector('.passage-body[data-section="' + entry.id + '"]');
    if (Object.hasOwn(doc.sections, entry.id)) body.replaceChildren(renderNodes(doc.sections[entry.id], document));
    else body.replaceChildren(...entry.paragraphs.map(paragraph));
  }
  measureReading();
}
export function previewPassage(id) { activate(id, { force: true }); }
export function focusPassage(id) { jumpTo(id, { historyMode: 'replace', animate: false }); }
export function refreshReadingMeasurements() { measureReading(); }
buildArticle(); buildContents(); buildHotspots();
try {
  const response = await fetch('./text.json', { cache: 'no-cache' });
  if (response.ok) applyCopy(await response.json());
} catch (error) { console.warn('Saved text could not load; showing the original copy.', error); }
ready = true; measureReading(); readRoute(); loadArtwork();
