import { artwork, macros, steps, details, annotations } from './content.js';
import { clamp, fitBounds, zoomAt, constrainCamera } from './camera.mjs';

const $ = id => document.getElementById(id);
const viewport = $('viewport'), world = $('world');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const detailById = new Map(details.map(item => [item.id, item]));
const allBounds = [0, 0, artwork.width, artwork.height];
const svgNS = 'http://www.w3.org/2000/svg';
let stepIndex = -1, pinned = null, preview = null, renderedMacro = null;
let camera = { x: 0, y: 0, scale: 1 }, animation = 0, frame = 0;
let hoverTimer = 0, leaveTimer = 0, suppressHoverUntil = 0, moved = false;
let activeGesture = null;
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
  suppressHoverUntil = performance.now() + 180; clearTimers();
  if (preview) { preview = null; renderSelection(); }
}
function zoom(factor, point = { x: viewport.clientWidth / 2, y: viewport.clientHeight / 2 }, animate = false) {
  manuallyMoved(); setCamera(zoomAt(camera, point, clamp(camera.scale * factor, minScale(), 8)), animate);
}
function setRoute(index, detail = null, { historyMode = 'push', animate = true } = {}) {
  clearTimers();
  if (document.activeElement?.closest('#hotspots')) viewport.focus({ preventScroll: true });
  stepIndex = index >= 0 && index < steps.length ? index : -1;
  pinned = detail && currentStep()?.details.includes(detail) ? detail : null;
  preview = null;
  if (historyMode !== 'none') {
    const hash = currentStep() ? `#${currentStep().id}${pinned ? `/${pinned}` : ''}` : '';
    const url = `${location.pathname}${location.search}${hash}`;
    if (url !== `${location.pathname}${location.search}${location.hash}`) history[historyMode === 'replace' ? 'replaceState' : 'pushState'](null, '', url);
  }
  render(); fit(animate);
  $('announcement').textContent = detailById.get(pinned)?.title ?? currentStep()?.title ?? 'Whole canvas';
}
function readRoute() {
  const [id, detail] = location.hash.slice(1).split('/');
  setRoute(steps.findIndex(item => item.id === id), detail, { historyMode: 'none', animate: false });
}
function advance(direction) {
  if (stepIndex === steps.length - 1 && direction > 0) { setRoute(-1); return; }
  setRoute(clamp(stepIndex + direction, 0, steps.length - 1));
}
function go(index, detail) { setRoute(index, detail); viewport.focus({ preventScroll: true }); }
function renderContents() {
  const macro = currentStep()?.macro ?? 'ingestion';
  for (const button of $('macro-tabs').children) button.setAttribute('aria-current', String(button.dataset.macro === macro));
  if (renderedMacro === macro) return;
  renderedMacro = macro;
  const fragment = document.createDocumentFragment();
  steps.forEach((step, index) => {
    if (step.macro !== macro) return;
    const group = make('div', undefined, 'toc-group');
    const button = make('button', undefined, 'toc-link toc-major');
    button.type = 'button'; button.dataset.step = step.id;
    button.append(make('span', String(index + 1).padStart(2, '0'), 'toc-number'), make('span', step.title));
    button.addEventListener('click', () => go(index)); group.append(button);
    for (const id of step.details) {
      const item = detailById.get(id), link = make('button', item.title, 'toc-link toc-detail');
      link.type = 'button'; link.dataset.detail = id;
      link.addEventListener('click', () => go(index, id)); group.append(link);
    }
    fragment.append(group);
  });
  $('toc').replaceChildren(fragment); $('notes-scroll').scrollTop = 0;
}
function renderHotspots() {
  const group = $('hotspots'); group.replaceChildren();
  const indices = stepIndex < 0 ? [0, 6] : [];
  for (const index of indices) {
    const step = steps[index], button = make('button', undefined, 'hotspot');
    button.type = 'button'; button.setAttribute('aria-label', `Explore ${step.title}`);
    position(button, step.bounds);
    button.addEventListener('click', () => { if (!moved) go(index); }); group.append(button);
  }
  // Macro views expose every detail in that chunk, just like their contents list.
  const items = currentStep()?.details.length ? currentStep().details.map(id => detailById.get(id)) :
    details.filter(item => currentStep() && ['context', 'decision'].includes(currentStep().id) && steps.find(s => s.id === item.step)?.macro === currentStep().macro);
  for (const item of items) {
    const button = make('button', undefined, 'hotspot is-available');
    button.type = 'button'; button.dataset.detail = item.id;
    button.setAttribute('aria-label', item.title);
    position(button, item.bounds);
    button.addEventListener('pointerenter', event => {
      clearTimers();
      if (event.pointerType === 'touch' || pointers.size || performance.now() < suppressHoverUntil) return;
      hoverTimer = setTimeout(() => { preview = item.id; renderSelection(); }, 140);
    });
    button.addEventListener('pointerleave', () => { clearTimers(); leaveTimer = setTimeout(() => { preview = null; renderSelection(); }, 120); });
    button.addEventListener('focus', () => { preview = item.id; renderSelection(); });
    button.addEventListener('blur', () => { preview = null; renderSelection(); });
    button.addEventListener('click', () => { if (!moved) go(steps.findIndex(s => s.id === item.step), item.id); });
    group.append(button);
  }
}
function renderSelection() {
  const selected = preview ?? pinned;
  for (const link of $('toc').querySelectorAll('button')) {
    const active = pinned ? link.dataset.detail === pinned : link.dataset.step === currentStep()?.id;
    if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
    link.classList.toggle('is-previewed', Boolean(preview && link.dataset.detail === preview));
  }
  $('detail-outline').hidden = !selected;
  if (selected) position($('detail-outline'), detailById.get(selected).bounds);
  const paragraphs = annotations[selected ?? currentStep()?.id] ?? [];
  $('annotation').hidden = !paragraphs.length;
  $('annotation').replaceChildren(...paragraphs.map(text => make('p', text)));
}
function render() {
  const active = stepIndex >= 0;
  world.classList.toggle('focused', active);
  $('start').replaceChildren(document.createTextNode(active ? 'Whole canvas ' : 'Start walkthrough '), make('span', active ? '↖' : '→'));
  $('start').lastChild.setAttribute('aria-hidden', 'true');
  $('lifted').hidden = !active;
  if (active) {
    const bounds = detailById.get(pinned)?.bounds ?? currentStep().bounds;
    position($('lifted'), bounds);
    $('lifted').querySelector('svg')?.setAttribute('viewBox', bounds.join(' '));
  }
  renderContents(); renderHotspots(); renderSelection();
  const current = $('toc').querySelector('[aria-current="location"]');
  if (current) {
    const scroller = $('notes-scroll'), a = current.getBoundingClientRect(), b = scroller.getBoundingClientRect();
    if (a.top < b.top || a.bottom > b.bottom) scroller.scrollTop += a.top - b.top - 16;
  }
}
for (const macro of macros) {
  const button = make('button', macro.title, 'macro-tab');
  button.type = 'button'; button.dataset.macro = macro.id;
  button.addEventListener('click', () => go(steps.findIndex(s => s.id === macro.entry)));
  $('macro-tabs').append(button);
}
$('start').addEventListener('click', () => go(stepIndex < 0 ? 0 : -1));
document.querySelector('.skip-link').addEventListener('click', event => { event.preventDefault(); $('macro-tabs').querySelector('button').focus({ preventScroll: true }); });
window.addEventListener('hashchange', readRoute);
window.addEventListener('popstate', readRoute);
document.addEventListener('keydown', event => {
  if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey || event.target.closest('input,textarea,select,[contenteditable="true"]')) return;
  const nativeSpace = event.key === ' ' && event.target.closest('button,a,summary');
  if (event.key === 'Escape') { event.preventDefault(); if (pinned) go(stepIndex); else if (stepIndex >= 0) go(-1); return; }
  if (event.key === 'ArrowRight' || event.key === 'ArrowLeft' || (event.key === ' ' && !nativeSpace)) {
    if (stepIndex < 0) return;
    event.preventDefault(); if (!event.repeat) advance(event.key === 'ArrowLeft' || (event.key === ' ' && event.shiftKey) ? -1 : 1);
  } else if (event.key === '+' || event.key === '=') { event.preventDefault(); zoom(1.2); }
  else if (event.key === '-') { event.preventDefault(); zoom(1 / 1.2); }
  else if (event.key.toLowerCase() === 'f') { event.preventDefault(); fit(); }
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
    manuallyMoved();
    setCamera({ ...camera, x: camera.x - (event.shiftKey ? event.deltaY : event.deltaX) * factor, y: camera.y - (event.shiftKey ? event.deltaX : event.deltaY) * factor });
  }
}, { passive: false });
viewport.addEventListener('dblclick', event => { if (!event.target.closest('button')) zoom(event.shiftKey ? .65 : 1.5, localPoint(event), true); });
// A camera owns this coordinate system. Native focus scrolling must never
// create an untracked second translation (also covers older overflow:clip support).
viewport.addEventListener('scroll', () => {
  if (viewport.scrollTop || viewport.scrollLeft) viewport.scrollTo(0, 0);
});
new ResizeObserver(() => fit(false)).observe(viewport);


async function loadArtwork() {
  $('loading').hidden = false; $('asset-error').hidden = true;
  try {
    const response = await fetch(artwork.src);
    if (!response.ok) throw new Error(`Artwork: ${response.status}`);
    const parsed = new DOMParser().parseFromString(await response.text(), 'image/svg+xml');
    if (parsed.querySelector('parsererror')) throw new Error('Invalid SVG');
    const source = parsed.documentElement;
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('viewBox', allBounds.join(' ')); svg.setAttribute('aria-hidden', 'true');
    const defs = document.createElementNS(svgNS, 'defs');
    const drawing = document.createElementNS(svgNS, 'g');
    drawing.id = 'pi-drawing'; drawing.setAttribute('transform', `scale(${artwork.width / artwork.nativeWidth})`);
    for (const child of [...source.children]) {
      if (child.localName === 'defs') for (const def of child.children) defs.append(document.importNode(def, true));
      else drawing.append(document.importNode(child, true));
    }
    defs.append(drawing); svg.append(defs);
    const use = document.createElementNS(svgNS, 'use'); use.setAttribute('href', '#pi-drawing'); svg.append(use);
    $('artwork').replaceChildren(svg);
    const focus = document.createElementNS(svgNS, 'svg');
    focus.setAttribute('preserveAspectRatio', 'none'); focus.append(use.cloneNode());
    $('lifted').replaceChildren(focus);
    await document.fonts.ready;
    render(); fit(false); $('loading').hidden = true;
  } catch (error) {
    console.error(error); $('loading').hidden = true; $('asset-error').hidden = false;
  }
}
$('retry').addEventListener('click', loadArtwork);
readRoute(); loadArtwork();
