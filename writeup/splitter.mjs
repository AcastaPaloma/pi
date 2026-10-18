// Draggable dividers between the panels of a two-track grid. The split lives in
// --split-x (columns) or --split-y (rows) on the grid, so CSS keeps each layout's
// default and its media queries; a drag or arrow key overrides it, per viewer.
const KEY = 'pi-splits-v1';
const MIN = 120; // px each panel keeps
function load() { try { return JSON.parse(localStorage.getItem(KEY)) ?? {}; } catch { return {}; } }
function save(splits) { try { localStorage.setItem(KEY, JSON.stringify(splits)); } catch {} }

export function addSplitter(grid, { name, label = 'Resize panels' }) {
  const handle = document.createElement('div');
  handle.className = 'splitter'; handle.tabIndex = 0;
  handle.setAttribute('role', 'separator'); handle.setAttribute('aria-label', label);
  handle.setAttribute('aria-valuemin', '0'); handle.setAttribute('aria-valuemax', '100');
  grid.append(handle);
  // Two columns split side by side; otherwise (phones) the two rows split top and bottom.
  const axis = () => getComputedStyle(grid).gridTemplateColumns.trim().split(/\s+/).length > 1 ? 'x' : 'y';
  const id = a => `${name}-${a}`, variable = a => `--split-${a}`;
  const size = a => a === 'x' ? grid.clientWidth : grid.clientHeight;
  const current = a => { const rect = handle.getBoundingClientRect(), box = grid.getBoundingClientRect();
    return a === 'x' ? (rect.left + rect.width / 2 - box.left) / box.width * 100 : (rect.top + rect.height / 2 - box.top) / box.height * 100; };
  function apply(a, percent, persist) {
    const room = Math.min(45, MIN / Math.max(1, size(a)) * 100);
    const value = Math.max(room, Math.min(100 - room, percent));
    grid.style.setProperty(variable(a), `${value.toFixed(2)}%`);
    handle.setAttribute('aria-valuenow', String(Math.round(value)));
    if (persist) { const splits = load(); splits[id(a)] = value; save(splits); }
  }
  function layout() {
    const a = axis(); handle.dataset.axis = a;
    handle.setAttribute('aria-orientation', a === 'x' ? 'vertical' : 'horizontal');
    const stored = load()[id(a)];
    if (stored != null) apply(a, stored, false); else handle.setAttribute('aria-valuenow', String(Math.round(current(a))));
  }
  handle.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    event.preventDefault(); handle.setPointerCapture(event.pointerId);
    handle.classList.add('is-dragging'); document.documentElement.dataset.resizing = handle.dataset.axis;
  });
  handle.addEventListener('pointermove', event => {
    if (!handle.hasPointerCapture(event.pointerId)) return;
    const a = handle.dataset.axis, box = grid.getBoundingClientRect();
    apply(a, a === 'x' ? (event.clientX - box.left) / box.width * 100 : (event.clientY - box.top) / box.height * 100, true);
  });
  const end = event => {
    if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
    handle.classList.remove('is-dragging'); delete document.documentElement.dataset.resizing;
  };
  handle.addEventListener('pointerup', end); handle.addEventListener('pointercancel', end);
  handle.addEventListener('dblclick', () => {
    const a = handle.dataset.axis, splits = load(); delete splits[id(a)]; save(splits);
    grid.style.removeProperty(variable(a)); layout();
  });
  handle.addEventListener('keydown', event => {
    const a = handle.dataset.axis, back = a === 'x' ? 'ArrowLeft' : 'ArrowUp', forward = a === 'x' ? 'ArrowRight' : 'ArrowDown';
    if (event.key !== back && event.key !== forward) return;
    event.preventDefault(); event.stopPropagation();
    apply(a, current(a) + (event.key === forward ? 1 : -1) * (event.shiftKey ? 10 : 2), true);
  });
  new ResizeObserver(layout).observe(grid); layout();
  return handle;
}
