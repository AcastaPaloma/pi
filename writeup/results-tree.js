// Results tab, Tree view: every experiment as a lineage tree. Each node hangs from the checkpoint its runs resumed
// from (results-runs.js `lineage`, read from runs/<run>/run.json on the training boxes), so new experiments attach
// to the right parent on every sync. Hand-drawn SVG; the trunk is the path to the newest kept experiment, drawn
// straight along the top, with side branches below it. Hover or focus shows a small card; click selects.

const NS = 'http://www.w3.org/2000/svg';
const DX = 124, DY = 32, PAD = { l: 16, r: 190, t: 22, b: 26 };
const STROKE = { main: '#1f3a8a', dead: '#9a9a93', running: '#1f3a8a', queued: '#1f3a8a', finished: '#1f3a8a', root: '#1a1a1a' };

function svg(tag, attrs = {}) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

// The lineage parent of every node: an explicit `from`, else the node that owns the checkpoint one of its runs
// resumed from, else the queue entry's parent. Returns { id, step } or null.
export function lineageParents(nodes, lineage, lossPlan, roots) {
  const owner = new Map();
  const runsOf = n => [...(n.runs || []), ...((lossPlan[n.id] || {}).lines || []).map(l => l[0])];
  for (const n of nodes) for (const r of runsOf(n)) if (!owner.has(r)) owner.set(r, n.id);
  const ids = new Set([...nodes.map(n => n.id), ...roots.map(r => r.id)]);
  const parent = new Map();
  for (const n of nodes) {
    let p = null;
    if (n.from) p = { id: n.from };
    else {
      for (const r of runsOf(n)) {
        const src = lineage[r];
        const o = src && owner.get(src[0]);
        if (o && o !== n.id) { p = { id: o, step: src[1] }; break; }
      }
      if (!p && n.parent) p = { id: n.parent };
    }
    parent.set(n.id, p && ids.has(p.id) ? p : null);
  }
  return parent;
}

export function mountTree({ nodes, views, lineage, lossPlan, roots, select, describe }) {
  const scroll = document.getElementById('lineage-scroll');
  const card = document.getElementById('lineage-card');
  const wrap = document.getElementById('lineage');
  const parent = lineageParents(nodes, lineage, lossPlan, roots);
  const byId = new Map([...roots.map(r => [r.id, { ...r, root: true }]), ...nodes.map(n => [n.id, n])]);
  const order = new Map(nodes.map((n, i) => [n.id, i]));
  const kids = new Map();
  for (const n of nodes) {
    const p = parent.get(n.id);
    const key = p ? p.id : null;
    if (!kids.has(key)) kids.set(key, []);
    kids.get(key).push(n.id);
  }
  const stateOf = id => byId.get(id).root ? 'root' : views.get(id).state;
  // Layout, like a tree growing upward: each node's trunk (the child whose subtree holds the newest kept
  // experiment, else its largest training branch) runs along the bottom row; side branches stack just above it,
  // deeper nodes' branches nearest the trunk and the newest first, so every edge stays short. Tests go last.
  const newestMain = new Map(), size = new Map();
  const score = id => {
    if (newestMain.has(id)) return newestMain.get(id);
    const own = byId.get(id).kind === 'main' ? order.get(id) : -1;
    const best = Math.max(own, ...(kids.get(id) || []).map(score));
    newestMain.set(id, best);
    return best;
  };
  const count = id => size.get(id) ?? (size.set(id, 1 + (kids.get(id) || []).reduce((a, c) => a + count(c), 0)), size.get(id));
  const isTest = id => Boolean(byId.get(id).robust);
  const trunkOf = list => {
    const train = list.filter(id => !isTest(id));
    if (!train.length) return null;
    return train.reduce((best, id) => {
      const k = [score(id), count(id), order.get(id)], b = [score(best), count(best), order.get(best)];
      return k[0] > b[0] || (k[0] === b[0] && (k[1] > b[1] || (k[1] === b[1] && k[2] > b[2]))) ? id : best;
    }, train[0]);
  };
  const pos = new Map();
  let maxDepth = 0;
  // Returns rows bottom to top; rows[0] is this node's trunk row.
  const block = (id, depth) => {
    maxDepth = Math.max(maxDepth, depth);
    const ch = kids.get(id) || [];
    const t = trunkOf(ch);
    const rows = t ? block(t, depth + 1) : [[]];
    rows[0].push([id, depth]);
    const sides = ch.filter(c => c !== t).sort((a, b) => (isTest(a) - isTest(b)) || order.get(b) - order.get(a));
    for (const c of sides) rows.push(...block(c, depth + 1));
    return rows;
  };
  const tops = [...roots.map(r => r.id).filter(id => kids.has(id)), ...(kids.get(null) || [])];
  const allRows = [];
  for (const id of [...tops].reverse()) allRows.push(...block(id, 0));  // the first root's tree ends up on top
  const row = allRows.length;
  allRows.forEach((r, i) => r.forEach(([id, d]) => pos.set(id, { d, r: row - 1 - i })));
  const W = PAD.l + maxDepth * DX + PAD.r, H = PAD.t + Math.max(0, row - 1) * DY + PAD.b + 10;
  const X = id => PAD.l + pos.get(id).d * DX + 8, Y = id => PAD.t + pos.get(id).r * DY;

  const root = svg('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: 'lineage-svg', role: 'group',
    'aria-label': 'Lineage tree of every experiment' });
  const edges = svg('g', { class: 'edges' }), marks = svg('g', { class: 'marks' });
  root.append(edges, marks);
  // Edges as lanes: one vertical spine per parent just right of its dot, a short rounded stub into each child.
  const rank = { dead: 0, queued: 1, finished: 2, running: 3, main: 4 };
  const pairs = [...parent].filter(([id, p]) => p && pos.has(id) && pos.has(p.id))
    .sort((a, b) => (rank[stateOf(a[0])] ?? 2) - (rank[stateOf(b[0])] ?? 2));
  for (const [id, p] of pairs) {
    const x0 = X(p.id), y0 = Y(p.id), x1 = X(id), y1 = Y(id), st = stateOf(id), sx = x0 + 18, r = 7;
    const d = y0 === y1 ? `M${x0},${y0}H${x1}` : `M${sx},${y0}V${y1 + r}Q${sx},${y1} ${sx + r},${y1}H${x1}`;
    edges.append(svg('path', { d, class: `edge edge-${st}${byId.get(id).robust ? ' edge-test' : ''}` }));
  }
  const groups = new Map();
  for (const id of pos.keys()) {
    const n = byId.get(id), st = stateOf(id), x = X(id), y = Y(id);
    const g = svg('g', { class: `mark mark-${st}`, transform: `translate(${x},${y})`, tabindex: n.root ? -1 : 0,
      role: n.root ? 'img' : 'button', 'aria-label': n.root ? `${n.name}: ${n.note}` : describe(n).label, 'data-id': id });
    if (n.root) {
      g.append(svg('rect', { x: -5, y: -5, width: 10, height: 10, rx: 1.5, class: 'dot' }));
    } else {
      const v = views.get(id);
      if (v.score) {  // thin ring: the score out of its maximum
        const c = 2 * Math.PI * 10, f = Math.max(0.02, Math.min(1, v.score.value / v.score.of));
        g.append(svg('circle', { r: 10, class: 'ring-base' }),
          svg('circle', { r: 10, class: 'ring', 'stroke-dasharray': `${(f * c).toFixed(1)} ${c.toFixed(1)}`, transform: 'rotate(-90)' }));
      }
      if (st === 'running') g.append(svg('circle', { r: 6, class: 'pulse' }));
      g.append(svg('circle', { r: 5.5, class: 'dot' }));
      if (st === 'dead' && !(kids.get(id) || []).length) g.append(svg('path', { d: 'M15,-6V6', class: 'end' }));
    }
    const leaf = !(kids.get(id) || []).length, bottom = pos.get(id).r === row - 1;
    // Leaves: label to the right. Inner nodes: above their line, or below it on the bottom (trunk) row.
    const label = svg('text', { x: leaf ? (st === 'dead' ? 22 : 16) : 2, y: leaf ? 4 : bottom ? 22 : -12, class: 'label' });
    label.textContent = n.root ? n.name : n.short || n.name;
    g.append(label);
    marks.append(g);
    groups.set(id, g);
  }
  scroll.textContent = '';
  scroll.append(root);

  let zoom = 1, pinned = null;
  const fit = () => {
    const avail = scroll.clientWidth || W;
    zoom = Math.max(0.72, Math.min(1, avail / W));
    apply();
  };
  const apply = () => { root.setAttribute('width', Math.round(W * zoom)); root.setAttribute('height', Math.round(H * zoom)); };
  document.getElementById('lineage-in').addEventListener('click', () => { zoom = Math.min(1.6, zoom * 1.2); apply(); });
  document.getElementById('lineage-out').addEventListener('click', () => { zoom = Math.max(0.45, zoom / 1.2); apply(); });

  const showCard = id => {
    const n = byId.get(id), g = groups.get(id);
    card.textContent = '';
    const el = (tag, cls, text) => { const e = document.createElement(tag); e.className = cls; e.textContent = text; return e; };
    if (n.root) {
      card.append(el('b', 'card-name', n.name), el('p', 'card-note', n.note), el('p', 'card-meta', 'Starting point: not trained here.'));
    } else {
      const info = describe(n), p = parent.get(id);
      card.append(el('span', 'card-when', info.when), el('b', 'card-name', n.name), el('p', 'card-note', info.note));
      if (info.score) card.append(el('p', 'card-score', info.score));
      if (info.progress) card.append(el('p', 'card-meta', info.progress));
      if (p) card.append(el('p', 'card-meta', `From: ${byId.get(p.id).short || byId.get(p.id).name}${p.step ? `, step ${p.step.toLocaleString('en-US')}` : ''}`));
    }
    card.hidden = false;
    const box = wrap.getBoundingClientRect(), r = g.getBoundingClientRect();
    const left = Math.min(box.width - card.offsetWidth - 8, Math.max(8, r.left - box.left + 18));
    const below = r.bottom - box.top + 8, above = r.top - box.top - card.offsetHeight - 8;
    card.style.left = `${left}px`;
    card.style.top = `${above > 40 && below + card.offsetHeight > box.height ? above : below}px`;
  };
  const hideCard = () => { if (!pinned) card.hidden = true; };
  for (const [id, g] of groups) {
    const n = byId.get(id);
    g.addEventListener('pointerenter', e => { if (e.pointerType !== 'touch') showCard(id); });
    g.addEventListener('pointerleave', e => { if (e.pointerType !== 'touch') hideCard(); });
    g.addEventListener('focus', () => showCard(id));
    g.addEventListener('blur', hideCard);
    if (!n.root) {
      g.addEventListener('click', () => { pinned = null; select(n, 0, true); showCard(id); });
      g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(n, 0, true); } });
    }
  }
  scroll.addEventListener('scroll', () => { card.hidden = true; });
  new ResizeObserver(() => { if (!wrap.hidden) fit(); }).observe(scroll);

  return {
    fit,
    highlight(id) {
      for (const [gid, g] of groups) g.classList.toggle('is-current', gid === id);
      const g = groups.get(id);
      if (g && !wrap.hidden) {
        const box = scroll.getBoundingClientRect(), r = g.getBoundingClientRect();
        if (r.left < box.left || r.right > box.right) scroll.scrollLeft += r.left - box.left - box.width / 3;
      }
    },
  };
}
