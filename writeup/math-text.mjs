// Display-only prose formatting: math runs (shapes, equations, subscripted
// names) and cross-references. Saved text never changes. Every decorated span
// keeps its exact source in data-source, so editing and saving see plain text.

const FUNCTIONS = new Set(['Linear', 'Embedding', 'GELU', 'softmax', 'log-softmax', 'log', 'ln', 'exp', 'mean']);
const PREFIX = new Set(['log', 'ln', 'exp']); // may apply without parentheses: −log p
const NAMED = new Set(['gate', 'up', 'mask']); // operands named in words inside formulas
const LAYERS = new Set(['Linear', 'Embedding']); // learned layers stand out from plain functions
const SUB = '₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜᵢⱼᵣᵤᵥ';
const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼ⁿⁱᵀ';
const BINARY = '=+−×·/÷→←≈∈⊙@|≤≥<>*';
const MULTIPLY = '×·⊙@*';
const greek = c => /[α-ωΑ-Ωϵϑϕℒ]/.test(c) && c !== 'π' && c !== 'ο';
const latin = c => /[A-Za-zÀ-ÖØ-öø-ÿℓ]/.test(c);
const has = (set, c) => !!c && set.includes(c);
const wordChar = c => !!c && /[\p{L}\p{N}_]/u.test(c) && !SUB.includes(c) && !SUP.includes(c);

function closing(s, i, end) {
  // Index of the bracket matching s[i], or -1. Nested ()[]{} are respected.
  const pairs = { '(': ')', '[': ']', '{': '}' }, stack = [];
  for (let j = i; j < end; j++) {
    if (pairs[s[j]]) stack.push(pairs[s[j]]);
    else if (')]}'.includes(s[j])) { if (stack.pop() !== s[j]) return -1; if (!stack.length) return j; }
  }
  return -1;
}
function decorations(s, i, end, node) {
  // Unicode sub/superscripts, primes, a postfix star (A*), and _x or _{…} subscripts.
  node.deco = [];
  while (i < end) {
    const c = s[i];
    if (/[\u0300-\u036f]/.test(c) || SUB.includes(c) || SUP.includes(c) || c === '′' || c === '″') { node.deco.push({ text: c }); i++; }
    else if (c === "'" && !latin(s[i + 1] ?? '')) { node.deco.push({ text: '′' }); i++; }
    else if (c === '*' && !/[A-Za-z0-9(]/.test(s[i + 1] ?? '')) { node.deco.push({ star: true }); i++; }
    else if (c === '_' && s[i + 1] === '{') {
      const j = closing(s, i + 1, end); if (j < 0) break;
      const inner = s.slice(i + 2, j), parsed = sequence(inner, 0, inner.length, { lenient: true });
      node.deco.push({ sub: parsed && parsed.end === inner.length ? parsed.nodes : [{ t: 'word', text: inner }] }); i = j + 1;
    } else if (c === '_') {
      const m = /^(?:[A-Za-z0-9]+|[α-ωΑ-Ω])/.exec(s.slice(i + 1, end)); if (!m) break;
      node.deco.push({ sub: [m[0].length === 1 && !/\d/.test(m[0]) ? { t: 'var', base: m[0], deco: [] } : /^\d+$/.test(m[0]) ? { t: 'num', text: m[0] } : { t: 'word', text: m[0] }] });
      i += 1 + m[0].length;
    } else break;
  }
  return i;
}
function call(s, i, end, node) {
  // f(args): the arguments are a comma list in which named operands may appear.
  const j = closing(s, i, end); if (j < 0) return null;
  const parsed = sequence(s, i + 1, j, { list: true, call: true });
  if (!parsed || parsed.end !== j) return null;
  node.args = parsed.nodes; return decorations(s, j + 1, end, node);
}
function operand(s, i, end, ctx) {
  // One operand at i, or null. strength: strong (counts as math on its own),
  // weak (math only beside an operator), num, or gap (an ellipsis).
  const c = s[i];
  if (c === '[') {
    const j = closing(s, i, end); if (j < 0) return null;
    const parsed = sequence(s, i + 1, j, { list: true, lenient: true });
    if (!parsed || parsed.end !== j || !s.slice(i + 1, j).includes(',')) return null;
    return { end: j + 1, node: { t: 'group', open: '[', close: ']', shape: true, nodes: parsed.nodes }, strength: 'strong' };
  }
  if (c === '(') {
    const j = closing(s, i, end); if (j < 0) return null;
    const parsed = sequence(s, i + 1, j, { list: true, lenient: ctx.lenient });
    if (parsed && parsed.end === j) {
      const node = { t: 'group', open: '(', close: ')', nodes: parsed.nodes };
      return { end: decorations(s, j + 1, end, node), node, strength: parsed.math ? 'strong' : 'num' };
    }
    // "· (predicted velocity)": a few plain words, only right after an operator.
    const words = s.slice(i + 1, j);
    if (ctx.afterOperator && /^[a-z]+(?: [a-z]+){0,3}$/.test(words))
      return { end: j + 1, node: { t: 'group', open: '(', close: ')', nodes: [{ t: 'word', text: words }] }, strength: 'num' };
    return null;
  }
  const number = /^\d+(?:\.\d+)?(?:[A-Z](?![\p{L}\p{N}]))?/u.exec(s.slice(i, end));
  if (number) {
    if (wordChar(s[i + number[0].length])) return null;
    return { end: i + number[0].length, node: { t: 'num', text: number[0] }, strength: 'num' };
  }
  if (c === '∞') return { end: i + 1, node: { t: 'sym', text: c }, strength: 'strong' };
  if (c === '…' || c === '⋯') return { end: i + 1, node: { t: 'sym', text: c }, strength: 'gap' };
  if (greek(c)) {
    const node = { t: 'var', base: c };
    let next = decorations(s, i + 1, end, node);
    if (s[next] === '(' && !ctx.noCall) { const after = call(s, next, end, node); if (after) { node.t = 'fn'; node.name = c; node.italic = true; next = after; } }
    return { end: next, node, strength: 'strong' };
  }
  const word = /^[A-Za-zÀ-ÖØ-öø-ÿℓ][A-Za-z0-9À-ÖØ-öø-ÿ]*(?:-[A-Za-z][A-Za-z0-9]*)*/.exec(s.slice(i, end));
  if (!word) return null;
  const text = word[0], after = i + text.length;
  if (FUNCTIONS.has(text)) {
    if (s[after] === '(') {
      const node = { t: 'fn', name: text }, next = call(s, after, end, node);
      if (next) return { end: next, node, strength: 'strong' };
    }
    if (PREFIX.has(text) && s[after] === ' ') return { end: after, node: { t: 'fn', name: text, prefix: true }, strength: 'prefix' };
    return null;
  }
  if (s[after] === '_' || (text.length === 1 && (s[after] === '*' || s[after] === "'" || has(SUP, s[after]) || has(SUB, s[after]) || has('′″', s[after]) || /[\u0300-\u036f]/.test(s[after] ?? '')))) {
    if (wordChar(s[after]) && s[after] !== '_') return null;
    const node = { t: 'var', base: text }, next = decorations(s, after, end, node);
    if (node.deco.length) {
      if (s[next] === '(' && text.length === 1) { const fn = call(s, next, end, node); if (fn) { node.t = 'fn'; node.name = text; node.italic = true; return { end: fn, node, strength: 'strong' }; } }
      return { end: next, node, strength: 'strong' };
    }
  }
  if (wordChar(s[after]) || s[after] === '’' || (s[after] === "'" && latin(s[after + 1] ?? ''))) return null;
  if (ctx.lenient) return { end: after, node: text.length === 1 ? { t: 'var', base: text, deco: [] } : { t: 'word', text }, strength: 'strong' };
  if (NAMED.has(text) && (ctx.afterOperator || ctx.call)) return { end: after, node: { t: 'word', text }, strength: 'weak' };
  // "Loss = …": a capitalized word that names the left-hand side.
  if (/^[A-Z][a-z]+$/.test(text) && /^\s?=\s/.test(s.slice(after, end))) return { end: after, node: { t: 'word', text }, strength: 'weak' };
  if (text === 'BV') return { end: after, node: { t: 'word', text }, strength: 'weak' };
  if (text.length !== 1) return null;
  const before = s.slice(Math.max(0, i - 7), i);
  if (/['’.\-]$/.test(before) || /\bOption $/.test(before) || text === 'a' || text === 'I') return null;
  if (s[after] === '.' && latin(s[after + 1] ?? '')) return null; // e.g., i.e.
  const node = { t: 'var', base: text, deco: [] };
  if (s[after] === '(' && !ctx.noCall) { const fn = call(s, after, end, node); if (fn) { node.t = 'fn'; node.name = text; node.italic = true; return { end: fn, node, strength: 'strong' }; } }
  return { end: after, node, strength: text === 'A' ? 'weak' : 'strong' };
}
function binary(s, i, end, ctx, spaced) {
  const c = s[i];
  if (c === ',' && ctx.list) return ',';
  if (c === ':' && spaced && s[i + 1] === ' ') return ':';
  if (c === '-' && spaced && s[i + 1] === ' ') return '-';
  if (s.startsWith('mod ', i) && spaced) return 'mod';
  if (BINARY.includes(c) && i < end) return c;
  return null;
}
function sequence(s, start, end, ctx = {}) {
  // Operands joined by operators (or juxtaposed, as in Q Kᵀ). Returns the
  // longest prefix that ends on an operand, or null.
  const nodes = [];
  let i = start, expect = true, good = null, math = false, last = null, afterOperator = false;
  while (i < end) {
    let j = i; while (j < end && s[j] === ' ') j++;
    const space = s.slice(i, j);
    if (expect) {
      if ('−√<>≤≥'.includes(s[j]) || (s[j] === '-' && /\d/.test(s[j + 1] ?? '')) || (ctx.lenient && BINARY.includes(s[j]))) {
        if (space && nodes.length) nodes.push({ t: 'sp', text: space });
        nodes.push({ t: 'op', text: s[j], unary: true }); math = true; afterOperator = true; i = j + 1; continue;
      }
      const found = j < end ? operand(s, j, end, { ...ctx, afterOperator }) : null;
      if (!found) break;
      if (space && nodes.length) nodes.push({ t: 'sp', text: space });
      nodes.push(found.node); i = found.end; afterOperator = false;
      if (found.strength === 'prefix') { math = true; continue; }
      if (found.strength === 'strong') math = true;
      last = found.strength; expect = false; good = { i, count: nodes.length, math };
    } else {
      const op = binary(s, j, end, ctx, space.length > 0);
      if (op) {
        if (space) nodes.push({ t: 'sp', text: space });
        nodes.push({ t: 'op', text: op }); if (op !== ',') math = true;
        i = j + op.length; expect = true; afterOperator = op !== ','; continue;
      }
      // Juxtaposition (Q Kᵀ, Σ α·v): never next to a bare number or weak name.
      if (last === 'num' || last === 'weak' || j >= end) break;
      const found = operand(s, j, end, { ...ctx, noCall: false });
      if (!found || found.strength === 'num' || found.strength === 'weak') break;
      if (space) nodes.push({ t: 'sp', text: space });
      nodes.push(found.node); i = found.end;
      if (found.strength === 'prefix') { expect = true; math = true; continue; }
      if (found.strength === 'strong') math = true;
      last = found.strength; good = { i, count: nodes.length, math };
    }
  }
  return good && { end: good.i, nodes: nodes.slice(0, good.count), math: good.math };
}
function referenceAt(text, i, references, linked) {
  for (const ref of references) {
    if (linked.has(ref.target)) continue;
    ref.pattern.lastIndex = i;
    const m = ref.pattern.exec(text);
    if (m && !wordChar(text[i + m[0].length])) return { ref, length: m[0].length };
  }
  return null;
}
export function compileReferences(list) {
  // Longest phrases first so "World model" does not shadow a longer alias.
  return [...list].sort((a, b) => b.text.length - a.text.length).map(ref => ({
    ...ref,
    pattern: new RegExp(ref.text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + (ref.chapter ? '(?=[^.;]*\\bchapters?\\b)' : ''), ref.caseSensitive || ref.chapter ? 'y' : 'iy')
  }));
}
// Splits text into text, math, and ref segments. Joining every segment's source
// gives back the input exactly.
export function segment(text, references = [], linked = new Set()) {
  const out = [];
  let i = 0, plain = 0;
  const flush = to => { if (to > plain) out.push({ type: 'text', source: text.slice(plain, to) }); };
  while (i < text.length) {
    if (!wordChar(text[i - 1]) && !has(SUB, text[i - 1]) && text[i] !== ' ') {
      const run = sequence(text, i, text.length);
      if (run && run.math && run.end > i) {
        flush(i); out.push({ type: 'math', source: text.slice(i, run.end), nodes: run.nodes });
        i = plain = run.end; continue;
      }
      const ref = references.length ? referenceAt(text, i, references, linked) : null;
      if (ref) {
        flush(i); linked.add(ref.ref.target);
        out.push({ type: 'ref', source: text.slice(i, i + ref.length), target: ref.ref.target });
        i = plain = i + ref.length; continue;
      }
    }
    i++;
  }
  flush(text.length);
  return out;
}

function el(doc, tag, className, text) {
  const node = doc.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function renderNodes(doc, nodes, parent) {
  for (const node of nodes) parent.append(renderNode(doc, node));
  return parent;
}
function renderDecorations(doc, node, target) {
  for (const d of node.deco ?? []) {
    if (d.sub) target.append(renderNodes(doc, d.sub, el(doc, 'sub')));
    else if (d.star) target.append(el(doc, 'sup', 'm-star', '∗'));
    else target.append(d.text);
  }
}
function renderNode(doc, node) {
  if (node.t === 'sp') return doc.createTextNode(' ');
  if (node.t === 'num') return el(doc, 'span', 'mn', node.text);
  if (node.t === 'sym') return el(doc, 'span', 'mn', node.text);
  if (node.t === 'word') return el(doc, 'span', 'mt', node.text);
  if (node.t === 'op') {
    const text = { '-': '−', '*': '·' }[node.text] ?? node.text;
    const kind = node.text === ',' ? ' is-comma' : MULTIPLY.includes(node.text) ? ' is-mul' : '=≈→←∈:'.includes(node.text) ? ' is-rel' : '';
    return el(doc, 'span', 'mo' + kind + (node.unary ? ' is-unary' : ''), text);
  }
  if (node.t === 'var') {
    const holder = el(doc, 'span', 'mv');
    const single = [...node.base.normalize('NFC')].length === 1;
    holder.append(el(doc, single && !/[Α-ΩΣ]/.test(node.base) ? 'i' : 'span', single ? 'mi' : 'mi is-upright', node.base));
    renderDecorations(doc, node, holder); return holder;
  }
  if (node.t === 'fn') {
    const holder = el(doc, 'span', 'mfn');
    holder.append(node.italic ? el(doc, 'i', 'mi', node.name) : el(doc, 'span', LAYERS.has(node.name) ? 'mf is-layer' : 'mf', node.name));
    if (node.args) { holder.append(el(doc, 'span', 'mp', '(')); renderNodes(doc, node.args, holder); holder.append(el(doc, 'span', 'mp', ')')); }
    renderDecorations(doc, node, holder); return holder;
  }
  const group = el(doc, 'span', node.shape ? 'mshape' : 'mgroup');
  group.append(el(doc, 'span', node.shape ? 'mb' : 'mp', node.open));
  renderNodes(doc, node.nodes, group);
  group.append(el(doc, 'span', node.shape ? 'mb' : 'mp', node.close));
  renderDecorations(doc, node, group); return group;
}
export function renderMath(doc, part, source = true) {
  const span = el(doc, 'span', 'math');
  if (source) span.dataset.source = part.source;
  const operands = part.nodes.filter(n => !['op', 'sp'].includes(n.t));
  if (operands.every(n => n.shape)) span.classList.add('is-shape');
  return renderNodes(doc, part.nodes, span);
}

// Replaces qualifying text inside root with decorated spans and links.
export function decorate(root, { references = [], skip = new Set(), href = id => '#' + id } = {}) {
  const doc = root.ownerDocument, linked = new Set(skip);
  const walker = doc.createTreeWalker(root, 4, {
    acceptNode: node => node.parentElement.closest('[data-source], [data-equation], a') ? 2 : 1
  });
  const texts = []; while (walker.nextNode()) texts.push(walker.currentNode);
  for (const node of texts) {
    const parts = segment(node.data, references, linked);
    if (parts.every(part => part.type === 'text')) continue;
    const fragment = doc.createDocumentFragment();
    parts.forEach((part, index) => {
      if (part.type === 'text') { if (part.source) fragment.append(part.source); }
      else if (part.type === 'math') {
        // Opening and closing punctuation stay on the formula's line.
        const before = parts[index - 1]?.type === 'text' ? /[(“‘[]*$/.exec(parts[index - 1].source)[0] : '';
        const after = parts[index + 1]?.type === 'text' ? /^[.,;:!?)\]”’]*/.exec(parts[index + 1].source)[0] : '';
        if (!before && !after) { fragment.append(renderMath(doc, part)); return; }
        if (before) { fragment.lastChild.data = fragment.lastChild.data.slice(0, -before.length); if (!fragment.lastChild.data) fragment.lastChild.remove(); }
        if (after) parts[index + 1] = { ...parts[index + 1], source: parts[index + 1].source.slice(after.length) };
        const keep = el(doc, 'span', 'math-keep'); keep.dataset.source = before + part.source + after;
        keep.append(before, renderMath(doc, part, false), after); fragment.append(keep);
      } else {
        const link = el(doc, 'a', 'xref', part.source);
        link.href = href(part.target); link.dataset.target = part.target; link.dataset.source = part.source;
        fragment.append(link);
      }
    });
    node.replaceWith(fragment);
  }
}
function sourceLength(node) {
  if (node.nodeType === 3) return node.data.length;
  if (node.nodeType !== 1 && node.nodeType !== 11) return 0;
  if (node.dataset?.source !== undefined) return node.dataset.source.length;
  let total = 0; for (const child of node.childNodes) total += sourceLength(child); return total;
}
// Restores plain source text, keeping the caret at the same source position.
export function undecorate(root, selection) {
  const doc = root.ownerDocument;
  let caret = null;
  if (selection?.rangeCount && root.contains(selection.focusNode)) {
    const range = doc.createRange(); range.setStart(root, 0); range.setEnd(selection.focusNode, selection.focusOffset);
    caret = sourceLength(range.cloneContents());
  }
  for (const node of root.querySelectorAll('[data-source]')) node.replaceWith(doc.createTextNode(node.dataset.source));
  root.normalize();
  if (caret === null) return;
  const walker = doc.createTreeWalker(root, 4);
  let remaining = caret, lastText = null;
  while (walker.nextNode()) {
    lastText = walker.currentNode;
    if (remaining <= lastText.data.length) { selection.collapse(lastText, remaining); return; }
    remaining -= lastText.data.length;
  }
  if (lastText) selection.collapse(lastText, lastText.data.length);
}
