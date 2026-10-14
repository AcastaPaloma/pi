// A small, closed document format. No HTML, URLs, styles, or executable attributes
// are accepted from saved documents.
export const fonts = { serif: 'Georgia', sans: 'Arial', mono: 'Courier New' };
const tags = new Set(['p', 'div', 'span', 'strong', 'em', 'u', 'br', 'ul', 'ol', 'li', 'blockquote']);
export const emptyCopy = () => ({ version: 1, sections: {} });
export function validateCopy(value, allowedIds) {
  if (!value || value.version !== 1 || !value.sections || Array.isArray(value.sections) ||
      Object.keys(value).some(k => !['version', 'sections'].includes(k))) throw new Error('Invalid text document.');
  let count = 0, length = 0;
  function node(n, depth = 0) {
    if (!n || typeof n !== 'object' || Array.isArray(n) || ++count > 30000 || depth > 20) throw new Error('Text document is too complex.');
    if (Object.hasOwn(n, 'text')) {
      if (Object.keys(n).length !== 1 || typeof n.text !== 'string') throw new Error('Invalid text.');
      length += n.text.length;
      if (length > 500000) throw new Error('Text document is too large.');
      return;
    }
    if (n.equation === 'flow-loss' && Object.keys(n).length === 1) return;
    if (!tags.has(n.tag) || !Array.isArray(n.children) ||
        Object.keys(n).some(k => !['tag', 'children', 'font'].includes(k)) ||
        (n.font !== undefined && !Object.hasOwn(fonts, n.font))) throw new Error('Unsupported formatting.');
    n.children.forEach(child => node(child, depth + 1));
  }
  for (const [id, nodes] of Object.entries(value.sections)) {
    if (!allowedIds.has(id) || !Array.isArray(nodes)) throw new Error('Unknown text section.');
    nodes.forEach(n => node(n));
  }
  return value;
}
export function renderNodes(nodes, document) {
  const fragment = document.createDocumentFragment();
  function render(n) {
    if (Object.hasOwn(n, 'text')) return document.createTextNode(n.text);
    if (n.equation) {
      const holder = document.createElement('div');
      holder.dataset.equation = 'flow-loss'; holder.contentEditable = 'false';
      const ns = 'http://www.w3.org/1998/Math/MathML';
      const math = document.createElementNS(ns, 'math'); math.setAttribute('display', 'block');
      const sub = document.createElementNS(ns, 'msub');
      for (const [tag, text] of [['mi', 'ℒ'], ['mtext', 'flow']]) {
        const el = document.createElementNS(ns, tag); el.textContent = text; sub.append(el);
      }
      const period = document.createElementNS(ns, 'mo'); period.textContent = '.';
      math.append(sub, period); holder.append(math); return holder;
    }
    const el = document.createElement(n.tag);
    if (n.font) { el.dataset.font = n.font; el.style.fontFamily = fonts[n.font]; }
    el.append(...n.children.map(render)); return el;
  }
  fragment.append(...nodes.map(render)); return fragment;
}
export function serializeBody(root) {
  function walk(node) {
    if (node.nodeType === 3) return [{ text: node.textContent }];
    if (node.nodeType !== 1) return [];
    if (node.dataset.equation === 'flow-loss') return [{ equation: 'flow-loss' }];
    const sourceTag = node.tagName.toLowerCase();
    if (['script', 'style', 'iframe', 'object', 'img', 'svg'].includes(sourceTag)) return [];
    let children = [...node.childNodes].flatMap(walk);
    let tag = ({ b: 'strong', i: 'em', font: 'span' })[sourceTag] ?? sourceTag;
    if (!tags.has(tag)) return children;
    // Native editing commands use both semantic elements and styled spans.
    if (node.style.fontWeight === 'bold' || Number(node.style.fontWeight) >= 600) children = [{ tag: 'strong', children }];
    if (node.style.fontStyle === 'italic') children = [{ tag: 'em', children }];
    if (node.style.textDecorationLine.includes('underline') || node.style.textDecoration.includes('underline')) children = [{ tag: 'u', children }];
    const family = (node.getAttribute('face') || node.style.fontFamily || '').replace(/["']/g, '').toLowerCase();
    const font = Object.entries(fonts).find(([, name]) => family === name.toLowerCase())?.[0];
    const result = { tag, children };
    if (font) result.font = font;
    return [result];
  }
  return [...root.childNodes].flatMap(walk);
}
