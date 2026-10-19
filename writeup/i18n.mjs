import { plain, translator, sentencePairs } from './i18n-core.mjs';
const catalog = {};
let failed = false;
await Promise.all(['ui', 'article-a', 'article-b', 'inference', 'results'].map(async name => {
  try {
    const response = await fetch(`./locales/${name}.zh.json`);
    if (!response.ok) throw new Error(`${name}: ${response.status}`);
    Object.assign(catalog, await response.json());
  } catch (error) { failed = true; console.warn('Translation catalog unavailable', error); }
}));
let stored;
try { stored = localStorage.getItem('pi-language'); } catch {}
const isEditor = location.pathname.includes('editor');
const requested = new URLSearchParams(location.search).get('lang') || stored;
export const language = isEditor ? 'en' : requested === 'en' ? 'en' : 'zh';
export const t = translator(catalog, language);
const reverse = new Map(Object.entries(catalog).map(([en, zh]) => [zh, en]));
const normalized = new Map(Object.entries(catalog).map(([en, zh]) => [plain(en).trim(), { en: plain(en), zh: plain(zh) }]));
export function localizeCopy(value) {
  if (Array.isArray(value)) return value.map(localizeCopy);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, key === 'text' ? t(item) : localizeCopy(item)]));
}
export function bilingual(element, original) {
  if (isEditor) return element;
  const en = reverse.get(original) ?? original, zh = catalog[en];
  if (!zh || zh === en) return element;
  element.dataset.en = plain(en); element.dataset.zh = plain(zh);
  element.classList.add('bilingual'); element.tabIndex = 0;
  return element;
}
export function inlineBilingual(element, original, make) {
  if (isEditor) {
    original.split(/\*\*(.+?)\*\*/).forEach((part, i) => { if (part) element.append(i % 2 ? make('strong', part) : document.createTextNode(part)); });
    return element;
  }
  const en = reverse.get(original) ?? original, zh = catalog[en];
  for (const pair of zh ? sentencePairs(en, zh) : [{ en, zh: en }]) {
    const span = document.createElement('span'); span.className = 'translation-sentence';
    if (zh) { span.dataset.en = plain(pair.en); span.dataset.zh = plain(pair.zh); }
    (language === 'zh' ? pair.zh : pair.en).split(/\*\*(.+?)\*\*/).forEach((part, i) => {
      if (part) span.append(i % 2 ? make('strong', part) : document.createTextNode(part));
    }); element.append(span);
  }
  return bilingual(element, en);
}
export function localizeRich(root) {
  if (isEditor) return;
  const blocks = [...root.querySelectorAll('p,li')].map(el => [el, el.innerText || el.textContent]);
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const leaves = []; while (walker.nextNode()) leaves.push(walker.currentNode);
  for (const node of leaves) {
    if (!catalog[node.textContent]) continue;
    const span = document.createElement('span');
    inlineBilingual(span, node.textContent, (tag, text) => { const el = document.createElement(tag); el.textContent = text; return el; });
    span.removeAttribute('tabindex'); span.classList.remove('bilingual'); node.replaceWith(span);
  }
  for (const [el, en] of blocks) {
    if (en === el.textContent && language === 'zh') continue;
    el.dataset.en = en; el.dataset.zh = language === 'zh' ? (el.innerText || el.textContent) : '';
    if (language !== 'zh') {
      const holder = el.cloneNode(true); for (const node of holder.querySelectorAll('[data-zh]')) node.replaceChildren(document.createTextNode(node.dataset.zh));
      el.dataset.zh = holder.textContent;
    }
    if (el.dataset.en !== el.dataset.zh) { el.classList.add('bilingual'); el.tabIndex = 0; }
  }
}
if (!isEditor) {
const style = document.createElement('link'); style.rel = 'stylesheet'; style.href = './i18n.css'; document.head.append(style);
document.title = t(document.title);
document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
const tools = document.createElement('div'); tools.className = 'translation-tools';
const lang = document.createElement('button'); lang.type = 'button'; lang.textContent = language === 'zh' ? 'EN' : '中文';
lang.setAttribute('aria-label', language === 'zh' ? '切换为英文' : 'Switch to Chinese');
lang.addEventListener('click', () => {
  const next = language === 'zh' ? 'en' : 'zh'; try { localStorage.setItem('pi-language', next); } catch {}
  const url = new URL(location.href); url.searchParams.set('lang', next); location.href = url;
});
const mode = document.createElement('select'); mode.setAttribute('aria-label', 'Translation granularity / 对照范围');
mode.innerHTML = language === 'zh' ? '<option value="sentence">逐句对照</option><option value="paragraph">整段对照</option>' : '<option value="sentence">Sentence</option><option value="paragraph">Paragraph</option>';
try { mode.value = localStorage.getItem('pi-translation-mode') || 'sentence'; } catch {}
mode.addEventListener('change', () => { hide(); try { localStorage.setItem('pi-translation-mode', mode.value); } catch {} });
tools.append(mode, lang); (document.querySelector('.site-footer') || document.body).append(tools);
const tip = document.createElement('div'); tip.id = 'translation-tooltip'; tip.className = 'translation-tooltip'; tip.role = 'tooltip'; tip.hidden = true;
const label = document.createElement('div'); label.className = 'translation-label'; const copy = document.createElement('div');
const close = document.createElement('button'); close.type = 'button'; close.textContent = '×'; close.className = 'translation-close'; close.setAttribute('aria-label', language === 'zh' ? '关闭对照' : 'Close translation');
tip.append(label, copy, close); document.body.append(tip);
let active = null, timer = 0, keyboardFocus = false;
function hide() { clearTimeout(timer); if (active) active.removeAttribute('aria-describedby'); active = null; tip.hidden = true; }
function targetFor(target) { return target.closest(mode.value === 'sentence' ? '[data-en][data-zh]' : '.bilingual'); }
function show(target) {
  if (!target) return; clearTimeout(timer); if (active) active.removeAttribute('aria-describedby'); active = target;
  label.textContent = language === 'zh' ? '英文原文' : '中文译文';
  copy.textContent = target.dataset[language === 'zh' ? 'en' : 'zh']; target.setAttribute('aria-describedby', tip.id); tip.hidden = false;
  const rect = target.getBoundingClientRect(), width = tip.offsetWidth, height = tip.offsetHeight;
  tip.style.left = `${Math.max(8, Math.min(innerWidth - width - 8, rect.left))}px`;
  const below = rect.bottom + 8;
  tip.style.top = `${Math.max(8, below + height < innerHeight - 8 ? below : Math.min(innerHeight - height - 8, rect.top - height - 8))}px`;
}
close.addEventListener('click', hide);
document.addEventListener('pointerover', event => {
  if (event.pointerType === 'touch') return;
  if (tip.contains(event.target)) { clearTimeout(timer); return; }
  const target = targetFor(event.target); if (target === active) return;
  clearTimeout(timer); if (target) timer = setTimeout(() => show(target), 300); else if (active) timer = setTimeout(hide, 180);
});
document.addEventListener('focusin', event => { const target = targetFor(event.target); if (keyboardFocus && target) show(target); });
document.addEventListener('pointerup', event => { if (event.pointerType === 'touch' && !event.target.closest('a,button,input,select') && !getSelection()?.toString()) { const target = targetFor(event.target); if (target) target === active ? hide() : show(target); } });
document.addEventListener('keydown', event => { keyboardFocus = true; if (event.key === 'Escape' && !tip.hidden) { hide(); event.stopPropagation(); } }, true);
document.addEventListener('pointerdown', event => { keyboardFocus = false; if (!tip.contains(event.target) && !targetFor(event.target)) hide(); });
document.addEventListener('scroll', hide, true); window.addEventListener('resize', hide);
function localizeDOM(root) {
  if (root.nodeType === Node.TEXT_NODE) {
    const parent = root.parentElement;
    if (!parent || parent.closest('svg,math,script,style,.bilingual,.translation-tools,.translation-tooltip,[contenteditable="true"]')) return;
    const text = root.textContent, trimmed = text.trim();
    const translated = t(trimmed); if (translated !== trimmed) root.textContent = text.replace(trimmed, translated); return;
  }
  if (root.nodeType !== Node.ELEMENT_NODE || root.closest('svg,math,script,style,.translation-tools,.translation-tooltip,[contenteditable="true"]')) return;
  if (root.matches('p,li') && !root.classList.contains('bilingual') && !root.querySelector('p,li,button,input,svg,math')) {
    const pair = normalized.get(root.textContent.trim()); if (pair) {
      root.replaceChildren(); inlineBilingual(root, pair.en, (tag, text) => { const el = document.createElement(tag); el.textContent = text; return el; });
    }
  }
  for (const attr of ['aria-label', 'title', 'placeholder']) if (root.hasAttribute(attr)) root.setAttribute(attr, t(root.getAttribute(attr)));
  for (const child of [...root.childNodes]) localizeDOM(child);
}
localizeDOM(document.body);
const observer = new MutationObserver(records => {
  observer.disconnect();
  for (const record of records) {
    if (record.type === 'characterData') localizeDOM(record.target);
    else for (const node of record.addedNodes) localizeDOM(node);
  }
  observer.observe(document.body, { childList: true, subtree: true, characterData: true });
});
observer.observe(document.body, { childList: true, subtree: true, characterData: true });
if (failed) { const warning = document.createElement('div'); warning.className = 'translation-error'; warning.textContent = '部分翻译未能加载，请刷新重试。 / Some translations could not load. Please reload.'; document.body.append(warning); }

}
