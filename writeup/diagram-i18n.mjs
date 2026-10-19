export function translateArtwork(root, diagramCatalog, locale) {
  const started = performance.now();
  const labels = [...root.querySelectorAll('text')].map(text => {
    const original = text.dataset.originalText ?? text.textContent;
    return { text, original, translated: diagramCatalog[original] };
  }).filter(item => item.translated && item.translated !== item.original);
  // Batch every geometry read before any write. Interleaving these per label
  // would force hundreds of full-canvas layouts on the detailed training SVG.
  const originals = labels.map(({ text }) => text.hasAttribute('data-original-text') ? null : {
    font: text.getAttribute('font-family') || '', width: text.getComputedTextLength(),
  });
  labels.forEach(({ text, original }, i) => {
    if (!originals[i]) return;
    text.dataset.originalText = original;
    text.dataset.originalFont = originals[i].font;
    text.dataset.originalWidth = String(originals[i].width);
  });
  for (const { text, original, translated } of labels) {
    text.textContent = locale === 'zh' ? translated : original;
    text.setAttribute('font-family', locale === 'zh' ? "'PingFang SC','Microsoft YaHei','Noto Sans CJK SC',sans-serif" : text.dataset.originalFont);
    text.removeAttribute('textLength'); text.removeAttribute('lengthAdjust');
  }
  if (locale !== 'zh') { root.dataset.localizationMs = String(Math.round(performance.now() - started)); return; }
  const widths = labels.map(({ text }) => text.getComputedTextLength());
  labels.forEach(({ text }, i) => {
    const budget = Number(text.dataset.originalWidth), actual = widths[i];
    if (budget > 0 && actual > budget) {
      text.setAttribute('textLength', String(budget)); text.setAttribute('lengthAdjust', 'spacingAndGlyphs');
    }
    text.dataset.localizedWidth = String(budget > 0 ? Math.min(actual, budget) : actual);
  });
  root.dataset.localizationMs = String(Math.round(performance.now() - started));
}
