---
status: draft
updated: 2026-10-19
---
# Chinese reading and hover translation

The Inference, Training and Results pages support English and Simplified Chinese. English remains the default. Select 中文 in the footer or open a page with `?lang=zh`; the language and sentence/paragraph/off preference are remembered in local storage. The editor always stays in English and keeps its existing publishing target.

Article prose, interface labels and SVG text switch together. Model names, formulas, tensor shapes, run IDs, original drawings and English source content are preserved. Chinese labels use system fonts and fit within the original text width. Both the main canvas and focused crops use the same SVG definition.

Hover translation works in both directions. After a 900 ms dwell, a compact popup appears near the source line, preferably above it. The preview is limited to three lines, with explicit expansion for longer content. Moving away, selecting text, scrolling the reader, or pressing Escape dismisses it. Scrolling inside an expanded popup keeps it open. Keyboard focus opens paragraph translation; touch users can tap text to toggle it. Physical touchscreen testing remains outstanding.

## Preview and checks

Run `npm run dev --prefix writeup` and open <http://127.0.0.1:4173/>. No new dependency, build step, font download or translation API is required. Use `npm test --prefix writeup` for content coverage, language defaults, sentence pairing, SVG restoration/width constraints, popup placement and existing tests. The local editor serves the explicit localization dependencies from its existing static allowlist.

## Maintaining translations

The dictionaries in `locales/` use exact English source strings as keys. Update the matching dictionary when upstream prose, rich-text fragments, diagram labels or result nodes change. Unknown strings remain in English. A failed dictionary request reports that some translations could not load. Technical identifiers and formulas intentionally stay untranslated.

- `i18n-core.mjs`: language selection, dynamic labels and sentence pairing. A paragraph is used when sentence counts or formatting boundaries cannot be paired.
- `i18n.mjs`: bilingual prose, saved rich-text overrides, interface labels and translation controls. Original bold text, math and cross-reference decoration continue through the existing renderers.
- `diagram-i18n.mjs`: batched SVG text measurement, translation and exact English restoration without changing the diagram geometry.
- `translation-ui.mjs`: source-line anchoring and viewport-aware popup placement.

The Pages workflow copies these modules, CSS and dictionaries into the static artifact. Keep that list in sync when adding runtime files. Browser checks covered desktop layouts, 390 × 640 frames, English/Chinese switching, short/long popups, expansion, dismissal, and both training and inference diagrams. Simulation and training claims remain the original author's content.
