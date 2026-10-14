# Pi canvas

A static, text-led visual write-up using the supplied SVG artwork. No framework, runtime dependency, external font request, or build step.

## Published site

Open <https://acastapaloma.github.io/pi/>. The GitHub Pages workflow verifies and publishes changes to writeup/ on main. Only runtime HTML, CSS, JavaScript modules, and SVG assets are deployed.

## Preview

Run npm run dev in this directory, then open <http://127.0.0.1:4173/>. Any static HTTP server works, including GitHub Pages under a project path.

## Reading and navigation

- Desktop uses an equal split: canvas left, continuous article right. Phones keep the canvas above the article.
- Scrolling either half advances the article and focuses the diagram associated with the passage at the reading line. Long passages retain their diagram until the next heading crosses that line.
- Drag to explore the canvas. Pinch, Ctrl/Command-wheel, or + / − zooms. The next reading scroll returns the camera to the current passage. The floating button also resumes reading or shows the whole canvas.
- Clicking a diagram component scrolls to its matching article section. Hover outlines the target without changing the reading position.
- Contents provides direct links to all 29 main sections and 81 components across ingestion, subgoal policy, and decision model.
- Left/right arrows jump between main sections. Space, Page Up/Down, and up/down arrows scroll through the article. Escape closes contents, resumes a detached camera, or returns from a detail to its parent. F fits the current passage.
- URLs such as #observation/patches restore both text and camera. Browser back/forward works. Reduced motion disables camera and article transitions.

## Author commentary

author-copy.json preserves the supplied transcript; tests prevent unintended edits. Each destination has an explicit annotations entry in content.js. Supplied paragraphs appear verbatim. Empty arrays render lorem ipsum from reading.mjs until the author supplies text.

Text is inserted as plain text. The supplied flow-loss expression is rendered with MathML. Do not draft extra explanatory prose.

The steps array defines major sections, macro ownership, focus bounds, optional camera frame, and component IDs. The details array defines smaller regions. reading.mjs turns these into one ordered article and chooses the active passage. Bounds are [x, y, width, height] on a 2048-wide logical canvas. Native coordinates convert by multiplying by 2048 / 4118.032888760241; layer translations/scales apply first. A step's frame can show context while its bounds identifies the highlighted drawing.

## Artwork

The base full-backbone.svg preserves the original ingestion artwork. Its right-hand model is clipped out and replaced with decision-model.svg. loss-vlm.svg and loss-eam.svg sit beneath their branches. subgoal-generation.svg occupies the lower-left canvas and covers the high-level semantic subgoal policy: observation, visual encoding, text modalities, combined context, next subtask, and prediction/loss.

Editor metadata was removed; vector paths, text, bundled fonts, and authored embedded images remain. Layers and focused crops share one inline SVG definition. Colors and typography follow the personal website and Genie article. See [RESEARCH.md](RESEARCH.md) for the original interaction research.

## Checks and future additions

Run npm test and node --check app.js. Browser verification is recorded in [VERIFICATION.md](VERIFICATION.md).

For each new drawing, add named component bounds, contents destinations, and empty annotation arrays. Every destination automatically gets an article section with placeholder text. Place only verbatim supplied commentary at matching sections. Preserve IDs so shared links keep working. Bump entry, CSS, and content version queries together when publishing changes.
