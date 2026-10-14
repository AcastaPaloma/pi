# Pi canvas

A static canvas write-up using the supplied SVG artwork. No framework, runtime dependency, external font request, or build step.

## Published site

Open <https://acastapaloma.github.io/pi/>. The GitHub Pages workflow in `.github/workflows/pages.yml` verifies and publishes the canvas when `writeup/` changes on `main`. It can also be run manually from GitHub Actions. Only the HTML, styles, JavaScript modules, and SVG assets are included in the deployed site.

## Preview

Run `npm run dev` in this directory, then open <http://127.0.0.1:4173/>. Any static HTTP server works, including GitHub Pages under a project path.

## Navigation

- The single canvas button starts the walkthrough or returns to the whole drawing.
- Left/right arrows and Space traverse 22 main stops. Shift-Space goes back. Details never add extra stops.
- The contents panel switches between ingestion and the decision model. Each main part and detail is a direct camera destination.
- Hover or keyboard-focus a diagram region to highlight its contents entry. Click to focus it. Escape returns from a detail to its main stop, then to the whole canvas.
- Drag or two-finger scroll to pan. Pinch or `+` / `−` zooms; double-click zooms at the pointer (Shift reverses it). `F` fits the selection.
- URLs such as `#observation/patches` restore a selected region. Browser back/forward works.
- The panel sits below the canvas on phones. Reduced motion disables camera transitions.

The decision route is: whole decision model → VLM backbone / X context → flow matching → L1 → L2 → L33 → L34 → full transformer layer.

## Author commentary

The panel uses only the author’s supplied words. `author-copy.json` preserves the transcribed source; tests prevent unintended edits. Each step and component has an explicit `annotations` entry in `content.js`. Unwritten entries stay empty and show no placeholder copy:

```js
export const annotations = {
  // 'patches': ['Your first paragraph.', 'Your second paragraph.'],
};
```

Text is inserted as plain text. The commentary region remains hidden for empty entries.

`steps` defines the walkthrough, macro ownership, focus bounds, optional camera frame, and detail IDs. `details` defines smaller selectable regions. Bounds are `[x, y, width, height]` in a 2048 × 1591.439 logical canvas. SVG-native coordinates convert by multiplying by `2048 / 4118.032888760241`. A step's `frame` can show surrounding context while its `bounds` identifies the highlighted drawing.

## Artwork

`assets/full-backbone.svg` comes from the user's supplied `full_backbone.svg`. Embedded editor metadata was removed; vector paths, text, bundled fonts, and artwork were preserved. The original right-hand model is clipped out and replaced by `decision-model.svg` from the supplied `updated_cross_attention`. `loss-vlm.svg` and `loss-eam.svg` sit beneath their respective branches. Each source retains its native coordinates in `artwork.layers`; its translation and scale also map the selectable regions. All diagrams and focused crops share an inline SVG definition. Embedded images already present in the supplied SVGs remain as authored.

Colors and typography follow the local Genie article and the personal website. See [RESEARCH.md](RESEARCH.md) for the original interaction research.

## Checks

Run `npm test` and `node --check app.js`. Browser verification is recorded in [VERIFICATION.md](VERIFICATION.md). GitHub Pages serves the published version; local preview remains available for editing.

## Adding author material

For every new supplied drawing, identify its individual components, add their named bounds and direct contents links, and create empty annotation arrays. Place only verbatim supplied commentary at matching steps; do not draft extra explanation. Preserve existing IDs so shared links keep working. Render supplied math without changing its terms. Bump the entry/CSS/content version query together when changing deployed assets or copy so returning readers receive the update.
