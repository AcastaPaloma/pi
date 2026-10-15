# Pi visual notebook

<!-- impeccable:product-schema 1 -->

## Platform
web

## Stack
Static HTML, CSS, and JavaScript in `writeup/`, published by GitHub Pages at https://acastapaloma.github.io/pi/. No framework, runtime dependency, font download, or build step. A separate local Node editor publishes text through the existing GitHub CLI login.

## Product Purpose
Explain π0.7 in two tabs. Inference is the entry point: the inference timeline drawing, walked block by block (arrows, autoplay, loop), beside a cinematic 3D replay of two SO-101 arms with an overhead and two wrist cameras. Training is the illustrated architecture and training article, which need not be read linearly. A Start-here chapter (preface, map with reading routes, notation) orients first-time readers; component notes fold under each main section. Readers should associate each passage with the exact part of the drawing while retaining the freedom to explore the whole model.

## Capabilities and Constraints
Desktop uses an equal canvas/article split; phones stack the canvas above the article. Reading scroll focuses the passage's diagram. Diagram clicks and the compact Contents menu jump to matching passages, including every named component. Deep links restore both text and camera.

Drag or wheel over the canvas to pan; pinch, Ctrl/Command-wheel, and +/− zoom. Canvas wheel movement never advances the article. Free exploration stays detached until the pointer or keyboard focus returns to the reader, reading scroll resumes, or the reader uses Resume reading. The floating control also shows the whole canvas. Left/right arrows move between major sections; Space, Page Up/Down, and up/down arrows scroll the article. Support touch, keyboard access, and reduced motion.

The local editor runs with `npm --prefix writeup run edit` at http://127.0.0.1:4174/editor. It edits section bodies beside the same canvas; headings, IDs, and drawing geometry remain fixed. Its toolbar offers bold, italic, underline, serif/sans-serif/monospace fonts, indent/outdent, bulleted lists, undo/redo, and Save to live. Paste inserts plain text. Editor controls appear only on the local editor.

Save to live (Command/Ctrl-S) writes only `writeup/text.json` on GitHub's `main` branch and triggers Pages publication. Status distinguishes saving, saved/publishing, and published. Visitors see published copy after reloading. Local browser drafts survive reloads; stale or conflicting drafts remain downloadable. GitHub authentication stays on the local server, which listens only on loopback.

Article prose lives in `writeup/article.js`, written from the π0.7 paper (arXiv 2604.15483) and the work it cites; details not stated in the π0.7 paper are attributed. The author's original passages stay verbatim and in order. Where a drawn label disagrees with the paper, a short "On the drawing" note says so. The Preface is reserved for the author. Every new drawing needs individually selectable components, stable destinations, and written passages. Never invent trained-policy or reproduction claims.

## Brand Commitments
Preserve the personal-site identity and Genie write-up styling: pale paper, Georgia article typography, quiet rules, and the supplied hand-drawn artwork. References: `../genie3/docs/article.css` and `../../portfolios/portfolio-prod/DESIGN.md`. Public canvas chrome stays minimal; no header, footer, minimap, or zoom toolbar.

## Evidence on Hand
The supplied SVG artwork, image exports, and Excalidraw sources originate under `/Users/kuanw/Me/draw/pi0.7/`. `writeup/author-copy.json` preserves the original supplied transcript; `text.json` contains deliberate published body overrides. The project and explanations remain unfinished.
