# Canvas walkthrough research

Reviewed October 13, 2026. These are design references; no source code or artwork from these projects was copied.

## Useful precedents

**[Brendan Bycroft's LLM Visualization](https://bbycroft.net/llm), [implementation](https://github.com/bbycroft/llm-viz)** pairs a spatial model with a guided explanation. Its separation between navigation, commentary, and inspection is the closest precedent. Pi borrows that information hierarchy, using the user's existing 2D artwork instead of building a new 3D visualization.

**[Transformer Explainer](https://poloclub.github.io/transformer-explainer/), [implementation](https://github.com/poloclub/transformer-explainer)** ties explanations to operations in a visible model pipeline. Pi likewise maps a detail to an exact region. It uses short notes and optional tensor shapes to make inspection useful without filling the whole canvas with labels.

**[impress.js](https://impress.js.org/), [implementation](https://github.com/impress/impress.js/)** demonstrates predefined stops on a spatial canvas and keyboard-driven navigation. Pi adopts explicit section bounds and an ordered route, while avoiding rotations and long presentation transitions. The reader should follow the diagram, not watch the camera perform.

**[Reddit: infinite canvas navigation critique](https://www.reddit.com/r/UI_Design/comments/1usjjrg/infinite_scrolling_for_assets_canvas_looking_for/)** raises the practical problem of becoming lost in a freely movable asset field. This is qualitative community feedback rather than usability evidence. It informed the persistent chapter name, small map, fit control, and return-to-section action.

**[Reddit: spatial bookmarks in canvas tools](https://www.reddit.com/r/PKMS/comments/192b5tm/)** discusses returning to saved screen positions. Pi's route stores named bounds, and its URL identifies the current section and pinned detail.

## Decisions for this write-up

1. **Two levels of reading.** Nine major stops form the route. Patchification, embeddings, attention, residual paths, and tensor spaces remain optional inspections. Adding a note never makes the guided route longer.
2. **One stable reading location.** The panel stays beside the diagram on desktop, below it on mobile. Inspecting a detail replaces the long section prose while preserving the section title and progress. This avoids stacking two full explanations at once.
3. **Preview and commitment are distinct.** A 170 ms hover dwell avoids incidental flashes while moving over the canvas. Click pins the explanation. Pointer movement cannot overwrite a pinned note. Tab and Enter provide the same capability; sidebar buttons support touch.
4. **Focus retains context.** The original selected image region is duplicated as a precisely aligned crop, lifted by four logical pixels with a soft offset shadow. The rest remains visible at lower opacity. No reflow or substitute illustration changes the reader's spatial map.
5. **Camera motion is brief and interruptible.** A 320 ms ease-out connects nearby regions; the same controls work with motion disabled. Wheel/drag input interrupts movement. Zoom stays anchored under the pointer, and panning is bounded enough to recover the drawing.
6. **Progress survives exploration.** Free pan and zoom preserve the route position. “Return to section” restores its framing. Detail inspection never advances the tour. Escape unpins before exiting.
7. **The original style wins.** Pale Genie colors and personal-site typography frame the user's colored drawings. No generic dashboard chrome, hero metrics, decorative gradient, or unrelated illustration.

## Scope and evidence

The live Genie URL and profile root returned GitHub Pages 404s during research. The existing local Genie and portfolio source files supplied the visual authority. User-provided PNGs superseded an exploratory export from the open Excalidraw tab. The page uses the complete supplied `full_backbone` drawing, preserving its spatial relationships.
