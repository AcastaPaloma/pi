# Pi canvas

A static, text-led visual write-up using the supplied SVG artwork, in two tabs. **Inference** (index.html, the entry page) replays the inference timeline block by block beside a 3D model of two SO-101 arms. **Training** (training.html) is the illustrated architecture and training article. No framework, external font request, or build step; three.js r169 (MIT) is vendored in vendor/ for the 3D stage.

## Published site

Open <https://acastapaloma.github.io/pi/>. The GitHub Pages workflow verifies and publishes changes to writeup/ on main. Only runtime HTML, CSS, JavaScript modules, and SVG assets are deployed.

## Preview

Run npm run dev in this directory, then open <http://127.0.0.1:4173/>. Any static HTTP server works, including GitHub Pages under a project path.

## Inference tab

- The left column stacks the 3D stage over the inference drawing (assets/inference.svg); the right column holds one passage per timeline block. ← and →, the player buttons, timeline clicks, and reading scroll all select a block; the timeline camera lifts that block and the stage replays it.
- The walkthrough autoplays and loops. Each loop moves the robot on to its next two subtasks (Open the faucet, Scrub the plate, Put the sponge back, Close the faucet). Manual navigation pauses autoplay; Play resumes. Deep links use #inf-<block>, and older links without that prefix redirect to training.html.
- inference-content.js holds the blocks: text, drawing bounds, and stage directives (beat, robot task time, caption, BAGEL progress, action chunks). robot-stage.js builds the scene in pi_sim coordinates, solves IK for every scripted waypoint at load, and choreographs camera dives, shutter captures, subgoal denoising, and action-chunk dots. Nothing runs a model.
- assets/so101/ is generated from the vendored MuJoCo Menagerie SO-101 (Apache-2.0, LICENSE included) by scripts/build_so101_asset.py: visual meshes voxel-remeshed at 0.8 mm, decimated, quantized (about 300 KB gzipped). Rerun it after changing the vendored model.

## Training tab: reading and navigation

- Desktop uses an equal split: canvas left, continuous article right. Phones keep the canvas above the article.
- Scrolling the reading panel focuses the diagram associated with the passage at the reading line. Long passages retain their diagram until the next heading crosses that line.
- Drag or scroll over the canvas to explore freely. Pinch, Ctrl/Command-wheel, or + / − zooms. Camera tracking resumes when you return to the reading panel (pointer, focus, or reading scroll). Canvas scrolling never moves the article. The floating button also resumes reading or shows the whole canvas.
- Clicking a diagram component scrolls to its matching article section. Hover outlines the target without changing the reading position.
- Six chapters: Start here (preface, map, notation), Ingestion, Decision model, Training losses, High-level policy, World model. Contents links all 34 main sections and 85 component notes.
- Component notes are folded under their main section. Their chips outline the exact drawn region on hover or focus; clicking a chip, a diagram component, or a Contents entry opens the notes and moves the camera. Hovering a drawn component highlights its chip. Contents has a switch to open every note. The map section lists reading routes.
- Left/right arrows jump between main sections. Space, Page Up/Down, and up/down arrows scroll through the article. Escape closes contents, resumes a detached camera, or returns from a detail to its parent. F fits the current passage.
- URLs such as #observation/patches restore both text and camera. Browser back/forward works. Reduced motion disables camera and article transitions.

## Text editor

Run **npm run edit** in writeup/, then open <http://127.0.0.1:4174/editor>. The local editor runs beside the same canvas. Choose a section from Contents, click its text, and edit directly. Select text to apply bold, italic, underline, or serif/sans-serif/monospace fonts. Indent/Outdent, lists, Undo, and Redo are available. Paste inserts plain text.

Click **Save to live** (or Command/Ctrl-S). It updates only writeup/text.json on the repository's main branch and triggers the existing Pages workflow. The status moves from saving to publishing to published. Publication normally takes a minute or two; it is not instantaneous. Existing site visitors receive the new copy on reload.

The editor uses the computer's existing GitHub CLI login (gh auth login if needed). It listens only on 127.0.0.1, keeps GitHub credentials outside the browser, and is not included in the public Pages site. The publisher's sole writable path is writeup/text.json. It uses the [GitHub Contents API](https://docs.github.com/en/rest/repos/contents#create-or-update-file-contents), with a file SHA check to reject stale edits.

Unsaved changes are kept in local browser storage. Reload restores a draft when its base revision still matches GitHub. If another edit has been saved, the older draft remains downloadable and the editor loads the latest text. Conflicting saves retain the draft and show a recovery message.

The toolbar uses the browser's native contenteditable editing commands to retain [native undo history](https://developer.mozilla.org/en-US/docs/Web/API/Document/execCommand). The saved document is a validated tree of text and allowed formatting, not raw HTML. Fonts come from the system; no font downloads or editor packages are required.

The editor fetches current text from GitHub, so publishing does not modify or commit unrelated local work. After an editor save, use git pull --ff-only before making later code changes from this checkout.

## Article prose

article.js holds every passage, written from the π0.7 paper and the work it cites. Blocks are paragraphs (with **bold**), lists, "On the drawing" notes for labels that differ from the paper, and placeholders (the Preface is reserved for the author). author-copy.json preserves the original transcript; tests check that it still appears verbatim and in order. Published edits in text.json override individual section bodies; headings, diagram geometry, and section IDs stay fixed. The flow-loss expression is rendered with MathML.

The steps array defines major sections, macro ownership, focus bounds, optional camera frame, and component IDs. The details array defines smaller regions. reading.mjs turns these into one ordered article and chooses the active passage. Bounds are [x, y, width, height] on a 2048-wide logical canvas. Native coordinates convert by multiplying by 2048 / 4118.032888760241; layer translations/scales apply first. A step's frame can show context while its bounds identifies the highlighted drawing.

## Artwork

world-model-placeholder.svg reserves the lower-right slot for the unfinished world-model (BAGEL) drawing, with five dashed component slots matching the article. The base full-backbone.svg preserves the original ingestion artwork. Its right-hand model is clipped out and replaced with decision-model.svg. loss-vlm.svg and loss-eam.svg sit beneath their branches. subgoal-generation.svg occupies the lower-left canvas and covers the high-level semantic subgoal policy: observation, visual encoding, text modalities, combined context, next subtask, and prediction/loss.

Editor metadata was removed; vector paths, text, bundled fonts, and authored embedded images remain. Layers and focused crops share one inline SVG definition. Colors and typography follow the personal website and Genie article. See [RESEARCH.md](RESEARCH.md) for the original interaction research.

## Checks and future additions

Run npm test and node --check app.js inference.js robot-stage.js. The tests also solve every scripted waypoint against the real SO-101 kinematics (under 2 mm). Browser verification is recorded in [VERIFICATION.md](VERIFICATION.md).

For each new drawing, add named component bounds, contents destinations, and a passage in article.js. When the world-model drawing is ready, replace the placeholder layer and update the five wm-* bounds. Preserve IDs so shared links keep working. Bump entry, CSS, and content version queries together when publishing changes.
