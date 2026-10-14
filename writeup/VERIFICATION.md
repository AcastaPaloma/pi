# Verification — 2026-10-13 revision

- Five Node tests pass: responsive camera fitting, cursor-anchored zoom, pan constraints, valid region references, and the requested decision-model route with matching VLM/flow camera frames.
- JavaScript syntax check passes.
- Chrome desktop (1440 × 900) and phone (390 × 844) reviewed through the live local preview.
- Verified decision macro contents, VLM, flow matching, L1 → L2 → L33 → L34 → transformer via arrows and Space.
- Verified direct contents jumps to patchification and attention mask, SVG focus rendering, ingestion/decision contents switching, Escape returning to the parent region, and browser Back restoration.
- Phone detail framing reserves space below the floating button: focused crop starts at 66px, button ends at 54px. No document overflow at 390 × 844.
- Confirmed header, footer, toolbar, and minimap are absent. Commentary container is empty and hidden.
- No browser warnings or errors recorded during this review. Temporary viewport overrides were reset.
- Final review screenshot: `/tmp/pi-canvas-minimal.jpg`.

Main architecture regions render from the provided vector SVG. The source SVG's overview inset contains an embedded raster and is preserved as supplied. Safari and Firefox were not separately exercised.
