---
name: Pi visual notebook
description: A pale editorial article paired with the author's illustrated canvas.
colors:
  paper: "#fbfbfa"
  ink: "#1a1a1a"
  secondary: "#5a5a5a"
  accent: "#1f3a8a"
  rule: "#d8d8d4"
  lifted-paper: "#f7f2ef"
typography:
  headline:
    fontFamily: "Georgia, 'Times New Roman', serif"
    fontSize: "clamp(28px, 2.6vw, 40px)"
    fontWeight: 400
    lineHeight: 1.18
    letterSpacing: "-.02em"
  body:
    fontFamily: "Georgia, 'Times New Roman', serif"
    fontSize: "18px"
    lineHeight: 1.75
  editor-label:
    fontFamily: "Arial, sans-serif"
    fontSize: "13px"
    lineHeight: 1.4
rounded:
  control: "5px"
  region: "4px"
  tool: "3px"
spacing:
  paragraph: "24px"
  heading: "26px"
components:
  canvas-control:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "10px 14px"
  save-live:
    backgroundColor: "{colors.accent}"
    textColor: "white"
    rounded: "{rounded.region}"
    padding: "9px 14px"
---

# Design System: Pi visual notebook

## Overview

**Creative North Star: "The illustrated notebook"**

The established personal-site and Genie article language pairs calm Georgia prose with the author's hand-drawn diagrams. Pale paper, restrained blue emphasis, and generous reading space keep the illustration and explanation together.

**Key Characteristics:**
- Equal desktop space for drawing and article.
- Original artwork carries the expressive color.
- Small, quiet controls preserve the reading surface.

## Colors

The accent identifies the current passage, focus, and the editor's primary action. Paper and ink define the article; secondary text and rules support navigation. Lifted paper belongs to the focused drawing crop. Keep artwork colors intact.

## Typography

Georgia is the default article and reader-control family. Major headings use the headline token; component headings use `clamp(23px, 1.9vw, 29px)` with the same weight, line height, and tracking. Body paragraphs use the body token and paragraph spacing. On phones, major/component headings become 28px/23px and body text becomes 16px at 1.65 line height.

Editor tools use the compact Arial label treatment. Authored body formatting may explicitly choose Georgia, Arial, or Courier New; this does not alter headings or application navigation.

## Layout

The workspace fills the viewport. Desktop columns divide 50/50: canvas on the left, independently scrolling article on the right, separated by a quiet rule. The article container is at most 680px wide, including responsive horizontal padding of `clamp(24px, 4vw, 64px)`. A compact Contents menu opens over the reading column.

At 760px and below, stack canvas above article at 42%/58% of viewport height. The editor uses 32%/68% to accommodate its wrapping toolbar. Mobile article side padding is 26px. Keep the reader independently scrollable and the canvas visible.

## Elevation & Depth

Most surfaces stay flat. The focused SVG crop uses a subtle `0 4px 16px #322d2526` shadow while surrounding artwork fades to 22% opacity. The Contents popover uses `0 12px 30px #1a1a1a10`. Depth clarifies focus and overlays, without turning passages into cards.

## Shapes

Controls and overlays have small rounded corners; article sections have no enclosing card. Thin outlines identify hover, selection, and editable body focus. Visible keyboard focus uses the accent.

## Components

- **Canvas:** drag or ordinary wheel movement pans without moving the article; pinch and modified wheel zoom. Manual exploration stays detached until pointer/focus returns to the reader or reading scroll resumes. Diagram clicks select the matching passage. Hover outlines a target without advancing text.
- **Floating canvas control:** Whole canvas and Resume reading describe its current action.
- **Article and Contents:** component notes fold under each main section behind a row of chips that outline their drawn regions; current headings and the active Contents entry use the accent. Major and component passages share one continuous reading column. Preserve stable section destinations.
- **Local editor:** toolbar sits above the reader and wraps on narrow screens. Body outlines appear on hover/focus; headings stay outside editing. Tools provide B/I/U, three system fonts, indent/outdent, list, undo/redo, and Save to live. Save disables when clean or busy. Compact status text distinguishes drafts, saving, publishing, published, and recoverable errors. Public pages have no editor toolbar.
- **Motion:** camera and reading transitions support orientation; reduced-motion preferences disable transitions and smooth scrolling.

- **Inference stage:** dark studio backdrop (#2a2928) for contrast with the yellow arms, real SO-101 geometry, motion slowed and exaggerated. Camera shots look from the far side of the table so the arms face the viewer. Overlays (frustums, action-chunk dots) use the accent; pending chunks use orange. HUD panels are pale paper with Arial labels and Georgia values. Reduced motion cuts instead of animating and disables autoplay.

## Do's and Don'ts

- Do preserve the original drawing and its typography.
- Do retain pale paper, Georgia reading type, quiet rules, and clear focus states.
- Do keep canvas exploration independent of article scrolling.
- Don't reintroduce the superseded narrow contents panel or macro tabs.
- Don't add public editing chrome, a minimap, or a visible zoom toolbar.
