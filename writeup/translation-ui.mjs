export const HOVER_DELAY = 900;
export const LEAVE_DELAY = 300;
// Keep translations in the visual pane so they never cover the sentence being read.
export function translationPosition({ anchor, width, height, viewportWidth, viewportHeight, footerHeight = 34 }) {
  const inset = 14;
  const left = anchor?.left ?? 0;
  const bottom = Math.min(anchor?.bottom ?? viewportHeight - footerHeight, viewportHeight - footerHeight);
  return {
    left: Math.max(8, Math.min(viewportWidth - width - 8, left + inset)),
    top: Math.max(8, bottom - height - inset),
  };
}
