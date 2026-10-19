export const HOVER_DELAY = 900;
export const LEAVE_DELAY = 300;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
// A sentence can wrap across several lines. Anchor to the line under the pointer,
// rather than to the large bounding box around the entire paragraph.
export function translationLine(rects, point, fallback) {
  const lines = rects.filter(rect => rect.width > 1 && rect.height > 0 && rect.height <= 48);
  if (!lines.length) return fallback;
  if (!point) return lines[0];
  const distance = rect => Math.max(rect.top - point.y, point.y - rect.bottom, 0) * 1000 + Math.max(rect.left - point.x, point.x - rect.right, 0);
  return lines.reduce((best, rect) => distance(rect) < distance(best) ? rect : best);
}
export function translationPosition({ anchor, point, bounds, width, height, viewportWidth, viewportHeight, footerHeight = 34, preferredPlacement }) {
  const margin = 10, gap = 9;
  const leftEdge = Math.max(margin, (bounds?.left ?? 0) + margin);
  const rightEdge = Math.min(viewportWidth - margin, (bounds?.right ?? viewportWidth) - margin);
  const bottomEdge = viewportHeight - footerHeight - margin;
  const above = Math.max(0, anchor.top - gap - margin);
  const below = Math.max(0, bottomEdge - anchor.bottom - gap);
  const placement = preferredPlacement || (above >= height || above >= below ? 'above' : 'below');
  const maxHeight = placement === 'above' ? above : below;
  const usedHeight = Math.min(height, maxHeight);
  const sourceX = point?.x ?? (anchor.left + anchor.right) / 2;
  const left = clamp(sourceX - Math.min(72, width / 4), leftEdge, Math.max(leftEdge, rightEdge - width));
  return {
    left, top: placement === 'above' ? anchor.top - gap - usedHeight : anchor.bottom + gap,
    placement, maxHeight,
    arrowLeft: clamp(sourceX - left, 16, width - 16),
  };
}
