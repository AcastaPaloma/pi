export const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
export function fitBounds(bounds, width, height, padding = 30, maxScale = 2.5) {
  const [x, y, w, h] = bounds;
  const scale = Math.max(.04, Math.min((width - padding * 2) / w, (height - padding * 2) / h, maxScale));
  return { x: width / 2 - (x + w / 2) * scale, y: height / 2 - (y + h / 2) * scale, scale };
}
export function zoomAt(camera, point, nextScale) {
  const ratio = nextScale / camera.scale;
  return { x: point.x - (point.x - camera.x) * ratio, y: point.y - (point.y - camera.y) * ratio, scale: nextScale };
}
export function constrainCamera(camera, width, height, worldWidth, worldHeight) {
  // Keep at least a useful piece of the drawing reachable, even after a long drag.
  const margin = Math.min(80, width / 4, height / 4);
  return { ...camera, x: clamp(camera.x, margin - worldWidth * camera.scale, width - margin), y: clamp(camera.y, margin - worldHeight * camera.scale, height - margin) };
}
