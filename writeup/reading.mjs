export const placeholder = [
  'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer vitae justo eget magna fermentum iaculis. Amet consectetur adipiscing elit pellentesque habitant morbi tristique senectus. Sed viverra tellus in hac habitasse platea dictumst.',
  'Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.'
];

export function buildReadingEntries(steps, details, annotations) {
  const byId = new Map(details.map(item => [item.id, item]));
  return steps.flatMap((step, stepIndex) => [step, ...step.details.map(id => byId.get(id))].map(item => ({
    ...item,
    stepIndex,
    macro: step.macro,
    detail: item.id === step.id ? null : item.id,
    paragraphs: annotations[item.id]?.length ? annotations[item.id] : placeholder.slice(0, item.id === step.id ? 2 : 1),
    isPlaceholder: !annotations[item.id]?.length
  })));
}

// The most recent heading above the reading line owns the camera. Unlike
// visibility ratios, this stays deterministic for both short and long sections.
export function readingIndex(tops, scrollTop, anchor) {
  const line = scrollTop + anchor;
  let low = 0, high = tops.length - 1, found = 0;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (tops[middle] <= line + 1) { found = middle; low = middle + 1; }
    else high = middle - 1;
  }
  return found;
}
