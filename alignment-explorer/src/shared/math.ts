/**
 * Median of an array of numbers. Sorts in place; use a copy if original must be preserved.
 */
export function median(values: number[]): number {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Mean of an array of numbers.
 */
export function mean(values: number[]): number {
  if (values.length === 0) return NaN;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/**
 * Percentile (0–1) of a sorted array. Linear interpolation between indices.
 */
export function percentile(sortedArr: number[], p: number): number {
  if (sortedArr.length === 0) return NaN;
  if (p <= 0) return sortedArr[0];
  if (p >= 1) return sortedArr[sortedArr.length - 1];
  const index = p * (sortedArr.length - 1);
  const lo = Math.floor(index);
  const hi = Math.ceil(index);
  if (lo === hi) return sortedArr[lo];
  const w = index - lo;
  return sortedArr[lo] * (1 - w) + sortedArr[hi] * w;
}
