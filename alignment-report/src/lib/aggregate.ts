/** Generic stats helpers (no report types). Use perfMetrics for run/record metrics. */

export function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

export function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

export function createHistogramBins(
  values: number[],
  binCount: number = 20
): { bins: number[]; counts: number[] } {
  if (values.length === 0) return { bins: [], counts: [] };

  const min = Math.min(...values);
  const max = Math.max(...values);
  const binWidth = (max - min) / binCount;

  const bins: number[] = [];
  const counts: number[] = new Array(binCount).fill(0);

  for (let i = 0; i <= binCount; i++) {
    bins.push(min + i * binWidth);
  }

  values.forEach((v) => {
    const binIndex = Math.min(Math.floor((v - min) / binWidth), binCount - 1);
    counts[binIndex]++;
  });

  return { bins, counts };
}
