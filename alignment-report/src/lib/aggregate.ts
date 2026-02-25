import { Sentence, FilterOptions } from '../types/report';

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

export function filterSentences(
  sentences: Sentence[],
  filters: FilterOptions
): Sentence[] {
  return sentences.filter((s) => {
    if (filters.split !== 'all' && s.split !== filters.split) return false;
    if (filters.has_reference === 'true' && !s.has_reference) return false;
    if (filters.has_reference === 'false' && s.has_reference) return false;
    if (s.duration_ms < filters.duration_range[0] || s.duration_ms > filters.duration_range[1])
      return false;
    if (s.confidence.word_conf_mean < filters.confidence_threshold) return false;
    if (filters.search_id && !s.id.toLowerCase().includes(filters.search_id.toLowerCase()))
      return false;
    return true;
  });
}

export interface AggregateResults {
  count: number;
  bySpli: Record<string, number>;
  timing: {
    abs_err_ms_median: { mean: number; median: number; p90: number };
    abs_err_ms_p90: { mean: number; median: number; p90: number };
    offset_ms: { mean: number; median: number; p90: number };
    drift_ms_per_sec: { mean: number; median: number; p90: number };
  };
  confidence: {
    word_conf_mean: { mean: number; median: number; p90: number };
    low_conf_word_ratio: { mean: number; median: number; p90: number };
  };
  structural: {
    gap_ratio: { mean: number; median: number; p90: number };
    overlap_ratio: { mean: number; median: number; p90: number };
  };
}

export function computeAggregates(sentences: Sentence[]): AggregateResults {
  const count = sentences.length;
  const bySpli: Record<string, number> = {};

  sentences.forEach((s) => {
    bySpli[s.split] = (bySpli[s.split] || 0) + 1;
  });

  const absErrMedian = sentences.map((s) => s.timing.abs_err_ms_median);
  const absErrP90 = sentences.map((s) => s.timing.abs_err_ms_p90);
  const offsetMs = sentences.map((s) => s.timing.offset_ms);
  const driftMs = sentences.map((s) => s.timing.drift_ms_per_sec);
  const confMean = sentences.map((s) => s.confidence.word_conf_mean);
  const lowConfRatio = sentences.map((s) => s.confidence.low_conf_word_ratio);
  const gapRatio = sentences.map((s) => s.structural.gap_ratio);
  const overlapRatio = sentences.map((s) => s.structural.overlap_ratio);

  return {
    count,
    bySpli,
    timing: {
      abs_err_ms_median: {
        mean: mean(absErrMedian),
        median: median(absErrMedian),
        p90: percentile(absErrMedian, 90),
      },
      abs_err_ms_p90: {
        mean: mean(absErrP90),
        median: median(absErrP90),
        p90: percentile(absErrP90, 90),
      },
      offset_ms: {
        mean: mean(offsetMs),
        median: median(offsetMs),
        p90: percentile(offsetMs, 90),
      },
      drift_ms_per_sec: {
        mean: mean(driftMs),
        median: median(driftMs),
        p90: percentile(driftMs, 90),
      },
    },
    confidence: {
      word_conf_mean: {
        mean: mean(confMean),
        median: median(confMean),
        p90: percentile(confMean, 90),
      },
      low_conf_word_ratio: {
        mean: mean(lowConfRatio),
        median: median(lowConfRatio),
        p90: percentile(lowConfRatio, 90),
      },
    },
    structural: {
      gap_ratio: {
        mean: mean(gapRatio),
        median: median(gapRatio),
        p90: percentile(gapRatio, 90),
      },
      overlap_ratio: {
        mean: mean(overlapRatio),
        median: median(overlapRatio),
        p90: percentile(overlapRatio, 90),
      },
    },
  };
}

export function createHistogramBins(values: number[], binCount: number = 20): { bins: number[]; counts: number[] } {
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
