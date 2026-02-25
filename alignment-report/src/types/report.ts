export interface Report {
  schema_version: number;
  meta: Meta;
  sentences: Sentence[];
  aggregates?: Aggregates;
}

export interface Meta {
  generated_at: string;
  model_path: string;
  device: string;
  frame_stride_ms: number;
  case_count: number;
}

export interface Sentence {
  id: string;
  split: string;
  has_reference: boolean;
  duration_ms: number;
  word_count_pred: number;
  word_count_ref: number;
  structural: StructuralMetrics;
  confidence: ConfidenceMetrics;
  timing: TimingMetrics;
  per_word?: PerWord[];
  notes: string[];
}

export interface StructuralMetrics {
  negative_duration_word_count: number;
  overlap_word_count: number;
  non_monotonic_word_count: number;
  invalid_confidence_word_count: number;
  gap_ratio: number;
  overlap_ratio: number;
}

export interface ConfidenceMetrics {
  word_conf_mean: number;
  word_conf_min: number;
  low_conf_threshold_used?: number;
  avg_word_margin?: number | null;
  avg_boundary_confidence?: number | null;
  low_conf_word_ratio: number;
  blank_frame_ratio: number | null;
  token_entropy_mean: number | null;
}

export interface TimingMetrics {
  start: TimingStats;
  end: TimingStats;
  abs_err_ms_median: number;
  abs_err_ms_p90: number;
  trimmed_mean_abs_err_ms: number;
  offset_ms: number;
  drift_ms_per_sec: number;
}

export interface TimingStats {
  mean_signed_ms: number;
  median_abs_ms: number;
  p90_abs_ms: number;
  max_abs_ms: number;
}

export interface PerWord {
  word: string;
  ref_start_ms: number;
  ref_end_ms: number;
  pred_start_ms: number;
  pred_end_ms: number;
  start_err_ms: number;
  end_err_ms: number;
  conf: number | null;
  quality_confidence?: number | null;
  calibrated_confidence?: number | null;
  mean_logp?: number | null;
  geo_mean_prob?: number | null;
  min_logp?: number | null;
  p10_logp?: number | null;
  mean_margin?: number | null;
  coverage_frame_count?: number;
  boundary_confidence?: number | null;
}

export interface Aggregates {
  counts: {
    total: number;
    with_reference: number;
    without_reference: number;
  };
  global: AggregateMetrics;
  by_split: Record<string, AggregateMetrics>;
  outliers?: OutlierMetrics;
}

export interface AggregateMetrics {
  abs_err_ms_median: StatsDistribution | null;
  abs_err_ms_p90: StatsDistribution | null;
  drift_ms_per_sec: StatsDistribution | null;
  low_conf_word_ratio: StatsDistribution | null;
  avg_word_margin?: StatsDistribution | null;
  avg_boundary_confidence?: StatsDistribution | null;
  blank_frame_ratio: StatsDistribution | null;
}

export interface StatsDistribution {
  mean: number;
  p50: number;
  p90: number;
  p95: number;
  p99: number;
}

export interface OutlierMetrics {
  worst_abs_err_ms_p90?: OutlierEntry[];
  worst_drift_ms_per_sec?: OutlierEntry[];
  worst_low_conf_word_ratio?: OutlierEntry[];
}

export interface OutlierEntry {
  id: string;
  split: string;
  value: number;
}

export interface FilterOptions {
  split: 'all' | 'clean' | 'other';
  has_reference: 'all' | 'true' | 'false';
  duration_range: [number, number];
  confidence_threshold: number;
  search_id: string;
}

export interface LoadedReport {
  id: string;
  filename: string;
  data: Report;
  loadedAt: Date;
}
