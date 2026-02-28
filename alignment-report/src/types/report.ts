export type Aggregate = "median" | "mean" | "min" | "max" | "p50" | "p90" | "p95" | "p99";
export type DType = "f16" | "bf16" | "f32" | "f64";
export type Device = string;

export interface StageMemory {
  cpu: number;          // bytes (RSS peak during stage)
  gpu_alloc: number;    // bytes (allocated peak)
  gpu_reserved: number; // bytes (reserved peak)
}

export interface MemoryBreakdown {
  forward?: StageMemory;
  post?: StageMemory;
  dp?: StageMemory;
  group?: StageMemory;
  conf?: StageMemory;
  align?: StageMemory;
}

export interface RustPerfRecord {
  // identity
  utterance_id: string;
  audio_path: string;

  // input shape
  duration_ms: number;
  num_frames_t: number;
  state_len: number;
  ts_product: number;
  vocab_size: number;

  // runtime
  dtype: DType;
  device: Device;
  frame_stride_ms: number;

  // config
  warmup: number;
  repeats: number;
  aggregate: Aggregate;

  // timings (aggregated)
  forward_ms: number;
  post_ms: number;
  dp_ms: number;
  group_ms: number;
  conf_ms: number;
  align_ms: number;
  total_ms: number;

  align_ms_per_ts: number;
  align_ms_per_t: number;

  // raw repeats
  forward_ms_repeats: number[];
  post_ms_repeats: number[];
  dp_ms_repeats: number[];
  group_ms_repeats: number[];
  conf_ms_repeats: number[];
  align_ms_repeats: number[];
  total_ms_repeats: number[];

  // memory footprint
  memory?: MemoryBreakdown;
}

/** Run file format as written by Rust or loaded from JSON. */
export interface PerfRunFile {
  schema_version?: number;
  generated_at?: string;
  config?: Record<string, unknown>;
  records: RustPerfRecord[];
}

/** A run loaded in the app (user can rename, toggle visibility). */
export interface LoadedRun {
  id: string;
  name: string;
  records: RustPerfRecord[];
  loadedAt: Date;
}

/** Global filters applied to all runs for charts and KPIs. */
export interface GlobalFilters {
  duration_range: [number, number];
  frame_count_range: [number, number];
  search_id: string;
}