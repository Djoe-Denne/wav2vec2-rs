export type Aggregate = "median" | "mean" | "min" | "max" | "p50" | "p90" | "p95" | "p99";
export type DType = "f16" | "bf16" | "f32" | "f64";
export type Device = string;

/** GPU memory snapshot at a point in time (used and total in bytes). Used by Rust per-stage memory. */
export interface GpuMemorySnapshot {
  gpu_used: number;
  gpu_total: number;
}

/** Process-level GPU memory (bytes). One snapshot per alignment run. Python / flat format. */
export interface PerfMemory {
  /** Device-level used VRAM (cudaMemGetInfo: total - free). Comparable Rust/Python. */
  gpu_used: number;
  /** Device total VRAM (cudaMemGetInfo). */
  gpu_total: number;
  /** Optional: PyTorch allocated tensors only (torch.cuda.memory_allocated). Python only. */
  gpu_allocated?: number;
}

/** Per-stage GPU memory (Rust format). Each stage may have a snapshot. */
export interface PerfMemoryPerStage {
  forward?: GpuMemorySnapshot;
  post?: GpuMemorySnapshot;
  dp?: GpuMemorySnapshot;
  group?: GpuMemorySnapshot;
  conf?: GpuMemorySnapshot;
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

  // memory footprint: flat (Python) or per-stage (Rust)
  memory?: PerfMemory | PerfMemoryPerStage;
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