import type { GlobalFilters, GpuMemorySnapshot, PerfMemory, PerfMemoryPerStage, RustPerfRecord } from '../types/report';

export const STAGE_KEYS = ['forward', 'post', 'dp', 'group', 'conf'] as const;
export type StageKey = (typeof STAGE_KEYS)[number];

export const STAGE_MS_KEYS: Record<StageKey, keyof RustPerfRecord> = {
  forward: 'forward_ms',
  post: 'post_ms',
  dp: 'dp_ms',
  group: 'group_ms',
  conf: 'conf_ms',
};

// --- Filtering ---

export function filterRecords(
  records: RustPerfRecord[],
  filters: GlobalFilters
): RustPerfRecord[] {
  const [dMin, dMax] = filters.duration_range;
  const [fMin, fMax] = filters.frame_count_range;
  const search = filters.search_id.trim().toLowerCase();

  return records.filter((r) => {
    if (r.duration_ms < dMin || r.duration_ms > dMax) return false;
    if (r.num_frames_t < fMin || r.num_frames_t > fMax) return false;
    if (search && !r.utterance_id.toLowerCase().includes(search)) return false;
    return true;
  });
}

// --- Stats ---

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

export function p90(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (90 / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

// --- Per-record derived ---

export function rtf(record: RustPerfRecord): number {
  return record.duration_ms > 0 ? record.total_ms / record.duration_ms : 0;
}

export function costPerFrame(record: RustPerfRecord): number {
  return record.num_frames_t > 0 ? record.total_ms / record.num_frames_t : 0;
}

export function stageRatio(record: RustPerfRecord, stage: StageKey): number {
  const total = record.total_ms;
  if (total <= 0) return 0;
  const key = STAGE_MS_KEYS[stage];
  const ms = record[key] as number;
  return ms / total;
}

export interface PeakMemory {
  /** Peak GPU used (bytes) across stages; same as peak_gpu_used. */
  peak_gpu_alloc: number;
  peak_cpu: number;
  peak_stage: string;
}

function isFlatMemory(mem: PerfMemory | PerfMemoryPerStage): mem is PerfMemory {
  return 'gpu_used' in mem && typeof (mem as PerfMemory).gpu_used === 'number';
}

function peakGpuFromPerStage(mem: PerfMemoryPerStage): { peak_gpu_alloc: number; peak_stage: string } | undefined {
  let maxUsed = 0;
  let maxStage = 'align';
  for (const stage of STAGE_KEYS) {
    const snap: GpuMemorySnapshot | undefined = mem[stage];
    if (snap && typeof snap.gpu_used === 'number' && snap.gpu_used > maxUsed) {
      maxUsed = snap.gpu_used;
      maxStage = stage;
    }
  }
  return maxUsed > 0 ? { peak_gpu_alloc: maxUsed, peak_stage: maxStage } : undefined;
}

export function peakMemory(record: RustPerfRecord): PeakMemory | undefined {
  const mem = record.memory;
  if (!mem) return undefined;
  if (isFlatMemory(mem)) {
    return {
      peak_gpu_alloc: mem.gpu_used,
      peak_cpu: 0,
      peak_stage: 'align',
    };
  }
  const fromStages = peakGpuFromPerStage(mem as PerfMemoryPerStage);
  if (!fromStages) return undefined;
  return {
    peak_gpu_alloc: fromStages.peak_gpu_alloc,
    peak_cpu: 0,
    peak_stage: fromStages.peak_stage,
  };
}

/** Bin (x, y) points by x and compute median y per bin. Returns bin centers and median y (null for empty bins). */
export function binAndMedian(
  x: number[],
  y: number[],
  nBins: number,
  xMin?: number,
  xMax?: number
): { binCenters: number[]; medianYs: (number | null)[] } {
  if (x.length === 0 || nBins < 1) return { binCenters: [], medianYs: [] };
  const lo = xMin ?? Math.min(...x);
  const hi = xMax ?? Math.max(...x);
  const span = hi - lo || 1;
  const binWidth = span / nBins;
  const bins: number[][] = [];
  for (let i = 0; i < nBins; i++) bins.push([]);
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    const yi = y[i];
    let idx = Math.floor((xi - lo) / binWidth);
    if (idx >= nBins) idx = nBins - 1;
    if (idx < 0) idx = 0;
    bins[idx].push(yi);
  }
  const binCenters: number[] = [];
  const medianYs: (number | null)[] = [];
  for (let i = 0; i < nBins; i++) {
    binCenters.push(lo + (i + 0.5) * binWidth);
    medianYs.push(bins[i].length > 0 ? median(bins[i]) : null);
  }
  return { binCenters, medianYs };
}

// --- ECDF ---

export function ecdf(values: number[]): { x: number[]; y: number[] } {
  if (values.length === 0) return { x: [], y: [] };
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const y = sorted.map((_, i) => (i + 1) / n);
  return { x: sorted, y };
}

// --- Per-run aggregates (over filtered records) ---

export interface RunAggregates {
  medianTotalMs: number;
  p90TotalMs: number;
  medianRtf: number;
  medianCostPerFrame: number;
  medianStageMs: Record<StageKey, number>;
  medianStageRatios: Record<StageKey, number>;
  /** Median peak GPU alloc (bytes) over records with positive peak GPU; 0 if none. Zeros are excluded so runs with mixed 0/non-zero (e.g. Rust) show a meaningful median. */
  medianPeakGpuAlloc: number;
  /** Whether any record has memory data. */
  hasMemory: boolean;
}

export function computeRunAggregates(records: RustPerfRecord[]): RunAggregates {
  const totalMs = records.map((r) => r.total_ms);
  const rtfValues = records.map(rtf);
  const costValues = records.map(costPerFrame);

  const medianStageMs: Record<StageKey, number> = {
    forward: median(records.map((r) => r.forward_ms)),
    post: median(records.map((r) => r.post_ms)),
    dp: median(records.map((r) => r.dp_ms)),
    group: median(records.map((r) => r.group_ms)),
    conf: median(records.map((r) => r.conf_ms)),
  };

  const sumMedians =
    medianStageMs.forward +
    medianStageMs.post +
    medianStageMs.dp +
    medianStageMs.group +
    medianStageMs.conf;
  const medianStageRatios: Record<StageKey, number> = sumMedians
    ? {
        forward: medianStageMs.forward / sumMedians,
        post: medianStageMs.post / sumMedians,
        dp: medianStageMs.dp / sumMedians,
        group: medianStageMs.group / sumMedians,
        conf: medianStageMs.conf / sumMedians,
      }
    : { forward: 0, post: 0, dp: 0, group: 0, conf: 0 };

  const peakGpuValues = records.map((r) => peakMemory(r)).filter((p): p is PeakMemory => p != null).map((p) => p.peak_gpu_alloc);
  const hasMemory = peakGpuValues.length > 0;
  // Use median of positive values only so runs with mixed 0 and non-zero (e.g. Rust dual-context)
  // show a meaningful "typical" peak GPU instead of 0 when most records report 0.
  const positivePeakGpu = peakGpuValues.filter((v) => v > 0);
  const medianPeakGpuAlloc =
    hasMemory && positivePeakGpu.length > 0 ? median(positivePeakGpu) : 0;

  return {
    medianTotalMs: median(totalMs),
    p90TotalMs: p90(totalMs),
    medianRtf: median(rtfValues),
    medianCostPerFrame: median(costValues),
    medianStageMs,
    medianStageRatios,
    medianPeakGpuAlloc,
    hasMemory,
  };
}

/** Speedup vs baseline: baselineMedianTotalMs / runMedianTotalMs. */
export function speedupVsBaseline(
  runMedianTotalMs: number,
  baselineMedianTotalMs: number
): number {
  if (runMedianTotalMs <= 0) return 0;
  return baselineMedianTotalMs / runMedianTotalMs;
}
