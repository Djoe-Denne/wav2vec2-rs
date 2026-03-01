import { parseTextGrid } from '../features/textgrid/parseTextGrid';
import { compareBaselineVariant } from '../features/comparison/compare';
import {
  aggregateImplementationStats,
  aggregateStatsBySubset,
} from '../features/comparison/aggregateStats';
import type {
  ParsedTextGrid,
  SampleComparison,
  ImplementationStats,
  Subset,
} from '../shared/types';

export interface WorkerInputEntry {
  id: string;
  subset: Subset;
  baselineContent: string | null;
  variants: { suffix: string; content: string }[];
}

export interface WorkerProgress {
  type: 'progress';
  current: number;
  total: number;
  message: string;
}

export interface WorkerResult {
  type: 'done';
  comparisons: SampleComparison[];
  stats: ImplementationStats[];
  statsBySubset: ImplementationStats[];
  parsed: {
    baselines: Record<string, ParsedTextGrid | null>;
    variants: Record<string, Record<string, ParsedTextGrid>>;
  };
}

/** Result of processing one chunk (batched mode) */
export interface WorkerChunkResult {
  type: 'chunkDone';
  parsed: {
    baselines: Record<string, ParsedTextGrid | null>;
    variants: Record<string, Record<string, ParsedTextGrid>>;
  };
  comparisons: SampleComparison[];
}

export interface WorkerError {
  type: 'error';
  error: string;
}

export type WorkerMessage = WorkerProgress | WorkerResult | WorkerChunkResult | WorkerError;

function processEntries(entries: WorkerInputEntry[]): {
  comparisons: SampleComparison[];
  parsed: WorkerResult['parsed'];
} {
  const comparisons: SampleComparison[] = [];
  const baselines: Record<string, ParsedTextGrid | null> = {};
  const variants: Record<string, Record<string, ParsedTextGrid>> = {};

  const total = Math.max(1, entries.reduce((s, e) => s + e.variants.length, 0));
  let done = 0;

  for (const entry of entries) {
    let baseline: ParsedTextGrid | null = null;
    if (entry.baselineContent) {
      baseline = parseTextGrid(entry.baselineContent);
      baselines[entry.id] = baseline;
    } else {
      baselines[entry.id] = null;
    }

    variants[entry.id] = {};
    for (const v of entry.variants) {
      const parsed = parseTextGrid(v.content);
      variants[entry.id][v.suffix] = parsed;

      if (baseline && !baseline.error && baseline.words.length > 0) {
        const comparison = compareBaselineVariant(
          entry.id,
          v.suffix,
          baseline,
          parsed
        );
        comparisons.push(comparison);
      }

      done++;
      self.postMessage({
        type: 'progress',
        current: done,
        total,
        message: `Parsed ${done}/${total}`,
      } satisfies WorkerProgress);
    }
  }

  return { comparisons, parsed: { baselines, variants } };
}

self.onmessage = (
  e: MessageEvent<
    | { type: 'run'; entries: WorkerInputEntry[] }
    | { type: 'runChunk'; entries: WorkerInputEntry[] }
  >
) => {
  const { type, entries } = e.data;
  if (type !== 'run' && type !== 'runChunk') return;

  try {
    const { comparisons, parsed } = processEntries(entries);

    if (type === 'runChunk') {
      self.postMessage({
        type: 'chunkDone',
        parsed,
        comparisons,
      } satisfies WorkerChunkResult);
      return;
    }

    const stats = aggregateImplementationStats(comparisons);
    const entrySubset = new Map(entries.map((x) => [x.id, x.subset]));
    const { bySubset: statsBySubset } = aggregateStatsBySubset(comparisons, (id) =>
      entrySubset.get(id) ?? 'unknown'
    );

    self.postMessage({
      type: 'done',
      comparisons,
      stats,
      statsBySubset,
      parsed,
    } satisfies WorkerResult);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    self.postMessage({ type: 'error', error: message } satisfies WorkerError);
  }
};
