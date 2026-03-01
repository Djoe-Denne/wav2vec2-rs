import type {
  AudioEntry,
  CorpusState,
  ParsedTextGrid,
  SampleComparison,
} from '../../shared/types';
import {
  aggregateImplementationStats,
  aggregateStatsBySubset,
} from '../comparison/aggregateStats';
import type {
  WorkerChunkResult,
  WorkerError,
  WorkerInputEntry,
  WorkerProgress,
} from '../../workers/corpusWorker';

const CHUNK_SIZE = 80;

function readFileAsText(file: File): Promise<string> {
  return file.text();
}

/**
 * Read baseline and variant TextGrid contents for one entry.
 */
async function readEntryContents(entry: AudioEntry): Promise<WorkerInputEntry> {
  let baselineContent: string | null = null;
  if (entry.baselineFile) {
    try {
      baselineContent = await readFileAsText(entry.baselineFile);
    } catch {
      entry.errors.push(`Failed to read baseline: ${entry.baselineFile.name}`);
    }
  }

  const variants: { suffix: string; content: string }[] = [];
  for (const v of entry.variants) {
    try {
      const content = await readFileAsText(v.file);
      variants.push({ suffix: v.suffix, content });
    } catch {
      entry.errors.push(`Failed to read variant ${v.suffix}: ${v.file.name}`);
    }
  }

  return {
    id: entry.id,
    subset: entry.subset,
    baselineContent,
    variants,
  };
}

export interface RunWorkerOptions {
  entries: AudioEntry[];
  onProgress?: (current: number, total: number, message: string) => void;
}

/**
 * Process entries in chunks: read one chunk's files, send to worker, merge results. Repeat.
 * Aggregation runs on main thread after all chunks. Keeps memory and message size bounded.
 */
export async function runCorpusWorker(options: RunWorkerOptions): Promise<CorpusState> {
  const { entries, onProgress } = options;
  const totalChunks = Math.ceil(entries.length / CHUNK_SIZE) || 1;

  const allComparisons: SampleComparison[] = [];
  const allBaselines: Record<string, ParsedTextGrid | null> = {};
  const allVariants: Record<string, Record<string, ParsedTextGrid>> = {};

  const worker = new Worker(new URL('../../workers/corpusWorker.ts', import.meta.url), {
    type: 'module',
  });

  const runChunk = (chunkIndex: number): Promise<void> => {
    const start = chunkIndex * CHUNK_SIZE;
    const chunk = entries.slice(start, start + CHUNK_SIZE);
    return new Promise((resolve, reject) => {
      const onMessage = (e: MessageEvent<WorkerProgress | WorkerChunkResult | WorkerError>) => {
        const msg = e.data;
        if (msg.type === 'progress') {
          onProgress?.(
            chunkIndex * CHUNK_SIZE + Math.floor((msg.current / Math.max(1, msg.total)) * chunk.length),
            entries.length,
            `Chunk ${chunkIndex + 1}/${totalChunks}: ${msg.message}`
          );
        } else if (msg.type === 'chunkDone') {
          worker.removeEventListener('message', onMessage);
          for (const [id, value] of Object.entries(msg.parsed.baselines)) {
            allBaselines[id] = value;
          }
          for (const [id, map] of Object.entries(msg.parsed.variants)) {
            if (!allVariants[id]) allVariants[id] = {};
            Object.assign(allVariants[id], map);
          }
          allComparisons.push(...msg.comparisons);
          resolve();
        } else if (msg.type === 'error') {
          worker.removeEventListener('message', onMessage);
          reject(new Error(msg.error));
        }
      };
      worker.addEventListener('message', onMessage);
      readChunkAndPost();
      async function readChunkAndPost() {
        const workerInput: WorkerInputEntry[] = [];
        for (const entry of chunk) {
          workerInput.push(await readEntryContents(entry));
        }
        worker.postMessage({ type: 'runChunk', entries: workerInput });
      }
    });
  };

  try {
    for (let i = 0; i < totalChunks; i++) {
      onProgress?.(i * CHUNK_SIZE, entries.length, `Processing chunk ${i + 1}/${totalChunks}…`);
      await runChunk(i);
    }

    worker.terminate();

    const entrySubset = new Map(entries.map((e) => [e.id, e.subset]));
    const stats = aggregateImplementationStats(allComparisons);
    const { bySubset: statsBySubset } = aggregateStatsBySubset(allComparisons, (id) =>
      entrySubset.get(id) ?? 'unknown'
    );

    const updatedEntries: AudioEntry[] = entries.map((entry) => ({
      ...entry,
      baseline: allBaselines[entry.id] ?? null,
    }));

    return {
      entries: updatedEntries,
      stats,
      statsBySubset,
      comparisons: allComparisons,
      parsedVariants: allVariants,
    };
  } catch (e) {
    worker.terminate();
    throw e;
  }
}
