import type { LoadedRun, PerfRunFile, RustPerfRecord } from '../types/report';

export function isValidRecord(r: unknown): r is RustPerfRecord {
  if (!r || typeof r !== 'object') return false;
  const o = r as Record<string, unknown>;
  return (
    typeof o.utterance_id === 'string' &&
    typeof o.duration_ms === 'number' &&
    typeof o.num_frames_t === 'number' &&
    typeof o.total_ms === 'number'
  );
}

/** Parse JSONL: one JSON object per line (Rust streaming format). */
function parseJsonlRecords(text: string): RustPerfRecord[] {
  const records: RustPerfRecord[] = [];
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  for (const line of lines) {
    try {
      const obj: unknown = JSON.parse(line);
      if (isValidRecord(obj)) records.push(obj as RustPerfRecord);
    } catch {
      // skip invalid lines
    }
  }
  return records;
}

function parseRunFile(json: unknown, filename: string): LoadedRun | null {
  let records: RustPerfRecord[];
  if (Array.isArray(json)) {
    records = json.filter(isValidRecord) as RustPerfRecord[];
  } else if (json && typeof json === 'object' && 'records' in json) {
    const raw = (json as PerfRunFile).records;
    records = Array.isArray(raw) ? raw.filter(isValidRecord) as RustPerfRecord[] : [];
  } else {
    return null;
  }
  if (records.length === 0) return null;

  return {
    id: crypto.randomUUID(),
    name: filename.replace(/\.json$/i, ''),
    records,
    loadedAt: new Date(),
  };
}

/**
 * Parse perf run from raw text (single JSON or JSONL).
 * Returns a LoadedRun or null if parsing fails or no valid records.
 */
export function loadRunFromText(text: string, filename: string): LoadedRun | null {
  try {
    const json: unknown = JSON.parse(text);
    const run = parseRunFile(json, filename);
    if (run) return run;
  } catch {
    // Not single JSON; try JSONL (one record per line, e.g. Rust streaming output)
  }
  const records = parseJsonlRecords(text);
  if (records.length === 0) return null;
  return {
    id: crypto.randomUUID(),
    name: filename.replace(/\.json$/i, ''),
    records,
    loadedAt: new Date(),
  };
}
