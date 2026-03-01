/** File reference from folder picker */
export interface RawFileRef {
  file: File;
  relativePath: string;
}

/** Subset inferred from path */
export type Subset = 'test-clean' | 'test-other' | 'unknown';

/** Single word interval with times in ms */
export interface WordInterval {
  text: string;
  normalizedText?: string;
  startMs: number;
  endMs: number;
  midMs: number;
}

/** Parsed TextGrid: word tier only */
export interface ParsedTextGrid {
  words: WordInterval[];
  durationSec?: number;
  error?: string;
}

/** Variant TextGrid file reference; suffix = implementation id */
export interface VariantRef {
  file: File;
  suffix: string;
}

/** One audio sample: flac + baseline + variants */
export interface AudioEntry {
  id: string;
  audioFile: File;
  baseline: ParsedTextGrid | null;
  baselineFile: File | null;
  variants: VariantRef[];
  subset: Subset;
  errors: string[];
}

/** Per-word delta: variant minus baseline */
export interface WordDelta {
  wordIndex: number;
  baseline: WordInterval;
  variantId: string;
  deltaStartMs: number;
  deltaEndMs: number;
  deltaMidMs: number;
  absStartMs: number;
  absEndMs: number;
  absMidMs: number;
}

/** Comparison result for one entry × one implementation */
export interface SampleComparison {
  entryId: string;
  implementationId: string;
  match: boolean;
  wordCount?: number;
  wordDeltas?: WordDelta[];
  mismatchReason?: string;
}

/** Aggregated stats per implementation (global or per subset) */
export interface ImplementationStats {
  implementationId: string;
  subset?: Subset;
  comparedAudios: number;
  mismatches: number;
  matchedWords: number;
  medianAbsStartMs: number;
  medianAbsEndMs: number;
  medianAbsMidMs: number;
  meanAbsMidMs: number;
  p90AbsMidMs: number;
  p95AbsMidMs: number;
  maxAbsMidMs: number;
}

/** Parsed variant TextGrids by entry id then implementation suffix */
export type ParsedVariantsMap = Record<string, Record<string, ParsedTextGrid>>;

/** Corpus state after import + parse + compare */
export interface CorpusState {
  entries: AudioEntry[];
  stats: ImplementationStats[];
  statsBySubset: ImplementationStats[];
  comparisons: SampleComparison[];
  parsedVariants: ParsedVariantsMap;
}
