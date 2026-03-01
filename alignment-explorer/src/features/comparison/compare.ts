import type {
  ParsedTextGrid,
  WordInterval,
  WordDelta,
  SampleComparison,
} from '../../shared/types';
import { normalizeWordText } from '../../shared/utils';

function norm(w: WordInterval): string {
  return w.normalizedText ?? normalizeWordText(w.text);
}

/**
 * Compare baseline vs variant word-by-word by index. Only if normalized word sequence matches.
 */
export function compareBaselineVariant(
  entryId: string,
  implementationId: string,
  baseline: ParsedTextGrid,
  variant: ParsedTextGrid
): SampleComparison {
  if (baseline.error || baseline.words.length === 0) {
    return {
      entryId,
      implementationId,
      match: false,
      mismatchReason: baseline.error ?? 'No baseline words',
    };
  }
  if (variant.error) {
    return {
      entryId,
      implementationId,
      match: false,
      mismatchReason: variant.error,
    };
  }
  if (baseline.words.length !== variant.words.length) {
    return {
      entryId,
      implementationId,
      match: false,
      wordCount: baseline.words.length,
      mismatchReason: `Word count mismatch: baseline=${baseline.words.length}, variant=${variant.words.length}`,
    };
  }

  for (let i = 0; i < baseline.words.length; i++) {
    if (norm(baseline.words[i]) !== norm(variant.words[i])) {
      return {
        entryId,
        implementationId,
        match: false,
        wordCount: baseline.words.length,
        mismatchReason: `Word mismatch at index ${i}: "${baseline.words[i].text}" vs "${variant.words[i].text}"`,
      };
    }
  }

  const wordDeltas: WordDelta[] = baseline.words.map((b, i) => {
    const v = variant.words[i];
    const deltaStartMs = v.startMs - b.startMs;
    const deltaEndMs = v.endMs - b.endMs;
    const deltaMidMs = v.midMs - b.midMs;
    return {
      wordIndex: i,
      baseline: b,
      variantId: implementationId,
      deltaStartMs,
      deltaEndMs,
      deltaMidMs,
      absStartMs: Math.abs(deltaStartMs),
      absEndMs: Math.abs(deltaEndMs),
      absMidMs: Math.abs(deltaMidMs),
    };
  });

  return {
    entryId,
    implementationId,
    match: true,
    wordCount: baseline.words.length,
    wordDeltas,
  };
}
