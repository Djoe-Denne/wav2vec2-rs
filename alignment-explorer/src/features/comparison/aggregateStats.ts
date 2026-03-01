import type { SampleComparison, ImplementationStats, Subset } from '../../shared/types';
import { median, mean, percentile } from '../../shared/math';

function statsFromComparisons(comparisons: SampleComparison[], implementationId: string, subset?: Subset): ImplementationStats {
  const matched = comparisons.filter((c) => c.match && c.wordDeltas && c.wordDeltas.length > 0);
  const allAbsStart: number[] = [];
  const allAbsEnd: number[] = [];
  const allAbsMid: number[] = [];
  for (const c of matched) {
    for (const d of c.wordDeltas!) {
      allAbsStart.push(d.absStartMs);
      allAbsEnd.push(d.absEndMs);
      allAbsMid.push(d.absMidMs);
    }
  }
  const sortedMid = [...allAbsMid].sort((a, b) => a - b);
  return {
    implementationId,
    subset,
    comparedAudios: comparisons.length,
    mismatches: comparisons.filter((c) => !c.match).length,
    matchedWords: allAbsMid.length,
    medianAbsStartMs: allAbsStart.length ? median(allAbsStart) : 0,
    medianAbsEndMs: allAbsEnd.length ? median(allAbsEnd) : 0,
    medianAbsMidMs: sortedMid.length ? median(sortedMid) : 0,
    meanAbsMidMs: sortedMid.length ? mean(allAbsMid) : 0,
    p90AbsMidMs: sortedMid.length ? percentile(sortedMid, 0.9) : 0,
    p95AbsMidMs: sortedMid.length ? percentile(sortedMid, 0.95) : 0,
    maxAbsMidMs: sortedMid.length ? Math.max(...sortedMid) : 0,
  };
}

/**
 * Aggregate comparisons into per-implementation stats (global).
 */
export function aggregateImplementationStats(comparisons: SampleComparison[]): ImplementationStats[] {
  const byImpl = new Map<string, SampleComparison[]>();
  for (const c of comparisons) {
    if (!byImpl.has(c.implementationId)) byImpl.set(c.implementationId, []);
    byImpl.get(c.implementationId)!.push(c);
  }
  const stats: ImplementationStats[] = [];
  for (const [implementationId, list] of byImpl) {
    stats.push(statsFromComparisons(list, implementationId));
  }
  return stats;
}

/**
 * Aggregate stats per implementation and per subset.
 */
export function aggregateStatsBySubset(
  comparisons: SampleComparison[],
  getSubsetForEntry: (entryId: string) => Subset
): { global: ImplementationStats[]; bySubset: ImplementationStats[] } {
  const global = aggregateImplementationStats(comparisons);

  const bySubset: ImplementationStats[] = [];
  const subsets: Subset[] = ['test-clean', 'test-other', 'unknown'];
  for (const sub of subsets) {
    const filtered = comparisons.filter((c) => getSubsetForEntry(c.entryId) === sub);
    const byImpl = new Map<string, SampleComparison[]>();
    for (const c of filtered) {
      if (!byImpl.has(c.implementationId)) byImpl.set(c.implementationId, []);
      byImpl.get(c.implementationId)!.push(c);
    }
    for (const [implementationId, list] of byImpl) {
      bySubset.push(statsFromComparisons(list, implementationId, sub));
    }
  }

  return { global, bySubset };
}
