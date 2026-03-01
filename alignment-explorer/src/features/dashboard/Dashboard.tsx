import { useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useCorpus } from '../../context/CorpusContext';
import type { ImplementationStats, Subset } from '../../shared/types';

type SortKey = keyof ImplementationStats;
const SORT_KEYS: { key: SortKey; label: string; title: string }[] = [
  { key: 'implementationId', label: 'Implementation', title: 'Implementation name (from filename suffix)' },
  { key: 'comparedAudios', label: 'Compared', title: 'Number of audio samples compared' },
  { key: 'mismatches', label: 'Mismatches', title: 'Samples with word count or word sequence mismatch' },
  { key: 'matchedWords', label: 'Matched words', title: 'Total word pairs compared' },
  { key: 'medianAbsMidMs', label: 'Median |Δmid| ms', title: 'Median absolute midpoint delta (ms)' },
  { key: 'p95AbsMidMs', label: 'P95 |Δmid| ms', title: '95th percentile absolute midpoint delta (ms)' },
  { key: 'maxAbsMidMs', label: 'Max |Δmid| ms', title: 'Max absolute midpoint delta (ms)' },
];

function formatNum(n: number): string {
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(2);
}

export function Dashboard() {
  const { corpus } = useCorpus();
  const navigate = useNavigate();
  const [subsetFilter, setSubsetFilter] = useState<Subset | 'all'>('all');
  const [sortKey, setSortKey] = useState<SortKey>('implementationId');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  const statsList = useMemo(() => {
    if (!corpus) return [];
    if (subsetFilter === 'all') return corpus.stats;
    return corpus.statsBySubset.filter((s) => s.subset === subsetFilter);
  }, [corpus, subsetFilter]);

  const sortedStats = useMemo(() => {
    const list = [...statsList];
    list.sort((a, b) => {
      const va = a[sortKey];
      const vb = b[sortKey];
      if (typeof va === 'string' && typeof vb === 'string') {
        return sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      const na = Number(va);
      const nb = Number(vb);
      return sortDir === 'asc' ? na - nb : nb - na;
    });
    return list;
  }, [statsList, sortKey, sortDir]);

  const totalMismatches = useMemo(
    () => corpus?.comparisons.filter((c) => !c.match).length ?? 0,
    [corpus]
  );
  const totalMatchedWords = useMemo(
    () =>
      corpus?.comparisons
        .filter((c) => c.match && c.wordDeltas)
        .reduce((s, c) => s + (c.wordDeltas?.length ?? 0), 0) ?? 0,
    [corpus]
  );
  const implementationIds = useMemo(
    () => [...new Set(corpus?.stats.map((s) => s.implementationId) ?? [])],
    [corpus]
  );

  if (!corpus) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <p className="text-gray-600 mb-4">No corpus loaded.</p>
          <Link to="/" className="text-blue-600 hover:underline">
            Import a folder
          </Link>
        </div>
      </div>
    );
  }

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else setSortKey(key);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between">
        <Link to="/" className="text-gray-600 hover:text-gray-900">
          ← Import
        </Link>
        <h1 className="text-lg font-semibold text-gray-900">Dashboard</h1>
        <span />
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <p className="text-sm text-gray-500">Audio files</p>
            <p className="text-2xl font-semibold text-gray-900">{corpus.entries.length}</p>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <p className="text-sm text-gray-500">Implementations</p>
            <p className="text-2xl font-semibold text-gray-900">{implementationIds.length}</p>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <p className="text-sm text-gray-500">Mismatches</p>
            <p className="text-2xl font-semibold text-gray-900">{totalMismatches}</p>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <p className="text-sm text-gray-500">Matched words</p>
            <p className="text-2xl font-semibold text-gray-900">{totalMatchedWords}</p>
          </div>
        </div>

        <div className="mb-4 flex items-center gap-4">
          <label className="text-sm text-gray-600">Subset</label>
          <select
            value={subsetFilter}
            onChange={(e) => setSubsetFilter(e.target.value as Subset | 'all')}
            className="rounded border border-gray-300 px-2 py-1 text-sm"
          >
            <option value="all">All</option>
            <option value="test-clean">test-clean</option>
            <option value="test-other">test-other</option>
            <option value="unknown">unknown</option>
          </select>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden mb-8">
          {sortedStats.length === 0 ? (
            <p className="px-4 py-6 text-sm text-gray-500">No implementation stats for this subset.</p>
          ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 bg-gray-50">
                {SORT_KEYS.map(({ key, label, title }) => (
                  <th
                    key={key}
                    className="text-left px-3 py-2 font-medium text-gray-700 cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSort(key)}
                    title={title}
                  >
                    {label} {sortKey === key && (sortDir === 'asc' ? '↑' : '↓')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedStats.map((s) => (
                <tr key={s.implementationId + (s.subset ?? '')} className="border-b border-gray-100">
                  <td className="px-3 py-2">{s.implementationId}</td>
                  <td className="px-3 py-2">{s.comparedAudios}</td>
                  <td className="px-3 py-2">{s.mismatches}</td>
                  <td className="px-3 py-2">{s.matchedWords}</td>
                  <td className="px-3 py-2">{formatNum(s.medianAbsMidMs)}</td>
                  <td className="px-3 py-2">{formatNum(s.p95AbsMidMs)}</td>
                  <td className="px-3 py-2">{formatNum(s.maxAbsMidMs)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          )}
        </div>

        <h2 className="text-lg font-semibold text-gray-900 mb-3">Audio samples</h2>
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <ul className="divide-y divide-gray-100 max-h-96 overflow-y-auto">
            {corpus.entries.map((entry) => {
              const matchCount = corpus.comparisons.filter(
                (c) => c.entryId === entry.id && c.match
              ).length;
              const variantCount = entry.variants.length;
              return (
                <li key={entry.id}>
                  <button
                    type="button"
                    onClick={() => navigate(`/audio/${encodeURIComponent(entry.id)}`)}
                    className="w-full text-left px-4 py-2 hover:bg-gray-50 flex items-center justify-between"
                  >
                    <span className="font-mono text-sm truncate flex-1" title={entry.id}>
                      {entry.id}
                    </span>
                    <span className="text-xs text-gray-500 shrink-0 ml-2">
                      {entry.subset} · {matchCount}/{variantCount} match
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      </main>
    </div>
  );
}
