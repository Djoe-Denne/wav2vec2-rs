import { useReports } from '../context/ReportContext';

const MAX_SAFE = Number.MAX_SAFE_INTEGER;

export function Filters() {
  const { runs, filters, updateFilters, resetFilters } = useReports();

  if (runs.length === 0) return null;

  const [dMin, dMax] = filters.duration_range;
  const [fMin, fMax] = filters.frame_count_range;

  return (
    <div className="bg-white p-4 rounded shadow mb-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">Filters</h3>
        <button onClick={resetFilters} className="text-sm text-blue-600 hover:underline">
          Reset
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Duration min (ms)</label>
          <input
            type="number"
            min={0}
            value={dMin === 0 ? '' : dMin}
            onChange={(e) => {
              const v = e.target.value === '' ? 0 : parseInt(e.target.value, 10);
              if (!Number.isNaN(v)) updateFilters({ duration_range: [v, dMax] });
            }}
            placeholder="0"
            className="w-full border rounded px-2 py-1"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Duration max (ms)</label>
          <input
            type="number"
            min={0}
            value={dMax >= MAX_SAFE ? '' : dMax}
            onChange={(e) => {
              const v = e.target.value === '' ? MAX_SAFE : parseInt(e.target.value, 10);
              if (!Number.isNaN(v)) updateFilters({ duration_range: [dMin, v] });
            }}
            placeholder="No limit"
            className="w-full border rounded px-2 py-1"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Frame count min</label>
          <input
            type="number"
            min={0}
            value={fMin === 0 ? '' : fMin}
            onChange={(e) => {
              const v = e.target.value === '' ? 0 : parseInt(e.target.value, 10);
              if (!Number.isNaN(v)) updateFilters({ frame_count_range: [v, fMax] });
            }}
            placeholder="0"
            className="w-full border rounded px-2 py-1"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Frame count max</label>
          <input
            type="number"
            min={0}
            value={fMax >= MAX_SAFE ? '' : fMax}
            onChange={(e) => {
              const v = e.target.value === '' ? MAX_SAFE : parseInt(e.target.value, 10);
              if (!Number.isNaN(v)) updateFilters({ frame_count_range: [fMin, v] });
            }}
            placeholder="No limit"
            className="w-full border rounded px-2 py-1"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Utterance ID search</label>
          <input
            type="text"
            value={filters.search_id}
            onChange={(e) => updateFilters({ search_id: e.target.value })}
            placeholder="Filter by ID..."
            className="w-full border rounded px-2 py-1"
          />
        </div>
      </div>
    </div>
  );
}
