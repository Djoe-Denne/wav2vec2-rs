import React from 'react';
import { useReports } from '../context/ReportContext';

export function Filters() {
  const { filters, updateFilters, resetFilters, selectedReport } = useReports();

  if (!selectedReport) return null;

  const maxDuration = Math.max(
    ...selectedReport.data.sentences.map((s) => s.duration_ms)
  );

  return (
    <div className="bg-white p-4 rounded shadow mb-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">Filters</h3>
        <button
          onClick={resetFilters}
          className="text-sm text-blue-600 hover:underline"
        >
          Reset
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Split</label>
          <select
            value={filters.split}
            onChange={(e) => updateFilters({ split: e.target.value as any })}
            className="w-full border rounded px-2 py-1"
          >
            <option value="all">All</option>
            <option value="clean">Clean</option>
            <option value="other">Other</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Has Reference</label>
          <select
            value={filters.has_reference}
            onChange={(e) => updateFilters({ has_reference: e.target.value as any })}
            className="w-full border rounded px-2 py-1"
          >
            <option value="all">All</option>
            <option value="true">True</option>
            <option value="false">False</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">
            Min Confidence: {filters.confidence_threshold.toFixed(2)}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={filters.confidence_threshold}
            onChange={(e) =>
              updateFilters({ confidence_threshold: parseFloat(e.target.value) })
            }
            className="w-full"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Search ID</label>
          <input
            type="text"
            value={filters.search_id}
            onChange={(e) => updateFilters({ search_id: e.target.value })}
            placeholder="Enter sentence ID..."
            className="w-full border rounded px-2 py-1"
          />
        </div>
      </div>
    </div>
  );
}
