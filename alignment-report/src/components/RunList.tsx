import { useState } from 'react';
import { Eye, EyeOff, Star, Trash2 } from 'lucide-react';
import { useReports } from '../context/ReportContext';
import { filterRecords, computeRunAggregates } from '../lib/perfMetrics';

export function RunList() {
  const { runs, baselineId, visibility, filters, setBaseline, setVisibility, updateRun, removeRun } = useReports();

  if (runs.length === 0) return null;

  return (
    <div className="bg-white rounded shadow p-4 mb-4">
      <h3 className="font-semibold mb-3">Runs</h3>
      <ul className="space-y-2">
        {runs.map((run) => {
          const filtered = filterRecords(run.records, filters);
          const agg = computeRunAggregates(filtered);
          const visible = visibility[run.id] !== false;
          const isBaseline = run.id === baselineId;

          return (
            <RunRow
              key={run.id}
              run={run}
              visible={visible}
              isBaseline={isBaseline}
              recordCount={filtered.length}
              medianTotalMs={agg.medianTotalMs}
              onToggleVisibility={() => setVisibility(run.id, !visible)}
              onSetBaseline={() => setBaseline(isBaseline ? null : run.id)}
              onRename={(name) => updateRun(run.id, { name })}
              onRemove={() => removeRun(run.id)}
            />
          );
        })}
      </ul>
      <p className="text-xs text-gray-500 mt-2">
        Baseline is used for speedup comparison. Toggle visibility to show/hide runs in charts.
      </p>
    </div>
  );
}

function RunRow({
  run,
  visible,
  isBaseline,
  recordCount,
  medianTotalMs,
  onToggleVisibility,
  onSetBaseline,
  onRename,
  onRemove,
}: {
  run: { id: string; name: string };
  visible: boolean;
  isBaseline: boolean;
  recordCount: number;
  medianTotalMs: number;
  onToggleVisibility: () => void;
  onSetBaseline: () => void;
  onRename: (name: string) => void;
  onRemove: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState(run.name);

  const submitName = () => {
    const trimmed = editValue.trim();
    if (trimmed) onRename(trimmed);
    setEditing(false);
  };

  return (
    <li className="flex items-center gap-2 py-2 border-b border-gray-100 last:border-0">
      <button
        type="button"
        onClick={onToggleVisibility}
        className="p-1 rounded hover:bg-gray-100"
        title={visible ? 'Hide from charts' : 'Show in charts'}
        aria-label={visible ? 'Hide' : 'Show'}
      >
        {visible ? <Eye className="w-4 h-4 text-gray-600" /> : <EyeOff className="w-4 h-4 text-gray-400" />}
      </button>
      <button
        type="button"
        onClick={onSetBaseline}
        className={`p-1 rounded hover:bg-gray-100 ${isBaseline ? 'text-amber-500' : 'text-gray-400'}`}
        title={isBaseline ? 'Unset baseline' : 'Set as baseline'}
        aria-label="Set as baseline"
      >
        <Star className={`w-4 h-4 ${isBaseline ? 'fill-current' : ''}`} />
      </button>
      <div className="flex-1 min-w-0">
        {editing ? (
          <input
            type="text"
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={submitName}
            onKeyDown={(e) => {
              if (e.key === 'Enter') submitName();
              if (e.key === 'Escape') {
                setEditValue(run.name);
                setEditing(false);
              }
            }}
            className="w-full border rounded px-2 py-1 text-sm"
            autoFocus
          />
        ) : (
          <button
            type="button"
            onClick={() => {
              setEditValue(run.name);
              setEditing(true);
            }}
            className="text-left font-medium text-gray-800 hover:underline truncate block w-full"
          >
            {run.name}
          </button>
        )}
      </div>
      <span className="text-xs text-gray-500 shrink-0">
        {recordCount} rec · med {medianTotalMs.toFixed(0)} ms
      </span>
      <button
        type="button"
        onClick={onRemove}
        className="p-1 rounded hover:bg-red-50 text-gray-400 hover:text-red-600"
        title="Remove run"
        aria-label="Remove run"
      >
        <Trash2 className="w-4 h-4" />
      </button>
    </li>
  );
}
