import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useReports } from '../context/ReportContext';
import { filterSentences } from '../lib/aggregate';
import { Sentence } from '../types/report';

type SortKey =
  | keyof Sentence
  | 'word_conf_mean'
  | 'low_conf_word_ratio'
  | 'low_conf_threshold_used'
  | 'abs_err_ms_median'
  | 'abs_err_ms_p90'
  | 'drift_ms_per_sec'
  | 'drift_delta_ms';

const OUTLIER_TAG_HELP: Record<string, string> = {
  abs_p90: 'Top sentence-level P90 timing-error outlier.',
  drift: 'Top robust drift outlier (ranked by absolute drift).',
  low_conf: 'Top low-confidence ratio outlier.',
};

export function Sentences() {
  const { selectedReport, filters } = useReports();
  const navigate = useNavigate();
  const [sortKey, setSortKey] = useState<SortKey>('id');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  const filteredSentences = useMemo(() => {
    if (!selectedReport) return [];
    return filterSentences(selectedReport.data.sentences, filters);
  }, [selectedReport, filters]);

  const outlierTagsById = useMemo(() => {
    const tagsById = new Map<string, string[]>();
    const outliers = selectedReport?.data.aggregates?.outliers;

    const addTag = (id: string, tag: string) => {
      const existing = tagsById.get(id) ?? [];
      if (!existing.includes(tag)) {
        existing.push(tag);
      }
      tagsById.set(id, existing);
    };

    outliers?.worst_abs_err_ms_p90?.forEach((entry) => addTag(entry.id, 'abs_p90'));
    outliers?.worst_drift_ms_per_sec?.forEach((entry) => addTag(entry.id, 'drift'));
    outliers?.worst_low_conf_word_ratio?.forEach((entry) => addTag(entry.id, 'low_conf'));

    return tagsById;
  }, [selectedReport]);

  const sortedSentences = useMemo(() => {
    const sorted = [...filteredSentences].sort((a, b) => {
      let aVal: any;
      let bVal: any;

      if (sortKey === 'word_conf_mean') {
        aVal = a.confidence.word_conf_mean;
        bVal = b.confidence.word_conf_mean;
      } else if (sortKey === 'low_conf_word_ratio') {
        aVal = a.confidence.low_conf_word_ratio;
        bVal = b.confidence.low_conf_word_ratio;
      } else if (sortKey === 'low_conf_threshold_used') {
        aVal = a.confidence.low_conf_threshold_used ?? 0.5;
        bVal = b.confidence.low_conf_threshold_used ?? 0.5;
      } else if (sortKey === 'abs_err_ms_median') {
        aVal = a.timing.abs_err_ms_median;
        bVal = b.timing.abs_err_ms_median;
      } else if (sortKey === 'abs_err_ms_p90') {
        aVal = a.timing.abs_err_ms_p90;
        bVal = b.timing.abs_err_ms_p90;
      } else if (sortKey === 'drift_ms_per_sec') {
        aVal = a.timing.drift_ms_per_sec;
        bVal = b.timing.drift_ms_per_sec;
      } else if (sortKey === 'drift_delta_ms') {
        aVal = a.timing.drift_delta_ms ?? 0;
        bVal = b.timing.drift_delta_ms ?? 0;
      } else {
        aVal = a[sortKey as keyof Sentence];
        bVal = b[sortKey as keyof Sentence];
      }

      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
      }

      const aStr = String(aVal);
      const bStr = String(bVal);
      return sortDirection === 'asc'
        ? aStr.localeCompare(bStr)
        : bStr.localeCompare(aStr);
    });

    return sorted;
  }, [filteredSentences, sortKey, sortDirection]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDirection('asc');
    }
  };

  if (!selectedReport) {
    return (
      <div className="text-center py-8 text-gray-500">
        Please load a report to view sentences
      </div>
    );
  }

  return (
    <div className="bg-white rounded shadow">
      <div className="p-4 border-b">
        <h2 className="text-xl font-bold">
          Sentences ({sortedSentences.length})
        </h2>
        <p className="text-xs text-gray-600 mt-1">
          Hover sortable headers and outlier tags for metric definitions.
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 border-b">
            <tr>
              <SortableHeader
                label="ID"
                sortKey="id"
                currentKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
                description="Sentence identifier from the dataset."
              />
              <SortableHeader
                label="Split"
                sortKey="split"
                currentKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
                description="Dataset subset: clean or other."
              />
              <SortableHeader
                label="Duration (ms)"
                sortKey="duration_ms"
                currentKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
                description="Audio duration in milliseconds."
              />
              <SortableHeader
                label="Words"
                sortKey="word_count_pred"
                currentKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
                description="Predicted/reference word counts."
              />
              <SortableHeader
                label="Calib Conf"
                sortKey="word_conf_mean"
                currentKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
                description="Sentence mean calibrated confidence."
              />
              <SortableHeader
                label="Low Conf Thresh"
                sortKey="low_conf_threshold_used"
                currentKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
                description="Adaptive threshold used to classify low-confidence words."
              />
              <SortableHeader
                label="Low Conf %"
                sortKey="low_conf_word_ratio"
                currentKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
                description="Fraction of words below the low-confidence threshold."
              />
              <SortableHeader
                label="Abs Err Med"
                sortKey="abs_err_ms_median"
                currentKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
                description="Median absolute timing error (ms)."
              />
              <SortableHeader
                label="Abs Err P90"
                sortKey="abs_err_ms_p90"
                currentKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
                description="90th percentile absolute timing error (ms)."
              />
              <SortableHeader
                label="Drift"
                sortKey="drift_ms_per_sec"
                currentKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
                description="Timing drift rate in ms/s."
              />
              <SortableHeader
                label="Drift Δ (ms)"
                sortKey="drift_delta_ms"
                currentKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
                description="End minus start signed error in ms."
              />
              <th
                className="px-4 py-2 text-left font-medium"
                title="Outlier tags from backend ranked lists."
              >
                Outliers
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedSentences.map((sentence) => (
              <tr
                key={sentence.id}
                className="border-b hover:bg-gray-50 cursor-pointer"
                onClick={() => navigate(`/sentences/${sentence.id}`)}
              >
                <td className="px-4 py-2">{sentence.id}</td>
                <td className="px-4 py-2">
                  <span
                    className={`px-2 py-1 rounded text-xs ${
                      sentence.split === 'clean'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-gray-100 text-gray-800'
                    }`}
                  >
                    {sentence.split}
                  </span>
                </td>
                <td className="px-4 py-2">{sentence.duration_ms}</td>
                <td className="px-4 py-2">
                  {sentence.word_count_pred} / {sentence.word_count_ref}
                </td>
                <td className="px-4 py-2">
                  {sentence.confidence.word_conf_mean.toFixed(3)}
                </td>
                <td className="px-4 py-2">
                  {(sentence.confidence.low_conf_threshold_used ?? 0.5).toFixed(2)}
                </td>
                <td className="px-4 py-2">
                  {(sentence.confidence.low_conf_word_ratio * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-2">
                  {sentence.timing.abs_err_ms_median.toFixed(1)}
                </td>
                <td className="px-4 py-2">
                  {sentence.timing.abs_err_ms_p90.toFixed(1)}
                </td>
                <td className="px-4 py-2">
                  {sentence.timing.drift_ms_per_sec.toFixed(2)}
                </td>
                <td className="px-4 py-2">
                  {sentence.timing.drift_delta_ms == null
                    ? 'N/A'
                    : sentence.timing.drift_delta_ms.toFixed(1)}
                </td>
                <td className="px-4 py-2">
                  <div className="flex flex-wrap gap-1">
                    {(outlierTagsById.get(sentence.id) ?? []).map((tag) => (
                      <span
                        key={`${sentence.id}-${tag}`}
                        className="px-2 py-0.5 rounded text-xs bg-indigo-100 text-indigo-700"
                        title={OUTLIER_TAG_HELP[tag] ?? 'Outlier tag'}
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SortableHeader({
  label,
  sortKey,
  currentKey,
  direction,
  onSort,
  description,
}: {
  label: string;
  sortKey: SortKey;
  currentKey: SortKey;
  direction: 'asc' | 'desc';
  onSort: (key: SortKey) => void;
  description?: string;
}) {
  const isActive = currentKey === sortKey;

  return (
    <th
      className="px-4 py-2 text-left font-medium cursor-pointer hover:bg-gray-100"
      onClick={() => onSort(sortKey)}
      title={description}
    >
      <div className="flex items-center gap-1">
        <span className="underline decoration-dotted">{label}</span>
        {isActive && (
          <span className="text-xs">{direction === 'asc' ? '▲' : '▼'}</span>
        )}
      </div>
    </th>
  );
}
