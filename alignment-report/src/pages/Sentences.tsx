import React, { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useReports } from '../context/ReportContext';
import { filterSentences } from '../lib/aggregate';
import { Sentence } from '../types/report';

type SortKey = keyof Sentence | 'word_conf_mean' | 'abs_err_ms_median' | 'drift_ms_per_sec';

export function Sentences() {
  const { selectedReport, filters } = useReports();
  const navigate = useNavigate();
  const [sortKey, setSortKey] = useState<SortKey>('id');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  const filteredSentences = useMemo(() => {
    if (!selectedReport) return [];
    return filterSentences(selectedReport.data.sentences, filters);
  }, [selectedReport, filters]);

  const sortedSentences = useMemo(() => {
    const sorted = [...filteredSentences].sort((a, b) => {
      let aVal: any;
      let bVal: any;

      if (sortKey === 'word_conf_mean') {
        aVal = a.confidence.word_conf_mean;
        bVal = b.confidence.word_conf_mean;
      } else if (sortKey === 'abs_err_ms_median') {
        aVal = a.timing.abs_err_ms_median;
        bVal = b.timing.abs_err_ms_median;
      } else if (sortKey === 'drift_ms_per_sec') {
        aVal = a.timing.drift_ms_per_sec;
        bVal = b.timing.drift_ms_per_sec;
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
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 border-b">
            <tr>
              <SortableHeader label="ID" sortKey="id" currentKey={sortKey} direction={sortDirection} onSort={handleSort} />
              <SortableHeader label="Split" sortKey="split" currentKey={sortKey} direction={sortDirection} onSort={handleSort} />
              <SortableHeader label="Duration (ms)" sortKey="duration_ms" currentKey={sortKey} direction={sortDirection} onSort={handleSort} />
              <SortableHeader label="Words" sortKey="word_count_pred" currentKey={sortKey} direction={sortDirection} onSort={handleSort} />
              <SortableHeader label="Calib Conf" sortKey="word_conf_mean" currentKey={sortKey} direction={sortDirection} onSort={handleSort} />
              <SortableHeader label="Low Conf %" sortKey="low_conf_word_ratio" currentKey={sortKey} direction={sortDirection} onSort={handleSort} />
              <SortableHeader label="Abs Err Med" sortKey="abs_err_ms_median" currentKey={sortKey} direction={sortDirection} onSort={handleSort} />
              <SortableHeader label="Abs Err P90" sortKey="abs_err_ms_p90" currentKey={sortKey} direction={sortDirection} onSort={handleSort} />
              <SortableHeader label="Drift" sortKey="drift_ms_per_sec" currentKey={sortKey} direction={sortDirection} onSort={handleSort} />
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
}: {
  label: string;
  sortKey: SortKey;
  currentKey: SortKey;
  direction: 'asc' | 'desc';
  onSort: (key: SortKey) => void;
}) {
  const isActive = currentKey === sortKey;

  return (
    <th
      className="px-4 py-2 text-left font-medium cursor-pointer hover:bg-gray-100"
      onClick={() => onSort(sortKey)}
    >
      <div className="flex items-center gap-1">
        {label}
        {isActive && (
          <span className="text-xs">{direction === 'asc' ? '▲' : '▼'}</span>
        )}
      </div>
    </th>
  );
}
