import React, { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { useReports } from '../context/ReportContext';
import { computeAggregates } from '../lib/aggregate';

export function Compare() {
  const { reports } = useReports();
  const [runAId, setRunAId] = useState<string>('');
  const [runBId, setRunBId] = useState<string>('');

  const runA = reports.find((r) => r.id === runAId);
  const runB = reports.find((r) => r.id === runBId);

  const aggregatesA = useMemo(() => {
    if (!runA) return null;
    return computeAggregates(runA.data.sentences);
  }, [runA]);

  const aggregatesB = useMemo(() => {
    if (!runB) return null;
    return computeAggregates(runB.data.sentences);
  }, [runB]);

  if (reports.length < 2) {
    return (
      <div className="text-center py-8 text-gray-500">
        Please load at least 2 reports to compare
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded shadow p-4">
        <h2 className="text-xl font-bold mb-4">Select Reports to Compare</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2">Run A</label>
            <select
              value={runAId}
              onChange={(e) => setRunAId(e.target.value)}
              className="w-full border rounded px-3 py-2"
            >
              <option value="">Select a run...</option>
              {reports.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.filename} ({new Date(r.loadedAt).toLocaleString()})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Run B</label>
            <select
              value={runBId}
              onChange={(e) => setRunBId(e.target.value)}
              className="w-full border rounded px-3 py-2"
            >
              <option value="">Select a run...</option>
              {reports.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.filename} ({new Date(r.loadedAt).toLocaleString()})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {aggregatesA && aggregatesB && (
        <>
          <div className="bg-white rounded shadow p-4">
            <h3 className="font-semibold mb-4">Comparison Summary</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 border-b">
                  <tr>
                    <th className="px-4 py-2 text-left">Metric</th>
                    <th className="px-4 py-2 text-right">Run A</th>
                    <th className="px-4 py-2 text-right">Run B</th>
                    <th className="px-4 py-2 text-right">Delta (B-A)</th>
                  </tr>
                </thead>
                <tbody>
                  <CompareRow
                    label="Count"
                    valueA={aggregatesA.count}
                    valueB={aggregatesB.count}
                    format={(v) => v.toString()}
                  />
                  <CompareRow
                    label="Avg Confidence"
                    valueA={aggregatesA.confidence.word_conf_mean.mean}
                    valueB={aggregatesB.confidence.word_conf_mean.mean}
                    format={(v) => v.toFixed(3)}
                  />
                  <CompareRow
                    label="Median Abs Err (ms)"
                    valueA={aggregatesA.timing.abs_err_ms_median.median}
                    valueB={aggregatesB.timing.abs_err_ms_median.median}
                    format={(v) => v.toFixed(1)}
                  />
                  <CompareRow
                    label="P90 Abs Err (ms)"
                    valueA={aggregatesA.timing.abs_err_ms_p90.median}
                    valueB={aggregatesB.timing.abs_err_ms_p90.median}
                    format={(v) => v.toFixed(1)}
                  />
                  <CompareRow
                    label="Avg Offset (ms)"
                    valueA={aggregatesA.timing.offset_ms.mean}
                    valueB={aggregatesB.timing.offset_ms.mean}
                    format={(v) => v.toFixed(1)}
                  />
                  <CompareRow
                    label="Avg Drift (ms/s)"
                    valueA={aggregatesA.timing.drift_ms_per_sec.mean}
                    valueB={aggregatesB.timing.drift_ms_per_sec.mean}
                    format={(v) => v.toFixed(2)}
                  />
                  <CompareRow
                    label="Avg Low Conf Ratio"
                    valueA={aggregatesA.confidence.low_conf_word_ratio.mean}
                    valueB={aggregatesB.confidence.low_conf_word_ratio.mean}
                    format={(v) => v.toFixed(3)}
                  />
                  <CompareRow
                    label="Avg Gap Ratio"
                    valueA={aggregatesA.structural.gap_ratio.mean}
                    valueB={aggregatesB.structural.gap_ratio.mean}
                    format={(v) => v.toFixed(3)}
                  />
                  <CompareRow
                    label="Avg Overlap Ratio"
                    valueA={aggregatesA.structural.overlap_ratio.mean}
                    valueB={aggregatesB.structural.overlap_ratio.mean}
                    format={(v) => v.toFixed(3)}
                  />
                </tbody>
              </table>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded shadow p-4">
              <h3 className="font-semibold mb-2">Confidence Distribution Comparison</h3>
              <Plot
                data={[
                  {
                    x: runA!.data.sentences.map((s) => s.confidence.word_conf_mean),
                    type: 'histogram',
                    name: 'Run A',
                    opacity: 0.6,
                    marker: { color: '#3b82f6' },
                  },
                  {
                    x: runB!.data.sentences.map((s) => s.confidence.word_conf_mean),
                    type: 'histogram',
                    name: 'Run B',
                    opacity: 0.6,
                    marker: { color: '#f59e0b' },
                  },
                ]}
                layout={{
                  barmode: 'overlay',
                  xaxis: { title: 'Confidence' },
                  yaxis: { title: 'Count' },
                  margin: { l: 50, r: 20, t: 20, b: 50 },
                  height: 300,
                }}
                config={{ displayModeBar: false }}
                className="w-full"
              />
            </div>

            <div className="bg-white rounded shadow p-4">
              <h3 className="font-semibold mb-2">Timing Error Comparison</h3>
              <Plot
                data={[
                  {
                    x: runA!.data.sentences.map((s) => s.timing.abs_err_ms_median),
                    type: 'histogram',
                    name: 'Run A',
                    opacity: 0.6,
                    marker: { color: '#3b82f6' },
                  },
                  {
                    x: runB!.data.sentences.map((s) => s.timing.abs_err_ms_median),
                    type: 'histogram',
                    name: 'Run B',
                    opacity: 0.6,
                    marker: { color: '#f59e0b' },
                  },
                ]}
                layout={{
                  barmode: 'overlay',
                  xaxis: { title: 'Abs Error Median (ms)' },
                  yaxis: { title: 'Count' },
                  margin: { l: 50, r: 20, t: 20, b: 50 },
                  height: 300,
                }}
                config={{ displayModeBar: false }}
                className="w-full"
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function CompareRow({
  label,
  valueA,
  valueB,
  format,
}: {
  label: string;
  valueA: number;
  valueB: number;
  format: (v: number) => string;
}) {
  const delta = valueB - valueA;
  const deltaStr = format(delta);
  const isPositive = delta > 0;
  const isZero = Math.abs(delta) < 0.001;

  return (
    <tr className="border-b">
      <td className="px-4 py-2">{label}</td>
      <td className="px-4 py-2 text-right">{format(valueA)}</td>
      <td className="px-4 py-2 text-right">{format(valueB)}</td>
      <td
        className={`px-4 py-2 text-right font-medium ${
          isZero ? '' : isPositive ? 'text-red-600' : 'text-green-600'
        }`}
      >
        {isPositive ? '+' : ''}
        {deltaStr}
      </td>
    </tr>
  );
}
