import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { useReports } from '../context/ReportContext';
import { filterSentences, computeAggregates, createHistogramBins } from '../lib/aggregate';

export function Overview() {
  const { selectedReport, filters } = useReports();

  const filteredSentences = useMemo(() => {
    if (!selectedReport) return [];
    return filterSentences(selectedReport.data.sentences, filters);
  }, [selectedReport, filters]);

  const aggregates = useMemo(() => {
    if (filteredSentences.length === 0) return null;
    return computeAggregates(filteredSentences);
  }, [filteredSentences]);

  if (!selectedReport) {
    return (
      <div className="text-center py-8 text-gray-500">
        Please load a report to view metrics
      </div>
    );
  }

  if (!aggregates) {
    return (
      <div className="text-center py-8 text-gray-500">
        No sentences match the current filters
      </div>
    );
  }

  const confValues = filteredSentences.map((s) => s.confidence.word_conf_mean);
  const absErrMedianValues = filteredSentences.map((s) => s.timing.abs_err_ms_median);

  return (
    <div className="space-y-6">
      <div className="bg-white p-4 rounded shadow">
        <h2 className="text-xl font-bold mb-2">Report Metadata</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="font-medium">Model:</span> {selectedReport.data.meta.model_path.split('/').pop()}
          </div>
          <div>
            <span className="font-medium">Device:</span> {selectedReport.data.meta.device}
          </div>
          <div>
            <span className="font-medium">Frame Stride:</span> {selectedReport.data.meta.frame_stride_ms}ms
          </div>
          <div>
            <span className="font-medium">Total Cases:</span> {selectedReport.data.meta.case_count}
          </div>
        </div>
      </div>

      <div className="bg-white p-4 rounded shadow">
        <h2 className="text-xl font-bold mb-4">Aggregates</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            title="Count"
            value={aggregates.count.toString()}
          />
          {Object.entries(aggregates.bySpli).map(([split, count]) => (
            <MetricCard
              key={split}
              title={`${split} count`}
              value={count.toString()}
            />
          ))}
          <MetricCard
            title="Avg Confidence"
            value={aggregates.confidence.word_conf_mean.mean.toFixed(3)}
          />
          <MetricCard
            title="Median Abs Err (ms)"
            value={aggregates.timing.abs_err_ms_median.median.toFixed(1)}
          />
          <MetricCard
            title="P90 Abs Err (ms)"
            value={aggregates.timing.abs_err_ms_p90.median.toFixed(1)}
          />
          <MetricCard
            title="Avg Offset (ms)"
            value={aggregates.timing.offset_ms.mean.toFixed(1)}
          />
          <MetricCard
            title="Avg Drift (ms/s)"
            value={aggregates.timing.drift_ms_per_sec.mean.toFixed(2)}
          />
          <MetricCard
            title="Avg Low Conf Ratio"
            value={aggregates.confidence.low_conf_word_ratio.mean.toFixed(3)}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-4 rounded shadow">
          <h3 className="font-semibold mb-2">Confidence Distribution</h3>
          <Plot
            data={[
              {
                x: confValues,
                type: 'histogram',
                nbinsx: 30,
                marker: { color: '#3b82f6' },
              },
            ]}
            layout={{
              xaxis: { title: 'Word Confidence Mean' },
              yaxis: { title: 'Count' },
              margin: { l: 50, r: 20, t: 20, b: 50 },
              height: 300,
            }}
            config={{ displayModeBar: false }}
            className="w-full"
          />
        </div>

        <div className="bg-white p-4 rounded shadow">
          <h3 className="font-semibold mb-2">Timing Error Distribution</h3>
          <Plot
            data={[
              {
                x: absErrMedianValues,
                type: 'histogram',
                nbinsx: 30,
                marker: { color: '#10b981' },
              },
            ]}
            layout={{
              xaxis: { title: 'Abs Error Median (ms)' },
              yaxis: { title: 'Count' },
              margin: { l: 50, r: 20, t: 20, b: 50 },
              height: 300,
            }}
            config={{ displayModeBar: false }}
            className="w-full"
          />
        </div>

        <div className="bg-white p-4 rounded shadow">
          <h3 className="font-semibold mb-2">Confidence vs Timing Error</h3>
          <Plot
            data={[
              {
                x: confValues,
                y: absErrMedianValues,
                mode: 'markers',
                type: 'scatter',
                marker: {
                  color: filteredSentences.map((s) => s.duration_ms),
                  colorscale: 'Viridis',
                  showscale: true,
                  colorbar: { title: 'Duration (ms)' },
                },
              },
            ]}
            layout={{
              xaxis: { title: 'Word Confidence Mean' },
              yaxis: { title: 'Abs Error Median (ms)' },
              margin: { l: 50, r: 20, t: 20, b: 50 },
              height: 300,
            }}
            config={{ displayModeBar: false }}
            className="w-full"
          />
        </div>

        <div className="bg-white p-4 rounded shadow">
          <h3 className="font-semibold mb-2">Sentences by Split</h3>
          <Plot
            data={[
              {
                x: Object.keys(aggregates.bySpli),
                y: Object.values(aggregates.bySpli),
                type: 'bar',
                marker: { color: '#f59e0b' },
              },
            ]}
            layout={{
              xaxis: { title: 'Split' },
              yaxis: { title: 'Count' },
              margin: { l: 50, r: 20, t: 20, b: 50 },
              height: 300,
            }}
            config={{ displayModeBar: false }}
            className="w-full"
          />
        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, value }: { title: string; value: string }) {
  return (
    <div className="border rounded p-3">
      <div className="text-sm text-gray-600">{title}</div>
      <div className="text-2xl font-bold mt-1">{value}</div>
    </div>
  );
}
