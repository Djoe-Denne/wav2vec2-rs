import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { useReports } from '../context/ReportContext';
import { filterSentences, computeAggregates } from '../lib/aggregate';
import type { OutlierEntry } from '../types/report';

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
  const cleanSentences = filteredSentences.filter((s) => s.split === 'clean');
  const otherSentences = filteredSentences.filter((s) => s.split === 'other');
  const cleanConfValues = cleanSentences.map((s) => s.confidence.word_conf_mean);
  const otherConfValues = otherSentences.map((s) => s.confidence.word_conf_mean);
  const cleanAbsErrMedianValues = cleanSentences.map((s) => s.timing.abs_err_ms_median);
  const otherAbsErrMedianValues = otherSentences.map((s) => s.timing.abs_err_ms_median);
  const backendAggregates = selectedReport.data.aggregates;
  const globalBackend = backendAggregates?.global;
  const sentencePassRates = globalBackend?.abs_err_ms_p90_pass_rate ?? null;
  const wordPassRates = globalBackend?.word_abs_err_pass_rate ?? null;
  const passRateLabels = ['<=50ms', '<=100ms', '<=150ms'];
  const passRateRows = [
    {
      threshold: '<=50ms',
      sentence: sentencePassRates?.le_50_ms ?? null,
      word: wordPassRates?.le_50_ms ?? null,
    },
    {
      threshold: '<=100ms',
      sentence: sentencePassRates?.le_100_ms ?? null,
      word: wordPassRates?.le_100_ms ?? null,
    },
    {
      threshold: '<=150ms',
      sentence: sentencePassRates?.le_150_ms ?? null,
      word: wordPassRates?.le_150_ms ?? null,
    },
  ];
  const passRateTraces: any[] = [];

  if (sentencePassRates) {
    passRateTraces.push({
      x: passRateLabels,
      y: [
        sentencePassRates.le_50_ms * 100,
        sentencePassRates.le_100_ms * 100,
        sentencePassRates.le_150_ms * 100,
      ],
      type: 'bar',
      name: 'Sentence abs_err_p90',
      marker: { color: '#6366f1' },
    });
  }
  if (wordPassRates) {
    passRateTraces.push({
      x: passRateLabels,
      y: [
        wordPassRates.le_50_ms * 100,
        wordPassRates.le_100_ms * 100,
        wordPassRates.le_150_ms * 100,
      ],
      type: 'bar',
      name: 'Word abs error',
      marker: { color: '#14b8a6' },
    });
  }

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
            description="Number of sentences after applying current filters."
          />
          {Object.entries(aggregates.bySpli).map(([split, count]) => (
            <MetricCard
              key={split}
              title={`${split} count`}
              value={count.toString()}
              description={`Filtered sentence count for the "${split}" split.`}
            />
          ))}
          <MetricCard
            title="Avg Calibrated Confidence"
            value={aggregates.confidence.word_conf_mean.mean.toFixed(3)}
            description="Average confidence score across filtered sentences. Higher is usually better."
          />
          <MetricCard
            title="Median Abs Err (ms)"
            value={aggregates.timing.abs_err_ms_median.median.toFixed(1)}
            description="Typical absolute timing error (ms) across filtered sentences."
          />
          <MetricCard
            title="P90 Abs Err (ms)"
            value={aggregates.timing.abs_err_ms_p90.median.toFixed(1)}
            description="Tail-focused timing error summary. Higher means more severe worst-word timing."
          />
          <MetricCard
            title="Avg Offset (ms)"
            value={aggregates.timing.offset_ms.mean.toFixed(1)}
            description="Average signed center shift. Positive means predicted timings are later than reference."
          />
          <MetricCard
            title="Avg Drift (ms/s)"
            value={aggregates.timing.drift_ms_per_sec.mean.toFixed(2)}
            description="Average timing slope over each sentence. Negative means ending tends to be earlier than starting error."
          />
          <MetricCard
            title="Avg Low Conf Ratio"
            value={aggregates.confidence.low_conf_word_ratio.mean.toFixed(3)}
            description="Average fraction of words flagged as low confidence in each sentence."
          />
        </div>
      </div>

      {backendAggregates && (
        <div className="bg-white p-4 rounded shadow">
          <h2 className="text-xl font-bold mb-1">Backend Diagnostics (Full Run)</h2>
          <p className="text-sm text-gray-600 mb-4">
            Values below come directly from report `aggregates` and are not affected by page filters.
          </p>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              title="Word Abs Err Mean (ms)"
              value={formatMetric(globalBackend?.word_abs_err_ms?.mean, 1)}
              description="Mean absolute word-boundary error from backend aggregate metrics."
            />
            <MetricCard
              title="Word Abs Err P90 (ms)"
              value={formatMetric(globalBackend?.word_abs_err_ms?.p90, 1)}
              description="90th percentile of absolute word-boundary error."
            />
            <MetricCard
              title="Drift Delta Mean (ms)"
              value={formatMetric(globalBackend?.drift_delta_ms?.mean, 2)}
              description="Mean end-minus-start signed error in milliseconds (not normalized by duration)."
            />
            <MetricCard
              title="Sentence Pass <=100ms"
              value={formatPercent(globalBackend?.abs_err_ms_p90_pass_rate?.le_100_ms)}
              description="Share of sentences whose abs_err_ms_p90 is at or below 100 ms."
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            <div className="border rounded p-3">
              <h3 className="font-semibold mb-2">Pass Rate Summary</h3>
              <p className="text-xs text-gray-600 mb-3">
                Cumulative pass rates at timing thresholds. Higher is better; values at
                {' <=150ms '}should be at least as high as{' <=100ms '}and{' <=50ms '}.
              </p>
              {passRateTraces.length > 0 ? (
                <>
                  <Plot
                    data={passRateTraces}
                    layout={{
                      barmode: 'group',
                      xaxis: { title: 'Threshold' },
                      yaxis: { title: 'Pass Rate (%)', range: [0, 100] },
                      margin: { l: 50, r: 20, t: 20, b: 50 },
                      height: 300,
                    }}
                    config={{ displayModeBar: false }}
                    className="w-full"
                  />

                  <div className="mt-3 border rounded p-3 bg-gray-50 text-xs text-gray-700 space-y-2">
                    <p>
                      <span className="font-semibold">Sentence abs_err_p90</span>: percentage of
                      sentences where the sentence-level P90 absolute timing error is at or below
                      the threshold.
                    </p>
                    <p>
                      <span className="font-semibold">Word abs error</span>: percentage of all
                      word-boundary absolute errors at or below the threshold.
                    </p>
                    <p>
                      <span className="font-semibold">How to read gaps</span>: if word pass is much
                      higher than sentence pass, many words are fine but each sentence still has a
                      small tail of difficult words.
                    </p>

                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-1">Threshold</th>
                            <th className="text-right py-1">Sentence abs_err_p90</th>
                            <th className="text-right py-1">Word abs error</th>
                          </tr>
                        </thead>
                        <tbody>
                          {passRateRows.map((row) => (
                            <tr key={row.threshold} className="border-b last:border-b-0">
                              <td className="py-1">{row.threshold}</td>
                              <td className="py-1 text-right">
                                {formatPercent(row.sentence)}
                              </td>
                              <td className="py-1 text-right">{formatPercent(row.word)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-sm text-gray-500 py-8">
                  Pass-rate metrics are not available in this report file.
                </div>
              )}
            </div>

            <div className="space-y-4">
              <OutlierList
                title="Worst Abs Err P90"
                entries={backendAggregates.outliers?.worst_abs_err_ms_p90}
                valueFormatter={(value) => `${value.toFixed(1)} ms`}
                description="Sentences with the largest tail timing errors."
              />
              <OutlierList
                title="Worst Drift (robust ranking)"
                entries={backendAggregates.outliers?.worst_drift_ms_per_sec}
                valueFormatter={(value) => `${value.toFixed(2)} ms/s`}
                description="Largest absolute drift after excluding very short utterances."
              />
              <OutlierList
                title="Worst Low Confidence Ratio"
                entries={backendAggregates.outliers?.worst_low_conf_word_ratio}
                valueFormatter={(value) => `${(value * 100).toFixed(1)}%`}
                description="Sentences where the biggest share of words are low-confidence."
              />
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-4 rounded shadow">
          <h3 className="font-semibold mb-2">Calibrated Confidence Distribution</h3>
          <Plot
            data={[
              {
                x: cleanConfValues,
                type: 'histogram',
                name: 'clean',
                nbinsx: 30,
                marker: { color: '#3b82f6' },
                opacity: 0.65,
              },
              {
                x: otherConfValues,
                type: 'histogram',
                name: 'other',
                nbinsx: 30,
                marker: { color: '#f59e0b' },
                opacity: 0.65,
              },
            ]}
            layout={{
              barmode: 'overlay',
              xaxis: { title: 'Word Calibrated Confidence Mean' },
              yaxis: { title: 'Count' },
              margin: { l: 50, r: 20, t: 20, b: 50 },
              height: 300,
            }}
            config={{ displayModeBar: false }}
            className="w-full"
          />
          <ChartHelp text="Histogram of sentence-level mean confidence. Blue/Orange overlap shows how clean vs other differ in confidence quality." />
        </div>

        <div className="bg-white p-4 rounded shadow">
          <h3 className="font-semibold mb-2">Timing Error Distribution</h3>
          <Plot
            data={[
              {
                x: cleanAbsErrMedianValues,
                type: 'histogram',
                name: 'clean',
                nbinsx: 30,
                marker: { color: '#10b981' },
                opacity: 0.65,
              },
              {
                x: otherAbsErrMedianValues,
                type: 'histogram',
                name: 'other',
                nbinsx: 30,
                marker: { color: '#ef4444' },
                opacity: 0.65,
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
          <ChartHelp text="Histogram of median absolute timing error per sentence (ms). More mass near 0 means better alignment timing." />
        </div>

        <div className="bg-white p-4 rounded shadow">
          <h3 className="font-semibold mb-2">Calibrated Confidence vs Timing Error</h3>
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
              xaxis: { title: 'Word Calibrated Confidence Mean' },
              yaxis: { title: 'Abs Error Median (ms)' },
              margin: { l: 50, r: 20, t: 20, b: 50 },
              height: 300,
            }}
            config={{ displayModeBar: false }}
            className="w-full"
          />
          <ChartHelp text="Each point is a sentence: x=confidence, y=timing error, color=duration. Top-left is best (high confidence, low error)." />
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
          <ChartHelp text="Count of filtered sentences in each LibriSpeech split. Use this to verify the data mix behind other charts." />
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  title,
  value,
  description,
}: {
  title: string;
  value: string;
  description?: string;
}) {
  return (
    <div className="border rounded p-3">
      <div
        className="text-sm text-gray-600 underline decoration-dotted cursor-help"
        title={description ?? title}
      >
        {title}
      </div>
      <div className="text-2xl font-bold mt-1">{value}</div>
      {description && <div className="text-xs text-gray-500 mt-2">{description}</div>}
    </div>
  );
}

function OutlierList({
  title,
  entries,
  valueFormatter,
  description,
}: {
  title: string;
  entries?: OutlierEntry[];
  valueFormatter: (value: number) => string;
  description?: string;
}) {
  const topEntries = (entries ?? []).slice(0, 5);

  return (
    <div className="border rounded p-3">
      <h3 className="font-semibold mb-2">{title}</h3>
      {description && <p className="text-xs text-gray-600 mb-2">{description}</p>}
      {topEntries.length === 0 ? (
        <div className="text-sm text-gray-500">No data</div>
      ) : (
        <div className="space-y-1 text-sm">
          {topEntries.map((entry) => (
            <div key={`${title}-${entry.id}`} className="flex justify-between gap-2">
              <span className="truncate">
                {entry.id} <span className="text-gray-500">({entry.split})</span>
              </span>
              <span className="font-medium">{valueFormatter(entry.value)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function formatMetric(value?: number | null, decimals: number = 2): string {
  if (value == null || !Number.isFinite(value)) {
    return 'N/A';
  }
  return value.toFixed(decimals);
}

function formatPercent(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) {
    return 'N/A';
  }
  return `${(value * 100).toFixed(1)}%`;
}

function ChartHelp({ text }: { text: string }) {
  return <p className="text-xs text-gray-600 mt-2">{text}</p>;
}
