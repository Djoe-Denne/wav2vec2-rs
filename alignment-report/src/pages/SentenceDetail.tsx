import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Plot from 'react-plotly.js';
import { useReports } from '../context/ReportContext';

export function SentenceDetail() {
  const { id } = useParams<{ id: string }>();
  const { selectedReport } = useReports();
  const navigate = useNavigate();

  const sentence = selectedReport?.data.sentences.find((s) => s.id === id);

  if (!sentence) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-500 mb-4">Sentence not found</p>
        <button
          onClick={() => navigate('/sentences')}
          className="text-blue-600 hover:underline"
        >
          Back to Sentences
        </button>
      </div>
    );
  }

  const hasStructuralIssues =
    sentence.structural.overlap_ratio > 0 ||
    sentence.structural.gap_ratio > 0.1 ||
    sentence.structural.non_monotonic_word_count > 0;

  const lowConfidenceHeavy = sentence.confidence.low_conf_word_ratio > 0.5;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">{sentence.id}</h2>
        <button
          onClick={() => navigate('/sentences')}
          className="text-blue-600 hover:underline"
        >
          Back to Sentences
        </button>
      </div>

      {(hasStructuralIssues || lowConfidenceHeavy) && (
        <div className="bg-yellow-50 border border-yellow-200 rounded p-4">
          <h3 className="font-semibold mb-2">Warnings</h3>
          <ul className="list-disc list-inside space-y-1 text-sm">
            {lowConfidenceHeavy && (
              <li>Low confidence heavy (ratio: {sentence.confidence.low_conf_word_ratio.toFixed(2)})</li>
            )}
            {hasStructuralIssues && (
              <li>
                Structural issues detected
                {sentence.structural.overlap_ratio > 0 && ` (overlap: ${sentence.structural.overlap_ratio.toFixed(2)})`}
                {sentence.structural.gap_ratio > 0.1 && ` (gap: ${sentence.structural.gap_ratio.toFixed(2)})`}
                {sentence.structural.non_monotonic_word_count > 0 && ` (non-monotonic words: ${sentence.structural.non_monotonic_word_count})`}
              </li>
            )}
          </ul>
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <InfoCard label="Split" value={sentence.split} />
        <InfoCard label="Duration" value={`${sentence.duration_ms} ms`} />
        <InfoCard
          label="Words"
          value={`${sentence.word_count_pred} / ${sentence.word_count_ref}`}
        />
        <InfoCard
          label="Has Reference"
          value={sentence.has_reference ? 'Yes' : 'No'}
        />
      </div>

      <div className="bg-white rounded shadow p-4">
        <h3 className="font-semibold mb-3">Confidence Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <InfoCard
            label="Mean Calibrated Confidence"
            value={sentence.confidence.word_conf_mean.toFixed(3)}
          />
          <InfoCard
            label="Min Calibrated Confidence"
            value={sentence.confidence.word_conf_min.toFixed(3)}
          />
          <InfoCard
            label="Low-Conf Threshold"
            value={(sentence.confidence.low_conf_threshold_used ?? 0.5).toFixed(2)}
          />
          <InfoCard
            label="Low Conf Ratio"
            value={(sentence.confidence.low_conf_word_ratio * 100).toFixed(1) + '%'}
          />
        </div>
      </div>

      <div className="bg-white rounded shadow p-4">
        <h3 className="font-semibold mb-3">Timing Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <InfoCard
            label="Abs Err Median"
            value={`${sentence.timing.abs_err_ms_median.toFixed(1)} ms`}
          />
          <InfoCard
            label="Abs Err P90"
            value={`${sentence.timing.abs_err_ms_p90.toFixed(1)} ms`}
          />
          <InfoCard
            label="Offset"
            value={`${sentence.timing.offset_ms.toFixed(1)} ms`}
          />
          <InfoCard
            label="Drift"
            value={`${sentence.timing.drift_ms_per_sec.toFixed(2)} ms/s`}
          />
        </div>
      </div>

      <div className="bg-white rounded shadow p-4">
        <h3 className="font-semibold mb-3">Structural Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <InfoCard
            label="Gap Ratio"
            value={(sentence.structural.gap_ratio * 100).toFixed(1) + '%'}
          />
          <InfoCard
            label="Overlap Ratio"
            value={(sentence.structural.overlap_ratio * 100).toFixed(1) + '%'}
          />
          <InfoCard
            label="Negative Duration"
            value={sentence.structural.negative_duration_word_count.toString()}
          />
          <InfoCard
            label="Non-Monotonic"
            value={sentence.structural.non_monotonic_word_count.toString()}
          />
        </div>
      </div>

      {sentence.per_word && sentence.per_word.length > 0 && (
        <>
          <div className="bg-white rounded shadow p-4">
            <h3 className="font-semibold mb-3">Word-Level Calibrated Confidence</h3>
            <Plot
              data={[
                {
                  x: sentence.per_word.map((_, i) => i),
                  y: sentence.per_word.map((w) => w.conf ?? 0),
                  type: 'scatter',
                  mode: 'lines+markers',
                  marker: { color: '#3b82f6' },
                  line: { color: '#3b82f6' },
                },
              ]}
              layout={{
                xaxis: { title: 'Word Index' },
                yaxis: { title: 'Calibrated Confidence Score' },
                margin: { l: 50, r: 20, t: 20, b: 50 },
                height: 300,
              }}
              config={{ displayModeBar: false }}
              className="w-full"
            />
          </div>

          <div className="bg-white rounded shadow p-4">
            <h3 className="font-semibold mb-3">Word-Level Timing Errors</h3>
            <Plot
              data={[
                {
                  x: sentence.per_word.map((_, i) => i),
                  y: sentence.per_word.map((w) => Math.abs(w.start_err_ms)),
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'Start Error',
                  marker: { color: '#10b981' },
                  line: { color: '#10b981' },
                },
                {
                  x: sentence.per_word.map((_, i) => i),
                  y: sentence.per_word.map((w) => Math.abs(w.end_err_ms)),
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'End Error',
                  marker: { color: '#f59e0b' },
                  line: { color: '#f59e0b' },
                },
              ]}
              layout={{
                xaxis: { title: 'Word Index' },
                yaxis: { title: 'Absolute Error (ms)' },
                margin: { l: 50, r: 20, t: 20, b: 50 },
                height: 300,
              }}
              config={{ displayModeBar: false }}
              className="w-full"
            />
          </div>

          <div className="bg-white rounded shadow overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 border-b">
                <tr>
                  <th className="px-4 py-2 text-left">Word</th>
                  <th className="px-4 py-2 text-right">Calib Conf</th>
                  <th className="px-4 py-2 text-right">Quality Conf</th>
                  <th className="px-4 py-2 text-right">Raw Geo Prob</th>
                  <th className="px-4 py-2 text-right">Start Err (ms)</th>
                  <th className="px-4 py-2 text-right">End Err (ms)</th>
                  <th className="px-4 py-2 text-right">Ref Start</th>
                  <th className="px-4 py-2 text-right">Ref End</th>
                </tr>
              </thead>
              <tbody>
                {sentence.per_word.map((word, i) => (
                  <tr key={i} className="border-b">
                    <td className="px-4 py-2 font-medium">{word.word}</td>
                    <td className="px-4 py-2 text-right">
                      {word.conf == null ? 'N/A' : word.conf.toFixed(3)}
                    </td>
                    <td className="px-4 py-2 text-right">
                      {word.quality_confidence == null
                        ? 'N/A'
                        : word.quality_confidence.toFixed(3)}
                    </td>
                    <td className="px-4 py-2 text-right">
                      {word.geo_mean_prob == null ? 'N/A' : word.geo_mean_prob.toFixed(3)}
                    </td>
                    <td className="px-4 py-2 text-right">
                      {word.start_err_ms.toFixed(1)}
                    </td>
                    <td className="px-4 py-2 text-right">
                      {word.end_err_ms.toFixed(1)}
                    </td>
                    <td className="px-4 py-2 text-right">
                      {word.ref_start_ms}
                    </td>
                    <td className="px-4 py-2 text-right">
                      {word.ref_end_ms}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function InfoCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="border rounded p-3">
      <div className="text-sm text-gray-600">{label}</div>
      <div className="text-lg font-semibold mt-1">{value}</div>
    </div>
  );
}
