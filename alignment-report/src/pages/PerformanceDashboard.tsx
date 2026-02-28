import { useMemo, useState } from 'react';
import { useReports } from '../context/ReportContext';
import { MetricTooltip } from '../components/MetricTooltip';
import { ChartWithFullscreen } from '../components/ChartWithFullscreen';
import {
  filterRecords,
  computeRunAggregates,
  ecdf,
  rtf,
  costPerFrame,
  speedupVsBaseline,
  peakMemory,
  binAndMedian,
  type StageKey,
  STAGE_KEYS,
} from '../lib/perfMetrics';
import type { LoadedRun } from '../types/report';

const RUN_COLORS = ['#2563eb', '#ea580c', '#059669', '#dc2626', '#7c3aed', '#0891b2'];
const STAGE_COLORS: Record<StageKey, string> = {
  forward: '#2563eb',
  post: '#ea580c',
  dp: '#059669',
  group: '#dc2626',
  conf: '#7c3aed',
};

function getRunColor(runIndex: number): string {
  return RUN_COLORS[runIndex % RUN_COLORS.length];
}

export function PerformanceDashboard() {
  const { runs, baselineId, visibility, filters } = useReports();

  const visibleRuns = useMemo(
    () => runs.filter((r) => visibility[r.id] !== false),
    [runs, visibility]
  );

  const runData = useMemo(() => {
    return visibleRuns.map((run) => {
      const filtered = filterRecords(run.records, filters);
      const agg = computeRunAggregates(filtered);
      return { run, filtered, agg };
    });
  }, [visibleRuns, filters]);

  const baselineAgg = useMemo(() => {
    if (!baselineId) return null;
    const d = runData.find((r) => r.run.id === baselineId);
    return d?.agg ?? null;
  }, [runData, baselineId]);

  if (runs.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        Load one or more run files to compare performance.
      </div>
    );
  }

  const hasFilteredData = runData.some((d) => d.filtered.length > 0);
  if (!hasFilteredData) {
    return (
      <div className="text-center py-12 text-gray-500">
        No records match the current filters. Adjust filters or load more runs.
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <Section1GlobalPerformance
        runData={runData}
        baselineId={baselineId}
        baselineAgg={baselineAgg}
      />
      <Section2TimeBreakdown runData={runData} />
      <Section3ScalingBehavior runData={runData} />
      <Section4MemoryVsLatency runData={runData} />
    </div>
  );
}

// --- Section 1: Global Performance ---

function Section1GlobalPerformance({
  runData,
  baselineId,
  baselineAgg,
}: {
  runData: { run: LoadedRun; filtered: import('../types/report').RustPerfRecord[]; agg: import('../lib/perfMetrics').RunAggregates }[];
  baselineId: string | null;
  baselineAgg: import('../lib/perfMetrics').RunAggregates | null;
}) {
  const ecdfTraces = useMemo(() => {
    return runData.map(({ run, filtered }, i) => {
      const totalMs = filtered.map((r) => r.total_ms);
      const { x, y } = ecdf(totalMs);
      return {
        x,
        y,
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: run.name,
        line: { color: getRunColor(i), width: 2 },
      };
    });
  }, [runData]);

  return (
    <section className="bg-white rounded shadow p-6">
      <h2 className="text-xl font-bold mb-4">1. Global Performance</h2>
      <p className="text-sm text-gray-600 mb-4">Which implementation is faster overall?</p>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4 mb-6">
        {runData.map(({ run, agg }, i) => {
          const isBaseline = run.id === baselineId;
          const speedup = baselineAgg && !isBaseline ? speedupVsBaseline(agg.medianTotalMs, baselineAgg.medianTotalMs) : null;
          return (
            <div
              key={run.id}
              className="border rounded-lg p-4"
              style={{ borderLeftWidth: 4, borderLeftColor: getRunColor(i) }}
            >
              <div className="font-semibold text-gray-800 truncate" title={run.name}>
                {run.name}
              </div>
              {isBaseline && (
                <div className="text-xs text-amber-600 font-medium mt-1">Baseline</div>
              )}
              <div className="mt-2 space-y-1 text-sm">
                <div>Median latency <MetricTooltip tip="Middle value of total processing time across utterances. Use it to compare typical run time between implementations." /><strong className="ml-1">: {agg.medianTotalMs.toFixed(1)} ms</strong></div>
                <div>P90 latency <MetricTooltip tip="90th percentile: 90% of utterances finish within this time. Shows tail latency and stability." /><strong className="ml-1">: {agg.p90TotalMs.toFixed(1)} ms</strong></div>
                <div>Median RTF <MetricTooltip tip="Real-time factor = total_ms ÷ duration_ms. Below 1 means faster than real time; lower is better." /><strong className="ml-1">: {agg.medianRtf.toFixed(4)}</strong></div>
                <div>Median cost/frame <MetricTooltip tip="Total time divided by number of frames. Measures per-frame overhead; helps compare algorithmic efficiency." /><strong className="ml-1">: {agg.medianCostPerFrame.toFixed(3)} ms</strong></div>
                {speedup != null && (
                  <div className="text-green-600">Speedup vs baseline <MetricTooltip tip="How many times faster this run is than the baseline (baseline median ÷ this run median)." /><strong className="ml-1">: {speedup.toFixed(2)}x</strong></div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <div>
        <h3 className="font-semibold mb-2">Latency distribution (ECDF) <MetricTooltip tip="Empirical cumulative distribution: at each x (latency), y is the proportion of utterances that finished in ≤ x ms. Steeper rise = more consistent; flat tail = outliers." /></h3>
        <ChartWithFullscreen
          title="Latency distribution (ECDF)"
          data={ecdfTraces}
          layout={{
            xaxis: { title: 'Total latency (ms)' },
            yaxis: { title: 'Proportion of utterances', range: [0, 1.02] },
            margin: { l: 50, r: 20, t: 20, b: 50 },
            height: 360,
            showlegend: true,
            legend: { x: 1, xanchor: 'right' },
          }}
          config={{ displayModeBar: false }}
          className="w-full"
        />
      </div>
    </section>
  );
}

// --- Section 2: Time Breakdown ---

function Section2TimeBreakdown({
  runData,
}: {
  runData: { run: LoadedRun; agg: import('../lib/perfMetrics').RunAggregates }[];
}) {
  const runNames = runData.map((d) => d.run.name);

  const absoluteTraces = useMemo(() => {
    return STAGE_KEYS.map((stage: StageKey) => ({
      x: runNames,
      y: runData.map((d) => d.agg.medianStageMs[stage]),
      name: stage,
      type: 'bar' as const,
      marker: { color: STAGE_COLORS[stage] },
    }));
  }, [runData, runNames]);

  const normalizedTraces = useMemo(() => {
    const sums = runData.map((d) => {
      const t =
        d.agg.medianStageMs.forward +
        d.agg.medianStageMs.post +
        d.agg.medianStageMs.dp +
        d.agg.medianStageMs.group +
        d.agg.medianStageMs.conf;
      return t || 1;
    });
    return STAGE_KEYS.map((stage: StageKey) => ({
      x: runNames,
      y: runData.map((d, i) => (sums[i] ? (d.agg.medianStageMs[stage] / sums[i]) * 100 : 0)),
      name: stage,
      type: 'bar' as const,
      marker: { color: STAGE_COLORS[stage] },
    }));
  }, [runData, runNames]);

  return (
    <section className="bg-white rounded shadow p-6">
      <h2 className="text-xl font-bold mb-4">2. Time Breakdown</h2>
      <p className="text-sm text-gray-600 mb-4">Where is the time spent? (median across filtered records)</p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h3 className="font-semibold mb-2">Absolute (ms) <MetricTooltip tip="Median time spent in each pipeline stage (forward, post, dp, group, conf) per run. Shows where wall-clock time goes in milliseconds." /></h3>
          <ChartWithFullscreen
            title="Time breakdown (absolute ms)"
            data={absoluteTraces}
            layout={{
              barmode: 'stack',
              xaxis: { title: 'Run' },
              yaxis: { title: 'Time (ms)' },
              margin: { l: 50, r: 20, t: 20, b: 60 },
              height: 320,
              showlegend: true,
            }}
            config={{ displayModeBar: false }}
            className="w-full"
          />
        </div>
        <div>
          <h3 className="font-semibold mb-2">Normalized (% of total) <MetricTooltip tip="Each bar sums to 100%. Shows which stage dominates regardless of total speed. Use to see if slowdown is in inference (forward) vs alignment (dp/group/conf)." /></h3>
          <ChartWithFullscreen
            title="Time breakdown (normalized %)"
            data={normalizedTraces}
            layout={{
              barmode: 'stack',
              xaxis: { title: 'Run' },
              yaxis: { title: '% of total time', range: [0, 100] },
              margin: { l: 50, r: 20, t: 20, b: 60 },
              height: 320,
              showlegend: true,
            }}
            config={{ displayModeBar: false }}
            className="w-full"
          />
        </div>
      </div>
    </section>
  );
}

const SCALING_BINS = 15;

// --- Section 3: Scaling Behavior ---

function Section3ScalingBehavior({
  runData,
}: {
  runData: { run: LoadedRun; filtered: import('../types/report').RustPerfRecord[]; agg: import('../lib/perfMetrics').RunAggregates }[];
}) {
  const [rtfViewMode, setRtfViewMode] = useState<'scatter' | 'binned'>('scatter');
  const [costViewMode, setCostViewMode] = useState<'scatter' | 'binned'>('scatter');

  const rtfTraces = useMemo(() => {
    return runData.map(({ run, filtered }, i) => ({
      x: filtered.map((r) => r.duration_ms),
      y: filtered.map((r) => rtf(r)),
      type: 'scatter' as const,
      mode: 'markers' as const,
      name: run.name,
      marker: { color: getRunColor(i), size: 6, opacity: 0.7 },
    }));
  }, [runData]);

  const rtfBinnedTraces = useMemo(() => {
    const allX = runData.flatMap(({ filtered }) => filtered.map((r) => r.duration_ms));
    const xMin = allX.length ? Math.min(...allX) : 0;
    const xMax = allX.length ? Math.max(...allX) : 1;
    return runData.map(({ run, filtered }, i) => {
      const x = filtered.map((r) => r.duration_ms);
      const y = filtered.map((r) => rtf(r));
      const { binCenters, medianYs } = binAndMedian(x, y, SCALING_BINS, xMin, xMax);
      return {
        x: binCenters,
        y: medianYs,
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        name: run.name,
        marker: { color: getRunColor(i), size: 8 },
        line: { color: getRunColor(i), width: 2 },
      };
    });
  }, [runData]);

  const costTraces = useMemo(() => {
    return runData.map(({ run, filtered }, i) => ({
      x: filtered.map((r) => r.num_frames_t),
      y: filtered.map((r) => costPerFrame(r)),
      type: 'scatter' as const,
      mode: 'markers' as const,
      name: run.name,
      marker: { color: getRunColor(i), size: 6, opacity: 0.7 },
    }));
  }, [runData]);

  const costBinnedTraces = useMemo(() => {
    const allX = runData.flatMap(({ filtered }) => filtered.map((r) => r.num_frames_t));
    const xMin = allX.length ? Math.min(...allX) : 0;
    const xMax = allX.length ? Math.max(...allX) : 1;
    return runData.map(({ run, filtered }, i) => {
      const x = filtered.map((r) => r.num_frames_t);
      const y = filtered.map((r) => costPerFrame(r));
      const { binCenters, medianYs } = binAndMedian(x, y, SCALING_BINS, xMin, xMax);
      return {
        x: binCenters,
        y: medianYs,
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        name: run.name,
        marker: { color: getRunColor(i), size: 8 },
        line: { color: getRunColor(i), width: 2 },
      };
    });
  }, [runData]);

  return (
    <section className="bg-white rounded shadow p-6">
      <h2 className="text-xl font-bold mb-4">3. Scaling Behavior</h2>
      <p className="text-sm text-gray-600 mb-4">Does performance scale with duration and frame count?</p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <div className="flex items-center justify-between gap-2 mb-2">
            <h3 className="font-semibold">RTF vs duration <MetricTooltip tip="Real-time factor vs audio length. Flat line = linear scaling; downward slope = longer clips are more efficient (e.g. batching); upward = overhead dominates on short clips." /></h3>
            <ChartViewToggle value={rtfViewMode} onChange={setRtfViewMode} />
          </div>
          <ChartWithFullscreen
            title="RTF vs duration"
            data={rtfViewMode === 'binned' ? rtfBinnedTraces : rtfTraces}
            layout={{
              xaxis: { title: 'Duration (ms)' },
              yaxis: { title: 'RTF' },
              margin: { l: 50, r: 20, t: 20, b: 50 },
              height: 320,
              showlegend: true,
            }}
            config={{ displayModeBar: false }}
            className="w-full"
          />
        </div>
        <div>
          <div className="flex items-center justify-between gap-2 mb-2">
            <h3 className="font-semibold">Cost per frame vs frame count <MetricTooltip tip="Time per frame vs number of frames. Flat line = linear complexity; upward slope = super-linear (e.g. alignment cost); downward = fixed overhead dominates on small inputs." /></h3>
            <ChartViewToggle value={costViewMode} onChange={setCostViewMode} />
          </div>
          <ChartWithFullscreen
            title="Cost per frame vs frame count"
            data={costViewMode === 'binned' ? costBinnedTraces : costTraces}
            layout={{
              xaxis: { title: 'Frame count' },
              yaxis: { title: 'Cost per frame (ms)' },
              margin: { l: 50, r: 20, t: 20, b: 50 },
              height: 320,
              showlegend: true,
            }}
            config={{ displayModeBar: false }}
            className="w-full"
          />
        </div>
      </div>
    </section>
  );
}

function ChartViewToggle({
  value,
  onChange,
}: {
  value: 'scatter' | 'binned';
  onChange: (v: 'scatter' | 'binned') => void;
}) {
  return (
    <div className="flex rounded border border-gray-200 p-0.5 bg-gray-50 shrink-0" role="group" aria-label="Chart view">
      <button
        type="button"
        onClick={() => onChange('scatter')}
        className={`px-2 py-1 text-xs font-medium rounded transition ${value === 'scatter' ? 'bg-white shadow text-gray-800' : 'text-gray-600 hover:text-gray-800'}`}
      >
        Dot cloud
      </button>
      <button
        type="button"
        onClick={() => onChange('binned')}
        className={`px-2 py-1 text-xs font-medium rounded transition ${value === 'binned' ? 'bg-white shadow text-gray-800' : 'text-gray-600 hover:text-gray-800'}`}
      >
        Binned + median
      </button>
    </div>
  );
}

// --- Section 4: Memory vs Latency ---

function Section4MemoryVsLatency({
  runData,
}: {
  runData: { run: LoadedRun; filtered: import('../types/report').RustPerfRecord[]; agg: import('../lib/perfMetrics').RunAggregates }[];
}) {
  const anyHasMemory = useMemo(() => runData.some((d) => d.agg.hasMemory), [runData]);

  if (!anyHasMemory) {
    return (
      <section className="bg-white rounded shadow p-6">
        <h2 className="text-xl font-bold mb-4">4. Memory vs Latency</h2>
        <p className="text-sm text-gray-500">No memory data in loaded runs. Load runs with memory profiling to see this section.</p>
      </section>
    );
  }

  const memoryScatterTraces = useMemo(() => {
    return runData
      .filter((d) => d.agg.hasMemory)
      .map(({ run, filtered }, i) => {
        const points = filtered
          .map((r) => {
            const p = peakMemory(r);
            return p ? { total_ms: r.total_ms, gpu: p.peak_gpu_alloc } : null;
          })
          .filter((p): p is { total_ms: number; gpu: number } => p != null);
        return {
          x: points.map((p) => p.total_ms),
          y: points.map((p) => p.gpu / 1024 / 1024), // MB
          type: 'scatter' as const,
          mode: 'markers' as const,
          name: run.name,
          marker: { color: getRunColor(i), size: 6, opacity: 0.7 },
        };
      });
  }, [runData]);

  return (
    <section className="bg-white rounded shadow p-6">
      <h2 className="text-xl font-bold mb-4">4. Memory vs Latency</h2>
      <p className="text-sm text-gray-600 mb-4">Is speed obtained by using more memory?</p>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mb-6">
        {runData
          .filter((d) => d.agg.hasMemory)
          .map(({ run, agg }) => (
            <div key={run.id} className="border rounded p-3">
              <div className="font-semibold text-gray-800 truncate text-sm" title={run.name}>
                {run.name}
              </div>
              <div className="text-sm mt-1">
                Median peak GPU <MetricTooltip tip="Median across utterances of the maximum GPU memory allocated during each run. Use to compare memory footprint between implementations." />: <strong>{(agg.medianPeakGpuAlloc / 1024 / 1024).toFixed(2)} MB</strong>
              </div>
            </div>
          ))}
      </div>

      <div>
        <h3 className="font-semibold mb-2">Peak GPU vs total latency (per record) <MetricTooltip tip="Each point is one utterance. Shows whether faster runs use more or less GPU memory. Helps spot memory–speed tradeoffs." /></h3>
        <ChartWithFullscreen
          title="Peak GPU vs total latency"
          data={memoryScatterTraces}
          layout={{
            xaxis: { title: 'Total latency (ms)' },
            yaxis: { title: 'Peak GPU alloc (MB)' },
            margin: { l: 50, r: 20, t: 20, b: 50 },
            height: 360,
            showlegend: true,
          }}
          config={{ displayModeBar: false }}
          className="w-full"
        />
      </div>
    </section>
  );
}