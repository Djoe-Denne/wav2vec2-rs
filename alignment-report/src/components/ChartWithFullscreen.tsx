import { useState } from 'react';
import Plot from 'react-plotly.js';
import { Expand, X } from 'lucide-react';

type PlotlyData = React.ComponentProps<typeof Plot>['data'];
type PlotlyLayout = React.ComponentProps<typeof Plot>['layout'];
type PlotlyConfig = React.ComponentProps<typeof Plot>['config'];

interface ChartWithFullscreenProps {
  data: PlotlyData;
  layout: PlotlyLayout;
  config?: PlotlyConfig;
  className?: string;
  /** Shown in the fullscreen modal header */
  title?: string;
  /** If true, fullscreen chart allows horizontal-only zoom (y axis fixed). Default true. */
  horizontalZoomOnly?: boolean;
}

export function ChartWithFullscreen({
  data,
  layout,
  config = { displayModeBar: false },
  className,
  title,
  horizontalZoomOnly = true,
}: ChartWithFullscreenProps) {
  const [fullscreenOpen, setFullscreenOpen] = useState(false);

  const fullscreenLayout: PlotlyLayout = horizontalZoomOnly && layout && typeof layout === 'object'
    ? {
        ...layout,
        height: undefined,
        autosize: true,
        margin: { ...(typeof layout.margin === 'object' && layout.margin ? layout.margin : {}), l: 60, r: 40, t: 40, b: 60 },
        xaxis: typeof layout.xaxis === 'object' && layout.xaxis
          ? { ...layout.xaxis, fixedrange: false }
          : { fixedrange: false },
        yaxis: typeof layout.yaxis === 'object' && layout.yaxis
          ? { ...layout.yaxis, fixedrange: true }
          : { fixedrange: true },
      }
    : { ...layout, height: undefined, autosize: true };

  return (
    <>
      <div className="relative">
        <button
          type="button"
          onClick={() => setFullscreenOpen(true)}
          className="absolute top-1 right-1 z-10 p-1.5 rounded bg-white/90 hover:bg-gray-100 border border-gray-200 shadow-sm text-gray-600 hover:text-gray-800 transition"
          title="Open chart in fullscreen"
          aria-label="Open chart in fullscreen"
        >
          <Expand className="w-4 h-4" />
        </button>
        <Plot
          data={data}
          layout={layout}
          config={config}
          className={className}
        />
      </div>

      {fullscreenOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          role="dialog"
          aria-modal="true"
          aria-label={title ?? 'Chart fullscreen'}
          onClick={() => setFullscreenOpen(false)}
        >
          <div
            className="bg-white rounded-lg shadow-xl flex flex-col w-full max-w-[95vw] h-[90vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-4 py-2 border-b shrink-0">
              <h3 className="font-semibold text-gray-800">{title ?? 'Chart'}</h3>
              <button
                type="button"
                onClick={() => setFullscreenOpen(false)}
                className="p-2 rounded hover:bg-gray-100 text-gray-600 hover:text-gray-800"
                aria-label="Close fullscreen"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 min-h-0 p-4">
              <Plot
                data={data}
                layout={fullscreenLayout}
                config={{
                  ...config,
                  displayModeBar: true,
                  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                  scrollZoom: true,
                }}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler
              />
            </div>
            {horizontalZoomOnly && (
              <p className="text-xs text-gray-500 px-4 pb-2">
                Drag to zoom on the horizontal axis; use the toolbar to reset or pan.
              </p>
            )}
          </div>
        </div>
      )}
    </>
  );
}
