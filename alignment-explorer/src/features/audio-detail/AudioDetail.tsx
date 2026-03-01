import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useParams, Link } from 'react-router-dom';
import { useCorpus } from '../../context/CorpusContext';
import { useWaveSurfer } from './useWaveSurfer';

function formatMs(ms: number): string {
  return `${(ms / 1000).toFixed(2)}s`;
}

export function AudioDetail() {
  const { entryId } = useParams<{ entryId: string }>();
  const { corpus } = useCorpus();
  const [focusedWordIndex, setFocusedWordIndex] = useState<number | null>(null);
  const [waveformFullscreen, setWaveformFullscreen] = useState(false);
  const previousAudioUrlRef = useRef<string | null>(null);
  const containerParentRef = useRef<HTMLDivElement>(null);
  const waveformSpectrogramContainerRef = useRef<HTMLDivElement>(null);
  const fullscreenContentRef = useRef<HTMLDivElement>(null);

  const entry = useMemo(
    () => corpus?.entries.find((e) => e.id === entryId) ?? null,
    [corpus, entryId]
  );

  const audioUrl = useMemo(() => {
    if (!entry?.audioFile) {
      if (previousAudioUrlRef.current) {
        URL.revokeObjectURL(previousAudioUrlRef.current);
        previousAudioUrlRef.current = null;
      }
      return null;
    }
    if (previousAudioUrlRef.current) {
      URL.revokeObjectURL(previousAudioUrlRef.current);
    }
    const url = URL.createObjectURL(entry.audioFile);
    previousAudioUrlRef.current = url;
    return url;
  }, [entry?.audioFile]);

  const comparisonsForEntry = useMemo(
    () => corpus?.comparisons.filter((c) => c.entryId === entryId) ?? [],
    [corpus, entryId]
  );

  const {
    waveformRef,
    spectrogramRef,
    wavesurferRef,
    isReady,
    error: waveSurferError,
  } = useWaveSurfer(
    audioUrl ?? null,
    entry,
    comparisonsForEntry,
    corpus?.parsedVariants ?? {},
    focusedWordIndex
  );

  useEffect(() => {
    const wrapper = waveformSpectrogramContainerRef.current;
    const parent = containerParentRef.current;
    if (!wrapper || !parent) return;
    if (waveformFullscreen) {
      const id = requestAnimationFrame(() => {
        const fullscreen = fullscreenContentRef.current;
        if (fullscreen && wrapper.parentNode !== fullscreen) {
          fullscreen.appendChild(wrapper);
        }
      });
      return () => cancelAnimationFrame(id);
    } else {
      if (wrapper.parentNode !== parent) parent.appendChild(wrapper);
    }
  }, [waveformFullscreen]);

  if (!corpus) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <p className="text-gray-600 mb-4">No corpus loaded.</p>
          <Link to="/" className="text-blue-600 hover:underline">
            Import a folder
          </Link>
        </div>
      </div>
    );
  }

  if (!entry) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <p className="text-gray-600 mb-4">Sample not found.</p>
          <Link to="/dashboard" className="text-blue-600 hover:underline">
            Back to dashboard
          </Link>
        </div>
      </div>
    );
  }

  const baseline = entry.baseline;
  const hasWords = baseline && baseline.words.length > 0;

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between">
        <Link to="/dashboard" className="text-gray-600 hover:text-gray-900">
          ← Dashboard
        </Link>
        <h1 className="text-lg font-semibold text-gray-900 truncate max-w-md" title={entry.id}>
          {entry.id}
        </h1>
        <span className="text-sm text-gray-500">{entry.subset}</span>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6 space-y-6">
        {entry.errors.length > 0 && (
          <div className="rounded-lg bg-amber-50 border border-amber-200 p-3 text-sm text-amber-800">
            {entry.errors.map((err, i) => (
              <div key={i}>{err}</div>
            ))}
          </div>
        )}

        {audioUrl && (
          <div>
            <audio
              src={audioUrl}
              controls
              className="w-full"
              ref={(el) => {
                if (el && wavesurferRef.current) {
                  const media = wavesurferRef.current.getMediaElement();
                  if (media && media !== el) {
                    el.src = media.src;
                  }
                }
              }}
            />
          </div>
        )}

        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden relative">
          <div className="px-3 py-2 border-b border-gray-100 flex items-center justify-between">
            <p className="text-sm text-gray-500">Waveform & spectrogram</p>
            <button
              type="button"
              onClick={() => setWaveformFullscreen(true)}
              className="text-sm text-gray-600 hover:text-gray-900 px-2 py-1 rounded hover:bg-gray-100"
              title="Open fullscreen"
            >
              ⛶ Fullscreen
            </button>
          </div>
          <div ref={containerParentRef} className="relative">
            <div ref={waveformSpectrogramContainerRef}>
              <div ref={waveformRef} className="min-h-[120px]" />
              <div ref={spectrogramRef} className="min-h-[128px]" />
            </div>
          </div>
          {!isReady && !waveSurferError && (
            <p className="px-3 py-4 text-sm text-gray-400 absolute top-12 left-0">Loading…</p>
          )}
          {waveSurferError && (
            <p className="px-3 py-4 text-sm text-red-600">{waveSurferError}</p>
          )}
        </div>

        {waveformFullscreen &&
          createPortal(
            <div
              className="fixed inset-0 z-50 bg-white flex flex-col"
              role="dialog"
              aria-modal="true"
              aria-label="Waveform and spectrogram fullscreen"
            >
              <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200">
                <span className="text-sm font-medium text-gray-700 truncate">{entry.id}</span>
                <button
                  type="button"
                  onClick={() => setWaveformFullscreen(false)}
                  className="text-sm text-gray-600 hover:text-gray-900 px-3 py-1.5 rounded hover:bg-gray-100"
                >
                  ✕ Close
                </button>
              </div>
              <div className="flex-1 min-h-0 overflow-auto p-4">
                <div ref={fullscreenContentRef} className="max-w-6xl mx-auto" />
              </div>
            </div>,
            document.body
          )}

        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setFocusedWordIndex(null)}
              className={`px-3 py-1.5 rounded text-sm font-medium ${focusedWordIndex === null ? 'bg-gray-900 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}
            >
              Global timeline
            </button>
            <span className="text-sm text-gray-500">
              {focusedWordIndex !== null ? `Focused on word #${focusedWordIndex + 1}` : 'All words'}
            </span>
          </div>
          {entry.variants.length > 0 && (
            <p className="text-xs text-gray-400" title="Baseline = reference; colored = implementations">
              Markers: baseline (dark) · implementations (colors)
            </p>
          )}
        </div>

        {hasWords ? (
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <p className="px-3 py-2 text-sm text-gray-500 border-b border-gray-100">
              Word table (click row to focus)
            </p>
            <div className="overflow-x-auto max-h-96 overflow-y-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200 bg-gray-50">
                    <th className="text-left px-3 py-2 font-medium text-gray-700">#</th>
                    <th className="text-left px-3 py-2 font-medium text-gray-700">Word</th>
                    <th className="text-left px-3 py-2 font-medium text-gray-700">Start</th>
                    <th className="text-left px-3 py-2 font-medium text-gray-700">End</th>
                    <th className="text-left px-3 py-2 font-medium text-gray-700">Mid</th>
                  </tr>
                </thead>
                <tbody>
                  {baseline!.words.map((w, i) => (
                    <tr
                      key={i}
                      onClick={() => setFocusedWordIndex(i)}
                      className={`border-b border-gray-100 cursor-pointer hover:bg-gray-50 ${focusedWordIndex === i ? 'bg-blue-50' : ''}`}
                    >
                      <td className="px-3 py-2">{i + 1}</td>
                      <td className="px-3 py-2 font-medium">{w.text}</td>
                      <td className="px-3 py-2 text-gray-600">{formatMs(w.startMs)}</td>
                      <td className="px-3 py-2 text-gray-600">{formatMs(w.endMs)}</td>
                      <td className="px-3 py-2 text-gray-600">{formatMs(w.midMs)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="bg-white rounded-lg border border-gray-200 p-4 text-sm text-gray-500">
            No word-level data for this sample.
          </div>
        )}
      </main>
    </div>
  );
}
