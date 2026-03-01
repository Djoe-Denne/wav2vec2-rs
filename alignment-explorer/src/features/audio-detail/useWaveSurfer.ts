import { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import SpectrogramPlugin from 'wavesurfer.js/dist/plugins/spectrogram.esm.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';
import type { AudioEntry, ParsedTextGrid, SampleComparison, WordInterval } from '../../shared/types';

const fmtTime = (ms: number) => `${(ms / 1000).toFixed(2)}s`;

/** Tooltip-only (no visible text): for global timeline to avoid overlap */
function makeTooltipOnlyRegion(word: WordInterval, implLabel?: string): HTMLElement {
  const span = document.createElement('span');
  span.style.cssText = 'display:block;width:100%;height:100%;';
  span.title = implLabel
    ? `${implLabel}: ${word.text} · ${fmtTime(word.startMs)} – ${fmtTime(word.endMs)}`
    : `${word.text} · ${fmtTime(word.startMs)} – ${fmtTime(word.endMs)}`;
  return span;
}

/** Multi-line label for focused word: baseline + all implementations (line breaks, no overlap) */
function makeFocusedWordLabel(
  baselineWord: WordInterval,
  variants: { suffix: string; word: WordInterval }[]
): HTMLElement {
  const div = document.createElement('div');
  div.style.cssText = 'font-size:11px;line-height:1.35;padding:2px 4px;white-space:pre-wrap;word-break:break-all;';
  const lines: string[] = [`${baselineWord.text}  (baseline)  ${fmtTime(baselineWord.startMs)}–${fmtTime(baselineWord.endMs)}`];
  for (const v of variants) {
    lines.push(`${v.word.text}  (${v.suffix})  ${fmtTime(v.word.startMs)}–${fmtTime(v.word.endMs)}`);
  }
  div.textContent = lines.join('\n');
  div.title = lines.join('\n');
  return div;
}

const IMPL_COLORS = [
  'rgba(59, 130, 246, 0.35)',
  'rgba(234, 179, 8, 0.35)',
  'rgba(34, 197, 94, 0.35)',
  'rgba(168, 85, 247, 0.35)',
  'rgba(236, 72, 153, 0.35)',
];
const BASELINE_COLOR = 'rgba(0, 0, 0, 0.2)';

function msToSec(ms: number): number {
  return ms / 1000;
}

export function useWaveSurfer(
  audioUrl: string | null,
  entry: AudioEntry | null,
  _comparisons: SampleComparison[],
  parsedVariants: Record<string, Record<string, ParsedTextGrid>>,
  focusedWordIndex: number | null
) {
  const waveformRef = useRef<HTMLDivElement>(null);
  const spectrogramRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<ReturnType<RegionsPlugin['addRegion']>[]>([]);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!audioUrl || !entry || !waveformRef.current || !spectrogramRef.current) return;

    const regionsPlugin = RegionsPlugin.create();
    const spectrogramPlugin = SpectrogramPlugin.create({
      container: spectrogramRef.current,
      height: 128,
      labels: true,
    });

    const ws = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#94a3b8',
      progressColor: '#64748b',
      url: audioUrl,
      height: 120,
      plugins: [regionsPlugin, spectrogramPlugin],
    });

    wavesurferRef.current = ws;

    const onReady = () => {
      regionsPlugin.clearRegions();
      regionsRef.current = [];

      const baseline = entry.baseline;
      const padding = 0.1;

      if (focusedWordIndex !== null && baseline && focusedWordIndex >= 0 && focusedWordIndex < baseline.words.length) {
        const w = baseline.words[focusedWordIndex];
        ws.zoom(ws.getWidth() / (msToSec(w.endMs - w.startMs) + 2 * padding));
        ws.setTime(Math.max(0, msToSec(w.startMs) - padding));
      }

      if (baseline?.words.length) {
        if (focusedWordIndex !== null && focusedWordIndex >= 0 && focusedWordIndex < baseline.words.length) {
          const w = baseline.words[focusedWordIndex];
          const entryVariants = parsedVariants[entry.id];
          const variantInfos = entryVariants
            ? Object.entries(entryVariants)
                .map(([suffix, parsed]) => {
                  const vw = parsed.words[focusedWordIndex];
                  return vw ? { suffix, word: vw } : null;
                })
                .filter((x): x is { suffix: string; word: WordInterval } => x != null)
            : [];
          const r = regionsPlugin.addRegion({
            start: msToSec(w.startMs),
            end: msToSec(w.endMs),
            color: BASELINE_COLOR,
            drag: false,
            resize: false,
            content: makeFocusedWordLabel(w, variantInfos),
          });
          regionsRef.current.push(r);
          if (entryVariants) {
            Object.entries(entryVariants).forEach(([suffix, parsed], i) => {
              const vw = parsed.words[focusedWordIndex];
              if (vw) {
                const rr = regionsPlugin.addRegion({
                  start: msToSec(vw.startMs),
                  end: msToSec(vw.endMs),
                  color: IMPL_COLORS[i % IMPL_COLORS.length],
                  drag: false,
                  resize: false,
                  content: makeTooltipOnlyRegion(vw, suffix),
                });
                regionsRef.current.push(rr);
              }
            });
          }
        } else {
          baseline.words.forEach((w) => {
            const r = regionsPlugin.addRegion({
              start: msToSec(w.startMs),
              end: msToSec(w.endMs),
              color: BASELINE_COLOR,
              drag: false,
              resize: false,
              content: makeTooltipOnlyRegion(w),
            });
            regionsRef.current.push(r);
          });
          const entryVariants = parsedVariants[entry.id];
          if (entryVariants) {
            Object.entries(entryVariants).forEach(([suffix, parsed], implIdx) => {
              parsed.words.forEach((vw) => {
                const rr = regionsPlugin.addRegion({
                  start: msToSec(vw.startMs),
                  end: msToSec(vw.endMs),
                  color: IMPL_COLORS[implIdx % IMPL_COLORS.length],
                  drag: false,
                  resize: false,
                  content: makeTooltipOnlyRegion(vw, suffix),
                });
                regionsRef.current.push(rr);
              });
            });
          }
        }
      }

      setIsReady(true);
      setError(null);
    };

    const onError = (e: Error) => {
      setError(e.message);
      setIsReady(false);
    };

    ws.on('ready', onReady);
    ws.on('error', onError);

    return () => {
      ws.un('ready', onReady);
      ws.un('error', onError);
      ws.destroy();
      wavesurferRef.current = null;
      regionsRef.current = [];
      setIsReady(false);
    };
  }, [audioUrl, entry?.id, focusedWordIndex, parsedVariants]);

  return {
    waveformRef,
    spectrogramRef,
    wavesurferRef,
    isReady,
    error,
  };
}
