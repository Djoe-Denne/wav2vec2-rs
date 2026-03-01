import type { ParsedTextGrid, WordInterval } from '../../shared/types';
import { normalizeWordText } from '../../shared/utils';

interface IntervalTier {
  name: string;
  class: string;
  intervals: { xmin: number; xmax: number; text: string }[];
}

/**
 * Parse Praat TextGrid ooTextFile format. Extracts interval tiers and their intervals.
 */
function parseTiers(content: string): IntervalTier[] {
  const tiers: IntervalTier[] = [];
  const lines = content.split(/\r?\n/);
  let i = 0;

  while (i < lines.length) {
    const itemMatch = lines[i].match(/^\s*item\s*\[\d+\]\s*:/);
    if (!itemMatch) {
      i++;
      continue;
    }
    i++;
    let tierName = '';
    let tierClass = '';
    let intervals: { xmin: number; xmax: number; text: string }[] = [];
    let intervalsSize = 0;

    while (i < lines.length) {
      const line = lines[i];
      if (line.match(/^\s*item\s*\[\d+\]\s*:/)) break;

      const nameMatch = line.match(/^\s*name\s*=\s*"((?:[^"\\]|\\.)*)"/);
      if (nameMatch) tierName = nameMatch[1].replace(/\\"/g, '"');

      const classMatch = line.match(/^\s*class\s*=\s*"([^"]*)"/);
      if (classMatch) tierClass = classMatch[1];

      const sizeMatch = line.match(/^\s*intervals:\s*size\s*=\s*(\d+)/);
      if (sizeMatch) {
        intervalsSize = parseInt(sizeMatch[1], 10);
        i++;
        while (intervals.length < intervalsSize && i < lines.length) {
          if (lines[i].match(/^\s*intervals\s*\[\d+\]\s*:/)) {
            i++;
            let xmin = NaN;
            let xmax = NaN;
            let text = '';
            while (i < lines.length && lines[i].match(/^\s*(xmin|xmax|text)\s*=/)) {
              const xminM = lines[i].match(/xmin\s*=\s*([\d.eE+-]+)/);
              const xmaxM = lines[i].match(/xmax\s*=\s*([\d.eE+-]+)/);
              const textM = lines[i].match(/text\s*=\s*"((?:[^"\\]|\\.)*)"/);
              if (xminM) xmin = parseFloat(xminM[1]);
              if (xmaxM) xmax = parseFloat(xmaxM[1]);
              if (textM) text = textM[1].replace(/\\"/g, '"');
              i++;
            }
            if (!Number.isNaN(xmin) && !Number.isNaN(xmax)) {
              intervals.push({ xmin, xmax, text });
            }
            continue;
          }
          i++;
        }
        break;
      }
      i++;
    }

    if (tierClass === 'IntervalTier' && tierName) {
      tiers.push({ name: tierName, class: tierClass, intervals });
    }
  }

  return tiers;
}

/**
 * Choose word tier: 1) "words", 2) "word", 3) first interval tier with non-empty labels.
 */
function selectWordTier(tiers: IntervalTier[]): IntervalTier | null {
  const byName = tiers.find((t) => t.name === 'words');
  if (byName) return byName;
  const wordTier = tiers.find((t) => t.name === 'word');
  if (wordTier) return wordTier;
  return tiers.find((t) => t.intervals.some((i) => (i.text || '').trim() !== '')) ?? null;
}

/**
 * Parse TextGrid content and return word intervals (non-empty only) with ms times and normalized text.
 */
export function parseTextGrid(content: string): ParsedTextGrid {
  try {
    const tiers = parseTiers(content);
    const wordTier = selectWordTier(tiers);
    if (!wordTier) {
      return { words: [], error: 'No word tier found (looked for "words", "word", or first tier with labels)' };
    }

    const words: WordInterval[] = [];
    for (const iv of wordTier.intervals) {
      const text = (iv.text || '').trim();
      if (text === '') continue;
      const startMs = iv.xmin * 1000;
      const endMs = iv.xmax * 1000;
      const midMs = (startMs + endMs) / 2;
      words.push({
        text,
        normalizedText: normalizeWordText(text),
        startMs,
        endMs,
        midMs,
      });
    }

    const durationSec =
      wordTier.intervals.length > 0
        ? wordTier.intervals[wordTier.intervals.length - 1].xmax
        : undefined;
    return { words, durationSec };
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    return { words: [], error: `Parse error: ${message}` };
  }
}
