import type { RawFileRef, AudioEntry, VariantRef, Subset } from '../../shared/types';
import { entryIdFromPath } from '../../shared/utils';

function inferSubset(relativePath: string): Subset {
  const normalized = relativePath.replace(/\\/g, '/');
  if (normalized.includes('test-clean')) return 'test-clean';
  if (normalized.includes('test-other')) return 'test-other';
  return 'unknown';
}

/**
 * Group files from a directory picker into AudioEntry[].
 * One entry per .flac; baseline = stem.TextGrid, variants = stem_suffix.TextGrid.
 * Does not read file contents.
 */
export function groupFilesIntoEntries(files: RawFileRef[]): AudioEntry[] {
  const byDir = new Map<string, RawFileRef[]>();
  for (const ref of files) {
    const path = ref.relativePath.replace(/\\/g, '/');
    const dir = path.includes('/') ? path.slice(0, path.lastIndexOf('/') + 1) : '';
    if (!byDir.has(dir)) byDir.set(dir, []);
    byDir.get(dir)!.push(ref);
  }

  const entries: AudioEntry[] = [];

  for (const [, dirFiles] of byDir) {
    const flacs = dirFiles.filter(
      (r) => r.file.name.toLowerCase().endsWith('.flac')
    );
    const textGrids = dirFiles.filter(
      (r) => r.file.name.endsWith('.TextGrid') || r.file.name.endsWith('.textgrid')
    );

    for (const flacRef of flacs) {
      const stem = flacRef.file.name.replace(/\.flac$/i, '');
      const baselineRef = textGrids.find(
        (r) => r.file.name === `${stem}.TextGrid` || r.file.name === `${stem}.textgrid`
      );
      const variantRefs = textGrids.filter((r) => {
        const name = r.file.name;
        if (!name.startsWith(stem + '_')) return false;
        return name.endsWith('.TextGrid') || name.endsWith('.textgrid');
      });

      const suffixFromName = (name: string): string => {
        const ext = name.endsWith('.textgrid') ? '.textgrid' : '.TextGrid';
        const base = name.slice(0, -ext.length);
        return base.slice(stem.length + 1);
      };

      const variants: VariantRef[] = variantRefs.map((r) => ({
        file: r.file,
        suffix: suffixFromName(r.file.name),
      }));

      const errors: string[] = [];
      if (!baselineRef) errors.push(`Missing baseline TextGrid for ${flacRef.file.name}`);

      const subset = inferSubset(flacRef.relativePath);
      const id = entryIdFromPath(flacRef.relativePath);

      entries.push({
        id,
        audioFile: flacRef.file,
        baseline: null,
        baselineFile: baselineRef?.file ?? null,
        variants,
        subset,
        errors,
      });
    }
  }

  return entries;
}
