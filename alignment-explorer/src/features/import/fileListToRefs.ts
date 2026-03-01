import type { RawFileRef } from '../../shared/types';

/**
 * Convert a FileList from <input type="file" webkitdirectory multiple /> to RawFileRef[].
 * Uses webkitRelativePath when available.
 */
export function fileListToRawFileRefs(fileList: FileList | null): RawFileRef[] {
  if (!fileList || fileList.length === 0) return [];
  const refs: RawFileRef[] = [];
  for (let i = 0; i < fileList.length; i++) {
    const file = fileList[i];
    const relativePath =
      'webkitRelativePath' in file && typeof file.webkitRelativePath === 'string'
        ? file.webkitRelativePath
        : file.name;
    refs.push({ file, relativePath });
  }
  return refs;
}
