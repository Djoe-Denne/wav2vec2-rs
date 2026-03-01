import type { RawFileRef } from '../../shared/types';

/** Directory handle with async entries (File System Access API; entries() may be missing in TS lib) */
interface DirHandleWithEntries {
  entries(): AsyncIterableIterator<[string, FileSystemFileHandle | FileSystemDirectoryHandle]>;
  kind: string;
}

/** Recursively collect all files from a directory handle with relative paths */
async function* listFilesRecursive(
  dirHandle: DirHandleWithEntries,
  basePath = ''
): AsyncGenerator<RawFileRef> {
  for await (const [name, handle] of dirHandle.entries()) {
    const relativePath = basePath ? `${basePath}/${name}` : name;
    if (handle.kind === 'file') {
      const file = await handle.getFile();
      yield { file, relativePath };
    } else if (handle.kind === 'directory') {
      yield* listFilesRecursive(handle as unknown as DirHandleWithEntries, relativePath);
    }
  }
}

/**
 * Use File System Access API to pick a folder. Returns all files with relative paths, or null if cancelled/unsupported.
 */
export async function pickDirectoryWithFS(): Promise<RawFileRef[] | null> {
  if (typeof window === 'undefined' || !('showDirectoryPicker' in window)) {
    return null;
  }
  try {
    const dirHandle = await (window as Window & { showDirectoryPicker: () => Promise<FileSystemDirectoryHandle> })
      .showDirectoryPicker();
    const refs: RawFileRef[] = [];
    for await (const ref of listFilesRecursive(dirHandle as unknown as DirHandleWithEntries)) {
      refs.push(ref);
    }
    return refs;
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      return null;
    }
    throw err;
  }
}

export function isDirectoryPickerSupported(): boolean {
  return typeof window !== 'undefined' && 'showDirectoryPicker' in window;
}
