import {
  createContext,
  useCallback,
  useContext,
  useState,
  type ReactNode,
} from 'react';
import type { CorpusState, RawFileRef } from '../shared/types';
import { groupFilesIntoEntries } from '../features/corpus/groupFiles';
import { fileListToRawFileRefs } from '../features/import/fileListToRefs';
import { runCorpusWorker } from '../features/import/runCorpusWorker';

export interface LoadProgress {
  current: number;
  total: number;
  message: string;
}

interface CorpusContextValue {
  corpus: CorpusState | null;
  setCorpus: (state: CorpusState | null) => void;
  loadFromFileList: (
    fileList: FileList | null
  ) => Promise<{
    entriesCount: number;
    error?: string;
    baselines?: number;
    variants?: number;
    errors?: number;
  }>;
  loadFromRawFileRefs: (refs: RawFileRef[]) => Promise<{
    entriesCount: number;
    error?: string;
    baselines?: number;
    variants?: number;
    errors?: number;
  }>;
  loadProgress: LoadProgress | null;
}

const CorpusContext = createContext<CorpusContextValue | null>(null);

export function CorpusProvider({ children }: { children: ReactNode }) {
  const [corpus, setCorpus] = useState<CorpusState | null>(null);
  const [loadProgress, setLoadProgress] = useState<LoadProgress | null>(null);

  const runLoad = useCallback(
    async (refs: RawFileRef[]): Promise<{
      entriesCount: number;
      error?: string;
      baselines?: number;
      variants?: number;
      errors?: number;
    }> => {
      setLoadProgress(null);
      if (refs.length === 0) {
        return { entriesCount: 0, error: 'No files in folder. Select the folder that contains test-clean and test-other (e.g. LibriSpeech), not a file or empty folder.' };
      }
      const entries = groupFilesIntoEntries(refs);
      if (entries.length === 0) {
        return {
          entriesCount: 0,
          error: 'No audio entries found. The folder should contain .flac files and matching .TextGrid files (e.g. stem.TextGrid and stem_suffix.TextGrid) in the same directory.',
        };
      }
      try {
        const state = await runCorpusWorker({
          entries,
          onProgress: (current, total, message) => {
            setLoadProgress({ current, total, message });
          },
        });
        setCorpus(state);
        setLoadProgress(null);
        const baselines = state.entries.filter((e) => e.baseline != null).length;
        const variants = state.entries.reduce((s, e) => s + e.variants.length, 0);
        const errors = state.entries.reduce((s, e) => s + e.errors.length, 0);
        return {
          entriesCount: state.entries.length,
          baselines,
          variants,
          errors,
        };
      } catch (e) {
        setLoadProgress(null);
        const message = e instanceof Error ? e.message : String(e);
        return { entriesCount: 0, error: message };
      }
    },
    []
  );

  const loadFromFileList = useCallback(
    async (
      fileList: FileList | null
    ): Promise<{
      entriesCount: number;
      error?: string;
      baselines?: number;
      variants?: number;
      errors?: number;
    }> => {
      if (!fileList || fileList.length === 0) {
        return { entriesCount: 0, error: 'No files selected. In the dialog, select the folder (e.g. LibriSpeech) and click "Select Folder" to confirm—do not double-click into the folder.' };
      }
      const refs = fileListToRawFileRefs(fileList);
      return runLoad(refs);
    },
    [runLoad]
  );

  const loadFromRawFileRefs = useCallback(
    (refs: RawFileRef[]) => runLoad(refs),
    [runLoad]
  );

  return (
    <CorpusContext.Provider value={{ corpus, setCorpus, loadFromFileList, loadFromRawFileRefs, loadProgress }}>
      {children}
    </CorpusContext.Provider>
  );
}

export function useCorpus() {
  const ctx = useContext(CorpusContext);
  if (!ctx) throw new Error('useCorpus must be used within CorpusProvider');
  return ctx;
}
