import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { loadRunFromText } from '../lib/reportLoader';
import type { GlobalFilters, LoadedRun } from '../types/report';

/** JSON files in public/reports/ to load on launch. */
const PUBLIC_REPORT_FILES = [
  'rust-perf-cuda.json',
  'python-perf.json',
  'rust-perf-generic-gpu.json',
];

export interface ReportContextType {
  runs: LoadedRun[];
  baselineId: string | null;
  visibility: Record<string, boolean>;
  filters: GlobalFilters;
  addRun: (run: LoadedRun) => void;
  removeRun: (id: string) => void;
  updateRun: (id: string, patch: Partial<Pick<LoadedRun, 'name'>>) => void;
  setBaseline: (id: string | null) => void;
  setVisibility: (id: string, visible: boolean) => void;
  updateFilters: (partial: Partial<GlobalFilters>) => void;
  resetFilters: () => void;
}

const defaultFilters: GlobalFilters = {
  duration_range: [0, Number.MAX_SAFE_INTEGER],
  frame_count_range: [0, Number.MAX_SAFE_INTEGER],
  search_id: '',
};

const ReportContext = createContext<ReportContextType | undefined>(undefined);

export function ReportProvider({ children }: { children: ReactNode }) {
  const [runs, setRuns] = useState<LoadedRun[]>([]);
  const [baselineId, setBaselineId] = useState<string | null>(null);
  const [visibility, setVisibilityMap] = useState<Record<string, boolean>>({});
  const [filters, setFilters] = useState<GlobalFilters>(defaultFilters);
  const publicReportsLoaded = useRef(false);

  const addRun = useCallback((run: LoadedRun) => {
    setRuns((prev) => [...prev, run]);
    setVisibilityMap((prev) => ({ ...prev, [run.id]: true }));
  }, []);

  const removeRun = useCallback((id: string) => {
    setRuns((prev) => prev.filter((r) => r.id !== id));
    setBaselineId((prev) => (prev === id ? null : prev));
    setVisibilityMap((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
  }, []);

  const updateRun = useCallback((id: string, patch: Partial<Pick<LoadedRun, 'name'>>) => {
    setRuns((prev) =>
      prev.map((r) => (r.id === id ? { ...r, ...patch } : r))
    );
  }, []);

  const setBaseline = useCallback((id: string | null) => {
    setBaselineId(id);
  }, []);

  const setVisibility = useCallback((id: string, visible: boolean) => {
    setVisibilityMap((prev) => ({ ...prev, [id]: visible }));
  }, []);

  const updateFilters = useCallback((partial: Partial<GlobalFilters>) => {
    setFilters((prev) => ({ ...prev, ...partial }));
  }, []);

  const resetFilters = useCallback(() => {
    setFilters(defaultFilters);
  }, []);

  useEffect(() => {
    if (publicReportsLoaded.current) return;
    publicReportsLoaded.current = true;
    PUBLIC_REPORT_FILES.forEach(async (filename) => {
      try {
        const res = await fetch(`/reports/${filename}`);
        if (!res.ok) return;
        const text = await res.text();
        const run = loadRunFromText(text, filename);
        if (run) addRun(run);
      } catch {
        // ignore fetch/parse errors for optional public reports
      }
    });
  }, [addRun]);

  return (
    <ReportContext.Provider
      value={{
        runs,
        baselineId,
        visibility,
        filters,
        addRun,
        removeRun,
        updateRun,
        setBaseline,
        setVisibility,
        updateFilters,
        resetFilters,
      }}
    >
      {children}
    </ReportContext.Provider>
  );
}

export function useReports() {
  const context = useContext(ReportContext);
  if (!context) {
    throw new Error('useReports must be used within a ReportProvider');
  }
  return context;
}
