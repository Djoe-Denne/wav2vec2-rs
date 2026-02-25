import React, { createContext, useContext, useState, ReactNode } from 'react';
import { LoadedReport, FilterOptions } from '../types/report';

interface ReportContextType {
  reports: LoadedReport[];
  selectedReport: LoadedReport | null;
  filters: FilterOptions;
  addReport: (report: LoadedReport) => void;
  selectReport: (id: string) => void;
  updateFilters: (filters: Partial<FilterOptions>) => void;
  resetFilters: () => void;
}

const defaultFilters: FilterOptions = {
  split: 'all',
  has_reference: 'all',
  duration_range: [0, Number.MAX_SAFE_INTEGER],
  confidence_threshold: 0,
  search_id: '',
};

const ReportContext = createContext<ReportContextType | undefined>(undefined);

export function ReportProvider({ children }: { children: ReactNode }) {
  const [reports, setReports] = useState<LoadedReport[]>([]);
  const [selectedReport, setSelectedReport] = useState<LoadedReport | null>(null);
  const [filters, setFilters] = useState<FilterOptions>(defaultFilters);

  const addReport = (report: LoadedReport) => {
    setReports((prev) => [...prev, report]);
    if (!selectedReport) {
      setSelectedReport(report);
    }
  };

  const selectReport = (id: string) => {
    const report = reports.find((r) => r.id === id);
    if (report) {
      setSelectedReport(report);
      resetFilters();
    }
  };

  const updateFilters = (newFilters: Partial<FilterOptions>) => {
    setFilters((prev) => ({ ...prev, ...newFilters }));
  };

  const resetFilters = () => {
    setFilters(defaultFilters);
  };

  return (
    <ReportContext.Provider
      value={{
        reports,
        selectedReport,
        filters,
        addReport,
        selectReport,
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
