import React, { useRef, useState } from 'react';
import { useReports } from '../context/ReportContext';
import type { LoadedRun, PerfRunFile, RustPerfRecord } from '../types/report';

function isValidRecord(r: unknown): r is RustPerfRecord {
  if (!r || typeof r !== 'object') return false;
  const o = r as Record<string, unknown>;
  return (
    typeof o.utterance_id === 'string' &&
    typeof o.duration_ms === 'number' &&
    typeof o.num_frames_t === 'number' &&
    typeof o.total_ms === 'number'
  );
}

/** Parse JSONL: one JSON object per line (Rust streaming format). */
function parseJsonlRecords(text: string): RustPerfRecord[] {
  const records: RustPerfRecord[] = [];
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  for (const line of lines) {
    try {
      const obj: unknown = JSON.parse(line);
      if (isValidRecord(obj)) records.push(obj as RustPerfRecord);
    } catch {
      // skip invalid lines
    }
  }
  return records;
}

function parseRunFile(json: unknown, filename: string): LoadedRun | null {
  let records: RustPerfRecord[];
  if (Array.isArray(json)) {
    records = json.filter(isValidRecord) as RustPerfRecord[];
  } else if (json && typeof json === 'object' && 'records' in json) {
    const raw = (json as PerfRunFile).records;
    records = Array.isArray(raw) ? raw.filter(isValidRecord) as RustPerfRecord[] : [];
  } else {
    return null;
  }
  if (records.length === 0) return null;

  return {
    id: crypto.randomUUID(),
    name: filename.replace(/\.json$/i, ''),
    records,
    loadedAt: new Date(),
  };
}

export function FileUpload() {
  const { addRun } = useReports();
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    try {
      const text = await file.text();
      let run: LoadedRun | null = null;
      try {
        const json: unknown = JSON.parse(text);
        run = parseRunFile(json, file.name);
      } catch {
        // Not single JSON; try JSONL (one record per line, e.g. Rust streaming output)
        const records = parseJsonlRecords(text);
        if (records.length > 0) {
          run = {
            id: crypto.randomUUID(),
            name: file.name.replace(/\.json$/i, ''),
            records,
            loadedAt: new Date(),
          };
        }
      }
      if (run) {
        addRun(run);
      } else {
        alert('No valid performance records in file. Expected { records: [...] }, an array of records, or JSONL (one record per line).');
      }
    } catch (error) {
      console.error('Error parsing file:', error);
      alert('Failed to parse file');
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const files = Array.from(e.dataTransfer.files).filter((f) => f.name.endsWith('.json') || f.type === 'application/json');
    files.forEach(handleFile);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) Array.from(files).forEach(handleFile);
    e.target.value = '';
  };

  return (
    <div className="mb-4">
      <div
        className={`border-2 border-dashed rounded p-6 text-center cursor-pointer transition ${
          dragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
        }`}
        onDrop={handleDrop}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onClick={() => fileInputRef.current?.click()}
      >
        <p className="text-gray-600">Drop performance run JSON files here or click to browse</p>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,application/json"
          multiple
          onChange={handleFileInput}
          className="hidden"
        />
      </div>
    </div>
  );
}
