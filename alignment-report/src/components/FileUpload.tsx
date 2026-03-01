import React, { useRef, useState } from 'react';
import { useReports } from '../context/ReportContext';
import { loadRunFromText } from '../lib/reportLoader';

export function FileUpload() {
  const { addRun } = useReports();
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    try {
      const text = await file.text();
      const run = loadRunFromText(text, file.name);
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
