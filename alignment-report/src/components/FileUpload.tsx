import React, { useRef, useState } from 'react';
import { useReports } from '../context/ReportContext';
import { Report } from '../types/report';

export function FileUpload() {
  const { addReport } = useReports();
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    try {
      const text = await file.text();
      const data: Report = JSON.parse(text);

      addReport({
        id: crypto.randomUUID(),
        filename: file.name,
        data,
        loadedAt: new Date(),
      });
    } catch (error) {
      console.error('Error parsing JSON:', error);
      alert('Failed to parse JSON file');
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);

    const files = Array.from(e.dataTransfer.files);
    files.forEach(handleFile);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      Array.from(files).forEach(handleFile);
    }
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
        <p className="text-gray-600">
          Drop JSON files here or click to browse
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          multiple
          onChange={handleFileInput}
          className="hidden"
        />
      </div>
    </div>
  );
}
