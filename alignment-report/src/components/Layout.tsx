import React, { useEffect } from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { FileUpload } from './FileUpload';
import { Filters } from './Filters';
import { useReports } from '../context/ReportContext';
import { Report } from '../types/report';

export function Layout() {
  const { reports, selectedReport, selectReport, addReport } = useReports();
  const location = useLocation();

  useEffect(() => {
    const loadDefaultReports = async () => {
      try {
        const response = await fetch('/reports/env-run.json');
        if (response.ok) {
          const data: Report = await response.json();
          addReport({
            id: crypto.randomUUID(),
            filename: 'env-run.json',
            data,
            loadedAt: new Date(),
          });
        }
      } catch (error) {
        console.log('No default reports found in /reports/');
      }
    };

    if (reports.length === 0) {
      loadDefaultReports();
    }
  }, []);

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold">Alignment Report Dashboard</h1>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="mb-6">
          <FileUpload />

          {reports.length > 0 && (
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">
                Select Report:
              </label>
              <select
                value={selectedReport?.id || ''}
                onChange={(e) => selectReport(e.target.value)}
                className="w-full md:w-96 border rounded px-3 py-2"
              >
                {reports.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.filename} ({new Date(r.loadedAt).toLocaleTimeString()})
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {selectedReport && (
          <>
            <nav className="bg-white rounded shadow mb-6">
              <div className="flex gap-4 px-4 py-3">
                <NavLink to="/" label="Overview" active={location.pathname === '/'} />
                <NavLink
                  to="/sentences"
                  label="Sentences"
                  active={location.pathname.startsWith('/sentences')}
                />
                <NavLink
                  to="/compare"
                  label="Compare"
                  active={location.pathname === '/compare'}
                />
              </div>
            </nav>

            {!location.pathname.startsWith('/sentences/') && <Filters />}
          </>
        )}

        <Outlet />
      </div>
    </div>
  );
}

function NavLink({
  to,
  label,
  active,
}: {
  to: string;
  label: string;
  active: boolean;
}) {
  return (
    <Link
      to={to}
      className={`px-3 py-2 rounded transition ${
        active
          ? 'bg-blue-600 text-white'
          : 'text-gray-700 hover:bg-gray-100'
      }`}
    >
      {label}
    </Link>
  );
}
