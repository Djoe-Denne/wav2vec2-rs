import { Outlet } from 'react-router-dom';
import { FileUpload } from './FileUpload';
import { RunList } from './RunList';
import { Filters } from './Filters';
import { Glossary } from './Glossary';

export function Layout() {
  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold">Performance Analysis Dashboard</h1>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-6">
        <Glossary />
        <FileUpload />
        <RunList />
        <Filters />
        <Outlet />
      </div>
    </div>
  );
}
