import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ReportProvider } from './context/ReportContext';
import { Layout } from './components/Layout';
import { PerformanceDashboard } from './pages/PerformanceDashboard';

function App() {
  return (
    <ReportProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<PerformanceDashboard />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ReportProvider>
  );
}

export default App;
