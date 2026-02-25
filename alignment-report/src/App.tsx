import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ReportProvider } from './context/ReportContext';
import { Layout } from './components/Layout';
import { Overview } from './pages/Overview';
import { Sentences } from './pages/Sentences';
import { SentenceDetail } from './pages/SentenceDetail';
import { Compare } from './pages/Compare';

function App() {
  return (
    <ReportProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Overview />} />
            <Route path="sentences" element={<Sentences />} />
            <Route path="sentences/:id" element={<SentenceDetail />} />
            <Route path="compare" element={<Compare />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ReportProvider>
  );
}

export default App;
