import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { CorpusProvider } from './context/CorpusContext';
import { ImportScreen } from './features/dashboard/ImportScreen';
import { Dashboard } from './features/dashboard/Dashboard';
import { AudioDetail } from './features/audio-detail/AudioDetail';

function App() {
  return (
    <CorpusProvider>
      <BrowserRouter>
        <Routes>
        <Route path="/" element={<ImportScreen />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/audio/:entryId" element={<AudioDetail />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
    </CorpusProvider>
  );
}

export default App;
