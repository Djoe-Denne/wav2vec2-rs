import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useCorpus } from '../../context/CorpusContext';
import { isDirectoryPickerSupported, pickDirectoryWithFS } from '../import/directoryPicker';

export function ImportScreen() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<{
    entries: number;
    baselines: number;
    variants: number;
    errors: number;
  } | null>(null);
  const { loadFromFileList, loadFromRawFileRefs, loadProgress } = useCorpus();
  const navigate = useNavigate();
  const useFSPicker = isDirectoryPickerSupported();

  async function handleSelectFolder() {
    setError(null);
    setSummary(null);
    if (!useFSPicker) {
      inputRef.current?.click();
      return;
    }
    setLoading(true);
    try {
      const refs = await pickDirectoryWithFS();
      if (refs === null) {
        setLoading(false);
        return;
      }
      const result = await loadFromRawFileRefs(refs);
      if (result.error) {
        setError(result.error);
        setLoading(false);
        return;
      }
      if (result.entriesCount > 0) {
        setSummary({
          entries: result.entriesCount,
          baselines: result.baselines ?? 0,
          variants: result.variants ?? 0,
          errors: result.errors ?? 0,
        });
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  async function handleFolderChange(e: React.ChangeEvent<HTMLInputElement>) {
    const fileList = e.target.files;
    setError(null);
    setSummary(null);
    if (!fileList?.length) {
      e.target.value = '';
      return;
    }
    setLoading(true);
    const result = await loadFromFileList(fileList);
    setLoading(false);
    if (result.error) {
      setError(result.error);
      e.target.value = '';
      return;
    }
    if (result.entriesCount > 0) {
      setSummary({
        entries: result.entriesCount,
        baselines: result.baselines ?? 0,
        variants: result.variants ?? 0,
        errors: result.errors ?? 0,
      });
    }
    e.target.value = '';
  }

  function openDashboard() {
    navigate('/dashboard');
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4">
      <div className="max-w-lg mx-auto">
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">
          LibriSpeech Alignment Explorer
        </h1>
        <p className="text-gray-600 mb-4">
          Select the folder that contains <strong>test-clean</strong> and <strong>test-other</strong> (e.g. your LibriSpeech folder). The app will read all .flac and TextGrid files inside it.
        </p>

        <input
          ref={inputRef}
          type="file"
          {...({ webkitdirectory: '', directory: '' } as React.InputHTMLAttributes<HTMLInputElement>)}
          multiple
          className="hidden"
          onChange={handleFolderChange}
          disabled={loading}
        />
        <button
          type="button"
          onClick={handleSelectFolder}
          disabled={loading}
          className="w-full py-3 px-4 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Loading…' : 'Select folder'}
        </button>

        {!useFSPicker && (
          <p className="mt-3 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-3 py-2">
            In the dialog: <strong>single-click</strong> the folder (e.g. LibriSpeech) so it is highlighted, then click <strong>&quot;Select Folder&quot;</strong> or <strong>&quot;Open&quot;</strong> to confirm. Do not double-click (that only opens the folder and then &quot;Open&quot; does nothing).
          </p>
        )}

        {loadProgress && (
          <p className="mt-4 text-sm text-gray-500">
            {loadProgress.message}
          </p>
        )}

        {error && (
          <div className="mt-4 p-3 rounded-lg bg-red-50 text-red-800 text-sm">
            {error}
          </div>
        )}

        {summary && (
          <div className="mt-6 p-4 rounded-lg border border-gray-200 bg-white">
            <p className="font-medium text-gray-900 mb-2">Found</p>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>{summary.entries} audio file(s)</li>
              <li>{summary.baselines} baseline TextGrid(s)</li>
              <li>{summary.variants} variant TextGrid(s)</li>
              {summary.errors > 0 && (
                <li className="text-amber-700">{summary.errors} error(s)</li>
              )}
            </ul>
            <button
              type="button"
              onClick={openDashboard}
              className="mt-4 py-2 px-4 rounded-lg bg-gray-900 text-white text-sm font-medium hover:bg-gray-800"
            >
              Open dashboard
            </button>
          </div>
        )}

        <p className="mt-6 text-xs text-gray-400">
          {useFSPicker
            ? 'Using the folder picker (Chrome/Edge). Select the LibriSpeech root folder.'
            : 'Best support in Chrome/Edge. Folder selection may be limited in other browsers.'}
        </p>
      </div>
    </div>
  );
}
