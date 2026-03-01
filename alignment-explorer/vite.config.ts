import path from 'path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      worker_threads: path.resolve(process.cwd(), 'src/stubs/worker_threads.ts'),
    },
  },
});
