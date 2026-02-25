# Alignment Report Dashboard

A static web dashboard for visualizing alignment metrics from speech recognition report JSON files.

## Features

- **No Backend Required**: Purely static, runs entirely in the browser
- **Multiple Report Loading**: Load reports via file upload or from `/public/reports/`
- **Interactive Filtering**: Filter by split, has_reference, duration, confidence threshold, and sentence ID
- **Rich Visualizations**: Charts powered by Plotly.js showing distributions and correlations
- **Detailed Views**:
  - **Overview**: KPIs, aggregates, and distribution charts
  - **Sentences**: Sortable table of all sentences with key metrics
  - **Sentence Detail**: Per-word visualizations and comprehensive metrics
  - **Compare**: Side-by-side comparison of two runs

## Getting Started

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Build for Production

```bash
npm run build
```

The static assets will be generated in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Adding Reports

There are two ways to load report JSON files:

### 1. File Upload (Recommended for Local Use)

1. Open the dashboard
2. Click the upload area or drag & drop JSON files directly
3. Multiple files can be loaded at once

### 2. Public Reports Directory

Place JSON report files in the `/public/reports/` directory before building. The dashboard will automatically detect and offer them in a dropdown.

Example:
```
public/
  reports/
    run-2026-01-15.json
    run-2026-01-20.json
```

## Report JSON Schema

The dashboard expects JSON files with the following structure:

```json
{
  "schema_version": 1,
  "meta": {
    "generated_at": "ISO8601 timestamp",
    "model_path": "string",
    "device": "string",
    "frame_stride_ms": number,
    "case_count": number
  },
  "sentences": [
    {
      "id": "string",
      "split": "clean" | "other",
      "has_reference": boolean,
      "duration_ms": number,
      "word_count_pred": number,
      "word_count_ref": number,
      "structural": {
        "negative_duration_word_count": number,
        "overlap_word_count": number,
        "non_monotonic_word_count": number,
        "gap_ratio": number,
        "overlap_ratio": number
      },
      "confidence": {
        "word_conf_mean": number,
        "word_conf_min": number,
        "low_conf_word_ratio": number,
        "blank_frame_ratio": number | null,
        "token_entropy_mean": number | null
      },
      "timing": {
        "start": { /* timing stats */ },
        "end": { /* timing stats */ },
        "abs_err_ms_median": number,
        "abs_err_ms_p90": number,
        "trimmed_mean_abs_err_ms": number,
        "offset_ms": number,
        "drift_ms_per_sec": number
      },
      "per_word": [ /* optional word-level data */ ],
      "notes": []
    }
  ]
}
```

## Deployment

### GitHub Pages

1. Build the project:
   ```bash
   npm run build
   ```

2. The `dist/` folder contains all static assets

3. Configure your repository's GitHub Pages settings to serve from the `dist/` directory (or use a GitHub Action to deploy)

4. Ensure `vite.config.ts` has the correct `base` path if deploying to a subdirectory:
   ```ts
   export default defineConfig({
     base: '/your-repo-name/',
     // ...
   });
   ```

### Other Static Hosts

The built files in `dist/` can be deployed to any static hosting service:
- Netlify
- Vercel
- AWS S3 + CloudFront
- Azure Static Web Apps
- etc.

## Tech Stack

- **Vite**: Build tool and dev server
- **React**: UI framework
- **TypeScript**: Type safety
- **React Router**: Client-side routing
- **Plotly.js**: Interactive charts
- **Tailwind CSS**: Styling

## Project Structure

```
src/
  components/      # Reusable UI components
  context/         # React context for state management
  lib/             # Utility functions and aggregations
  pages/           # Main page components
  types/           # TypeScript type definitions
  App.tsx          # Main app with routing
  main.tsx         # Entry point
public/
  reports/         # Optional: pre-loaded report JSON files
```

## Browser Compatibility

Modern browsers with ES2020 support:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

MIT
