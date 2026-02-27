# wav2vec2-rs

## Modular Pipeline Architecture

The crate is now organized by concept (`config`, `model`, `alignment`, `pipeline`,
`types`, `error`) and exposes a builder-first API so developers can replace core
alignment bricks.

Default builder usage:

```rust
use wav2vec2_rs::{ForcedAlignerBuilder, Wav2Vec2Config};

let config = Wav2Vec2Config {
    model_path: "models/wav2vec2-base-960h/model.safetensors".into(),
    config_path: "models/wav2vec2-base-960h/config.json".into(),
    vocab_path: "models/wav2vec2-base-960h/vocab.json".into(),
    device: "cpu".into(),
    expected_sample_rate_hz: Wav2Vec2Config::DEFAULT_SAMPLE_RATE_HZ,
};

let aligner = ForcedAlignerBuilder::new(config).build()?;
```

### Switch runtime backend

`ForcedAlignerBuilder` defaults to the Candle runtime. You can switch to ONNX Runtime
through the builder:

```rust
use wav2vec2_rs::{ForcedAlignerBuilder, RuntimeKind, Wav2Vec2Config};

let config = Wav2Vec2Config {
    model_path: "models/wav2vec2-base-960h/model.onnx".into(),
    config_path: "models/wav2vec2-base-960h/config.json".into(),
    vocab_path: "models/wav2vec2-base-960h/vocab.json".into(),
    device: "cuda".into(), // "cpu" and "cuda" are supported
    expected_sample_rate_hz: Wav2Vec2Config::DEFAULT_SAMPLE_RATE_HZ,
};

let aligner = ForcedAlignerBuilder::new(config)
    .with_runtime_kind(RuntimeKind::Onnx)
    .build()?;
```

Notes:

- ONNX support is behind the `onnx` feature: build with `--features onnx`.
- ⚠️ **ONNX Runtime with CUDA**: To use the ONNX runtime with CUDA, you need CUDA 12.8 or 13.1 and cuDNN 9+ installed and on your PATH.
- Candle keeps working as before (and remains the default path).
- `model_path` points to runtime-specific weights:
  - Candle: `model.safetensors`
  - ONNX: `model.onnx`

### Add custom pipeline components

You can replace any pipeline brick by implementing one of the public traits:

- `Tokenizer`
- `SequenceAligner`
- `WordGrouper`

Then inject your implementations into the builder with:

- `.with_tokenizer(...)`
- `.with_sequence_aligner(...)`
- `.with_word_grouper(...)`

Example:

```rust
use std::collections::HashMap;
use wav2vec2_rs::{
    AlignmentError, ForcedAlignerBuilder, SequenceAligner, TokenSequence, Tokenizer, Wav2Vec2Config,
    WordGrouper, WordTiming,
};

struct MyTokenizer;
impl Tokenizer for MyTokenizer {
    fn tokenize(
        &self,
        transcript: &str,
        vocab: &HashMap<char, usize>,
        blank_id: usize,
        _word_sep_id: usize,
    ) -> TokenSequence {
        // Minimal example: per-char tokens + blank
        let mut tokens = vec![blank_id];
        let mut chars = vec![None];
        for c in transcript.chars() {
            if let Some(&id) = vocab.get(&c) {
                tokens.push(id);
                chars.push(Some(c));
                tokens.push(blank_id);
                chars.push(None);
            }
        }
        TokenSequence { tokens, chars }
    }
}

struct MyAligner;
impl SequenceAligner for MyAligner {
    fn align_path(
        &self,
        _log_probs: &[Vec<f32>],
        _tokens: &[usize],
    ) -> Result<Vec<(usize, usize)>, AlignmentError> {
        // Replace with your own alignment algorithm.
        Ok(Vec::new())
    }
}

struct MyGrouper;
impl WordGrouper for MyGrouper {
    fn group_words(
        &self,
        _path: &[(usize, usize)],
        _token_sequence: &TokenSequence,
        _log_probs: &[Vec<f32>],
        _blank_id: usize,
        _word_sep_id: usize,
        _stride_ms: f64,
    ) -> Vec<WordTiming> {
        // Replace with your own grouping logic.
        Vec::new()
    }
}

let config = Wav2Vec2Config {
    model_path: "models/wav2vec2-base-960h/model.safetensors".into(),
    config_path: "models/wav2vec2-base-960h/config.json".into(),
    vocab_path: "models/wav2vec2-base-960h/vocab.json".into(),
    device: "cpu".into(),
    expected_sample_rate_hz: Wav2Vec2Config::DEFAULT_SAMPLE_RATE_HZ,
};

let aligner = ForcedAlignerBuilder::new(config)
    .with_tokenizer(Box::new(MyTokenizer))
    .with_sequence_aligner(Box::new(MyAligner))
    .with_word_grouper(Box::new(MyGrouper))
    .build()?;
```

## Prepare Test Data

This project uses LibriSpeech test sets as reference test data.
The setup script downloads archives from OpenSLR, extracts them, and generates
reference word alignments with a PyTorch `Wav2Vec2ForCTC` model.

By default, it loads the local model from `models/wav2vec2-base-960h`.

### 1) Install Python dependencies

CUDA 12.8 (priority):

```bash
pip install -r scripts/requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
```

CUDA 13.0:

```bash
pip install -r scripts/requirements.txt --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

CPU fallback:

```bash
pip install -r scripts/requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

### 2) Run setup script

```bash
python scripts/pytorch_aligner.py
```

The script will:

- download `test-clean.tar.gz` and `test-other.tar.gz` from OpenSLR
- extract data to `test-data/LibriSpeech/`
- generate alignment references:
  - `test-data/alignments/test-clean.json`
  - `test-data/alignments/test-other.json`
- run emissions with PyTorch Wav2Vec2 (`Wav2Vec2ForCTC`)
- decode FLAC audio with `soundfile` (no TorchCodec requirement)

### 3) Optional flags

Regenerate everything:

```bash
python scripts/pytorch_aligner.py --overwrite
```

Run a subset only:

```bash
python scripts/pytorch_aligner.py --subsets test-clean
```

Limit aligned utterances per subset:

```bash
python scripts/pytorch_aligner.py --limit 100
```

Use a different local model directory:

```bash
python scripts/pytorch_aligner.py --model-dir models/wav2vec2-base-960h
```

### 4) Generate LibriSpeech TextGrid files in batch (wav2vec2aligner)

Create and activate a virtual environment (Bash):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install CUDA-enabled PyTorch + script dependencies (Bash, pick one):

```bash
# CUDA 12.8
pip install -r scripts/requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple

# CUDA 13.0
# pip install -r scripts/requirements.txt --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

Install CUDA-enabled PyTorch + script dependencies (PowerShell, pick one):

```powershell
# CUDA 12.8
pip install -r scripts/requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple

# CUDA 13.0
# pip install -r scripts/requirements.txt --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

Install the local aligner package (Bash / PowerShell):

```bash
pip install -e wav2vec2aligner-main
```

```powershell
pip install -e wav2vec2aligner-main
```

Clean existing TextGrid files before regeneration (Bash):

```bash
find test-data/LibriSpeech -type f -name "*.TextGrid" -delete
```

Clean existing TextGrid files before regeneration (PowerShell):

```powershell
Get-ChildItem .\test-data\LibriSpeech -Recurse -Filter *.TextGrid | Remove-Item -Force
```

Run batch TextGrid generation (Bash; model loads once, outputs overwrite existing `.TextGrid` files):

```bash
python scripts/wav2vec2aligner_librispeech_textgrids.py \
  --dataset-root test-data/LibriSpeech \
  --subsets test-clean,test-other \
  --device cuda \
  --overwrite
```

Run batch TextGrid generation (PowerShell; model loads once, outputs overwrite existing `.TextGrid` files):

```powershell
python .\scripts\wav2vec2aligner_librispeech_textgrids.py `
  --dataset-root .\test-data\LibriSpeech `
  --subsets test-clean,test-other `
  --device cuda `
  --overwrite
```

Quick smoke run (Bash):

```bash
python scripts/wav2vec2aligner_librispeech_textgrids.py \
  --dataset-root test-data/LibriSpeech \
  --subsets test-clean \
  --limit 5 \
  --progress-every 1 \
  --device cuda \
  --overwrite
```

Quick smoke run (PowerShell):

```powershell
python .\scripts\wav2vec2aligner_librispeech_textgrids.py `
  --dataset-root .\test-data\LibriSpeech `
  --subsets test-clean `
  --limit 5 `
  --progress-every 1 `
  --device cuda `
  --overwrite
```

### CUDA launch quick reference (script + bin)

Python script (Bash):

```bash
python scripts/wav2vec2aligner_librispeech_textgrids.py \
  --dataset-root test-data/LibriSpeech \
  --subsets test-clean,test-other \
  --device cuda \
  --overwrite
```

Python script (PowerShell):

```powershell
python .\scripts\wav2vec2aligner_librispeech_textgrids.py `
  --dataset-root .\test-data\LibriSpeech `
  --subsets test-clean,test-other `
  --device cuda `
  --overwrite
```

Rust bin (`alignment_report`) TextGrid mode on CUDA (Bash):

```bash
cargo run --features "report-cli,cuda" --bin alignment_report -- \
  --dataset-root test-data \
  --output-format textgrid \
  --textgrid-suffix _cmp \
  --device cuda
```

Rust bin (`alignment_report`) TextGrid mode on CUDA (PowerShell):

```powershell
cargo run --features "report-cli,cuda" --bin alignment_report -- `
  --dataset-root .\test-data `
  --output-format textgrid `
  --textgrid-suffix _cmp `
  --device cuda
```

Perf mode examples (keep TextGrid output + write perf JSON sidecar):

Python script perf mode (Bash):

```bash
python scripts/wav2vec2aligner_librispeech_textgrids.py \
  --dataset-root test-data/LibriSpeech \
  --subsets test-clean,test-other \
  --device cuda \
  --overwrite \
  --perf-out target/perf/python-perf.json \
  --perf-warmup 10 \
  --perf-repeats 30 \
  --perf-aggregate median
```

```powershell
python scripts/wav2vec2aligner_librispeech_textgrids.py `
  --dataset-root test-data/LibriSpeech `
  --subsets test-clean,test-other `
  --device cuda `
  --overwrite `
  --perf-out target/perf/python-perf.json `
  --perf-warmup 10 `
  --perf-repeats 30 `
  --perf-aggregate median
```

Rust bin perf mode (Bash):

```bash
cargo run --features "report-cli,cuda" --bin alignment_report -- \
  --dataset-root test-data \
  --output-format textgrid \
  --device cuda \
  --perf-out target/perf/rust-perf.json \
  --perf-warmup 10 \
  --perf-repeats 30 \
  --perf-aggregate median
```

Rust bin perf append mode (JSONL + summary) (PowerShell):

```powershell
cargo run --features "report-cli,cuda" --bin alignment_report -- `
  --dataset-root .\test-data `
  --output-format textgrid `
  --device cuda `
  --perf-out .\target\perf\rust-perf.jsonl `
  --perf-append `
  --perf-warmup 10 `
  --perf-repeats 30 `
  --perf-aggregate median
```

### 5) Baseline comparative TextGrid files (upstream project)

The batch script above is intended to generate **baseline comparative** `.TextGrid` files.
You can use these files as a reference to compare your own alignment implementation.

To respect and credit the original authors, the upstream aligner project is maintained
separately and should be treated as an external dependency:

- Upstream project: [EveryVoiceTTS/wav2vec2aligner](https://github.com/EveryVoiceTTS/wav2vec2aligner)
- Keep it as its own repository in your workspace (for example in `wav2vec2aligner-main/`)
- Install it locally with `pip install -e wav2vec2aligner-main`

If `wav2vec2aligner-main/` is not present yet, clone it first:

```bash
git clone https://github.com/EveryVoiceTTS/wav2vec2aligner wav2vec2aligner-main
pip install -e wav2vec2aligner-main
```

```powershell
git clone https://github.com/EveryVoiceTTS/wav2vec2aligner wav2vec2aligner-main
pip install -e wav2vec2aligner-main
```

## Alignment Metrics Report (CLI)

The threshold-based integration test is replaced by a deterministic report
generator.

The report command does **not** fail based on timing quality. It only fails on:

- I/O or parsing failures
- inference/runtime failures
- invalid numeric metrics (NaN / Inf)

### Build and run

```bash
cargo run --features report-cli --bin alignment_report -- --help
```

For CUDA runs with this binary, enable both `report-cli` and `cuda` features:

```bash
cargo run --features "report-cli,cuda" --bin alignment_report -- --help
```

Basic run (JSON report mode, default):

```bash
cargo run --features report-cli --bin alignment_report -- \
  --dataset-root test-data \
  --out target/alignment_reports/run.json
```

Run a filtered subset from a case-id file:

```bash
cargo run --features report-cli --bin alignment_report -- \
  --cases-file current-failing-tests.txt \
  --offset 0 \
  --limit 200 \
  --out target/alignment_reports/failing-cases.json
```

Generate TextGrid files instead of JSON (written next to each `.flac`):

```bash
cargo run --features "report-cli,cuda" --bin alignment_report -- \
  --dataset-root test-data \
  --output-format textgrid \
  --textgrid-suffix _cmp \
  --device cuda
```

### Required local files

- Candle runtime: `models/wav2vec2-base-960h/model.safetensors`
- ONNX runtime (`--features onnx`): `models/wav2vec2-base-960h/model.onnx`
- `models/wav2vec2-base-960h/config.json`
- `models/wav2vec2-base-960h/vocab.json`
- `test-data/LibriSpeech/test-clean/**` and/or `test-data/LibriSpeech/test-other/**`
- For `--output-format=json`: TextGrid + sibling FLAC files (the CLI scans for
  `*.TextGrid` and uses matching `.flac`)
- For `--output-format=textgrid`: transcript files (`*.trans.txt`) + sibling
  FLAC files

`alignment_report` runtime default:

- with `--features onnx`: defaults to `--runtime onnx`
- without `--features onnx`: defaults to `--runtime candle`

### CLI options

- `--model-dir <PATH>`: model directory (default: `models/wav2vec2-base-960h`)
- `--dataset-root <PATH>`: dataset root (default: `test-data`)
- `--cases-file <PATH>`: optional case list file
- `--runtime <onnx|candle>`: inference runtime backend (default depends on enabled features)
- `--output-format <json|textgrid>`: output mode (default: `json`)
- `--textgrid-suffix <STRING>`: suffix appended before `.TextGrid` in textgrid
  mode (default: empty)
- `--out <PATH>`: output JSON path (default:
  `target/alignment_reports/alignment-report-<timestamp>.json`; ignored in
  `textgrid` mode
- `--limit <N>`: max selected cases
- `--offset <N>`: skip first N selected cases
- `--device <cpu|cuda>`: runtime device (default: `cpu`)
- `--perf-out <PATH>`: optional perf report path (JSON or JSONL with `--perf-append`)
- `--perf-warmup <N>`: warm-up iterations on first measured utterance only (default: `10`)
- `--perf-repeats <N>`: measured repeats per utterance (default: `30`)
- `--perf-aggregate <median|mean>`: repeat aggregation mode (default: `median`)
- `--perf-append`: append JSONL records to `--perf-out` (also writes `<perf-out>.summary.json`)

### Environment variables (optional)

Each CLI option can be configured through an environment variable:

- `WAV2VEC2_REPORT_MODEL_DIR` -> `--model-dir`
- `WAV2VEC2_REPORT_DATASET_ROOT` -> `--dataset-root`
- `WAV2VEC2_REPORT_CASES_FILE` -> `--cases-file`
- `WAV2VEC2_REPORT_RUNTIME` -> `--runtime`
- `WAV2VEC2_REPORT_FORMAT` -> `--output-format`
- `WAV2VEC2_REPORT_TEXTGRID_SUFFIX` -> `--textgrid-suffix`
- `WAV2VEC2_REPORT_OUT` -> `--out`
- `WAV2VEC2_REPORT_LIMIT` -> `--limit`
- `WAV2VEC2_REPORT_OFFSET` -> `--offset`
- `WAV2VEC2_REPORT_DEVICE` -> `--device`
- `WAV2VEC2_REPORT_PERF_OUT` -> `--perf-out`
- `WAV2VEC2_REPORT_PERF_WARMUP` -> `--perf-warmup`
- `WAV2VEC2_REPORT_PERF_REPEATS` -> `--perf-repeats`
- `WAV2VEC2_REPORT_PERF_AGGREGATE` -> `--perf-aggregate`
- `WAV2VEC2_REPORT_PERF_APPEND` -> `--perf-append`

Command-line flags take precedence over environment variables.

Example (Bash):

```bash
WAV2VEC2_REPORT_CASES_FILE=current-failing-tests.txt \
WAV2VEC2_REPORT_DEVICE=cpu \
cargo run --features report-cli --bin alignment_report -- --out target/alignment_reports/env-run.json
```

Example (PowerShell):

```powershell
$env:WAV2VEC2_REPORT_CASES_FILE = "current-failing-tests.txt"
$env:WAV2VEC2_REPORT_DEVICE = "cpu"
cargo run --features report-cli --bin alignment_report -- --out target/alignment_reports/env-run.json
```

### Case file format

`--cases-file` accepts one ID per line (e.g. `1089-134686-0000`). It also
tolerates:

- lines prefixed like `L123:<id>`
- test-name lines containing `::audio::<id>`
- comments starting with `#`

### Report output

With `--output-format=json`, the generated JSON includes:

- `schema_version = 1`
- `meta` (`generated_at`, `model_path`, `device`, `frame_stride_ms`,
  `case_count`)
- `sentences[]` with per-utterance metrics and `split = clean|other|unknown`
- `aggregates` with global/per-split distributions and deterministic outlier
  ranking

With `--output-format=textgrid`, the tool writes Praat TextGrid files next to
each `.flac` file. The output name is `<audio_stem><suffix>.TextGrid`, where
`suffix` comes from `--textgrid-suffix`.
