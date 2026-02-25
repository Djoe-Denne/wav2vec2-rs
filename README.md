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

Basic run:

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

### Required local files

- `models/wav2vec2-base-960h/model.safetensors`
- `models/wav2vec2-base-960h/config.json`
- `models/wav2vec2-base-960h/vocab.json`
- `test-data/LibriSpeech/test-clean/**` and/or `test-data/LibriSpeech/test-other/**`
- TextGrid + sibling FLAC files (the CLI scans for `*.TextGrid` and uses
  matching `.flac`)

### CLI options

- `--model-dir <PATH>`: model directory (default: `models/wav2vec2-base-960h`)
- `--dataset-root <PATH>`: dataset root (default: `test-data`)
- `--cases-file <PATH>`: optional case list file
- `--out <PATH>`: output JSON path (default:
  `target/alignment_reports/alignment-report-<timestamp>.json`)
- `--limit <N>`: max selected cases
- `--offset <N>`: skip first N selected cases
- `--device <cpu|cuda>`: runtime device (default: `cpu`)

### Environment variables (optional)

Each CLI option can be configured through an environment variable:

- `WAV2VEC2_REPORT_MODEL_DIR` -> `--model-dir`
- `WAV2VEC2_REPORT_DATASET_ROOT` -> `--dataset-root`
- `WAV2VEC2_REPORT_CASES_FILE` -> `--cases-file`
- `WAV2VEC2_REPORT_OUT` -> `--out`
- `WAV2VEC2_REPORT_LIMIT` -> `--limit`
- `WAV2VEC2_REPORT_OFFSET` -> `--offset`
- `WAV2VEC2_REPORT_DEVICE` -> `--device`

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

The generated JSON includes:

- `schema_version = 1`
- `meta` (`generated_at`, `model_path`, `device`, `frame_stride_ms`,
  `case_count`)
- `sentences[]` with per-utterance metrics and `split = clean|other|unknown`
- `aggregates` with global/per-split distributions and deterministic outlier
  ranking
