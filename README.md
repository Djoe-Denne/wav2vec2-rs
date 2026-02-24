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

## Integration Timestamp Tests

Integration tests compare Rust-generated word timestamps against the PyTorch
reference JSON under `test-data/alignments`.

Required local files:

- `models/wav2vec2-base-960h/model.safetensors`
- `models/wav2vec2-base-960h/config.json`
- `models/wav2vec2-base-960h/vocab.json`
- audio files referenced by `test-data/alignments/*.json`

Default run (deterministic random sample of 50 utterances):

```bash
cargo test pytorch_alignment_reference_matches_within_delta -- --nocapture
```

Full run (all utterances from `test-clean` + `test-other`):

```bash
WAV2VEC2_IT_FULL=1 cargo test pytorch_alignment_reference_matches_within_delta -- --nocapture
```

Override average timestamp tolerance (milliseconds):

```bash
WAV2VEC2_IT_DELTA_MS=30 cargo test pytorch_alignment_reference_matches_within_delta -- --nocapture
```

Optional overrides:

- `WAV2VEC2_IT_MODEL_DIR=/absolute/or/relative/model/dir`
- `WAV2VEC2_IT_SEED=123` (sample seed for default 50 mode)
- `WAV2VEC2_IT_DEVICE=cpu|cuda`
