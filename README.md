# wav2vec2-align

A Rust library for **CTC forced alignment** using wav2vec2 acoustic models. Maps a known transcript onto an audio signal at word level, producing millisecond-precision timing boundaries with per-word confidence scores.

Forced alignment solves the problem of synchronizing text with speech: given audio and its transcript, it determines exactly *when* each word is spoken. This is foundational for subtitle generation, speech corpus annotation, audiobook segmentation, and pronunciation analysis.

The library implements the full pipeline — from raw audio to timestamped words — with three parallel Viterbi backends (CPU, wgpu, CUDA) and a multi-policy blank expansion strategy that produces MFA-quality word boundaries.

---

## Acknowledgments

This project was heavily inspired by [wav2vec2aligner](https://github.com/EveryVoiceTTS/wav2vec2aligner) by EveryVoiceTTS. The original Python/TorchAudio implementation served both as a reference for understanding the alignment pipeline and as a comparison baseline for benchmarking. The benchmark harness on the Python side was also built on top of that project (see the patch file `wav2vec2aligner-main.perf-monitoring.patch`). Many thanks to the authors for making their work available.

---

## Features

- **CTC Viterbi forced alignment** with reachability-band pruning for O(T·S) DP instead of naive O(T·S²)
- **Three compute backends**: CPU (always available), wgpu (Vulkan/DX12/Metal), CUDA with zero-copy ORT integration
- **Two model runtimes**: Candle (pure Rust, safetensors) and ONNX Runtime (with CUDA EP support)
- **Zero-copy CUDA path**: when ORT runs on GPU, log-softmax + Viterbi execute entirely on device — only the T-length state path (T×4 bytes) is copied back to host
- **Multi-policy blank expansion** with acoustic-evidence candidate selection for robust word boundaries
- **Deterministic composite confidence scoring**: blends geometric mean emission probability, top-2 margin, p10 log-prob, and boundary blank evidence with piecewise-linear calibration
- **Evaluation and benchmark reporting**: structural, timing, confidence, and performance metrics with per-split aggregation, outlier ranking, and per-word traces

---

## Installation

### Prerequisites

- **Rust 1.75+** (edition 2021)
- A wav2vec2 CTC model in safetensors (Candle) or ONNX format, with `config.json` and `vocab.json`
- Audio must be **16 kHz mono f32** (the standard wav2vec2 sample rate)

### Feature flags

| Feature                  | Description                                                        |
|--------------------------|--------------------------------------------------------------------|
| `wgpu-dp`                 | wgpu Viterbi backend (Vulkan, DX12, Metal)                        |
| `cuda-dp`                | CUDA Viterbi backend via cudarc + NVRTC (requires CUDA toolkit)   |
| `onnx`                   | ONNX Runtime model backend (CPU or CUDA execution provider)       |
| `alignment-profiling`    | Per-stage timing and memory profiling (benchmark mode)             |

CI runs only CPU backends (default + `onnx`); GPU features (`wgpu-dp`, `cuda-dp`) are not tested in CI.

### Basic (CPU only, Candle runtime)

```bash
cargo build --release
```

### With GPU Viterbi (wgpu)

```bash
cargo build --release --features wgpu-dp
```

### Full CUDA pipeline (ONNX + CUDA Viterbi zero-copy)

```bash
cargo build --release --features "onnx,cuda-dp"
```

This gives the fastest path: ORT produces logits on GPU → on-device log-softmax kernel → on-device Viterbi → only the state path array is transferred to host.

### With profiling

```bash
cargo build --release --features "onnx,cuda-dp,alignment-profiling"
```

---

## Usage

### As a library

```rust
use wav2vec2_align::{ForcedAlignerBuilder, Wav2Vec2Config, AlignmentInput};

let config = Wav2Vec2Config {
    model_path: "model.safetensors".into(), // or "model.onnx"
    config_path: "config.json".into(),
    vocab_path: "vocab.json".into(),
    device: "cpu".into(), // or "cuda"
    expected_sample_rate_hz: 16_000,
};

let aligner = ForcedAlignerBuilder::new(config)
    // .with_runtime_kind(RuntimeKind::Onnx) // for ONNX backend
    .build()?;

let input = AlignmentInput {
    sample_rate_hz: 16_000,
    samples: audio_f32_16khz,
    transcript: "the quick brown fox".into(),
    normalized: None, // auto-computed; or precompute with normalize_audio()
};

let output = aligner.align(&input)?;
for word in &output.words {
    println!("{}: [{}, {}) ms  conf={:.2}",
        word.word, word.start_ms, word.end_ms,
        word.confidence.unwrap_or(0.0));
}
```

### Alignment report CLI (benchmark binary)

The project ships with an `alignment_report` binary that serves as both a quality evaluation tool and a performance benchmarker. It processes LibriSpeech test sets, compares predicted word timings against reference TextGrid files, and optionally produces detailed per-stage performance reports.

#### Generating a quality report (JSON)

```bash
cargo run --release --features "onnx,report-cli" --bin alignment_report -- \
    --model-dir models/onnx_wav2vec2_base_960h \
    --dataset-root test-data \
    --device cuda \
    --runtime onnx \
    --output-format json
```

This produces a JSON report with per-sentence structural, timing, and confidence metrics, plus aggregate statistics across test-clean and test-other splits.

#### Generating TextGrid output

```bash
cargo run --release --features "onnx,report-cli" --bin alignment_report -- \
    --model-dir models/onnx_wav2vec2_base_960h \
    --dataset-root test-data \
    --device cuda \
    --runtime onnx \
    --output-format textgrid
```

Writes `.TextGrid` files alongside each LibriSpeech `.flac` file, with tiers for words, word-confidence, and transcript.

#### Running performance benchmarks

```bash
cargo run --release --features "onnx,cuda-dp,alignment-profiling,report-cli" \
    --bin alignment_report -- \
    --model-dir models/onnx_wav2vec2_base_960h \
    --dataset-root test-data \
    --device cuda \
    --runtime onnx \
    --output-format perf \
    --perf-out target/perf/rust-cuda.jsonl \
    --perf-warmup 10 \
    --perf-repeats 30 \
    --perf-aggregate median \
    --perf-append
```

See [BENCHMARKS.md](./BENCHMARKS.md) for a detailed description of the benchmark methodology, known biases, and how to reproduce results.

#### Key CLI flags

| Flag | Description |
|------|-------------|
| `--model-dir` | Path to model directory (must contain model weights, `config.json`, `vocab.json`) |
| `--dataset-root` | Path to test data root (expects `LibriSpeech/test-clean` and `LibriSpeech/test-other` underneath) |
| `--device` | `cpu` or `cuda` |
| `--runtime` | `onnx` or `candle` |
| `--output-format` | `json` (quality report), `textgrid` (TextGrid files), `perf` (performance only) |
| `--limit` / `--offset` | Process a subset of cases |
| `--cases-file` | Filter to specific utterance IDs |
| `--perf-out` | Output path for perf JSON/JSONL |
| `--perf-warmup` | Number of warm-up iterations (default: 10) |
| `--perf-repeats` | Number of timed repeats per utterance (default: 30) |
| `--perf-aggregate` | `median` or `mean` |
| `--perf-append` | Append JSONL records (one per utterance) instead of writing a single JSON file |
| `--perf-scaling-report` | Print T×S scaling analysis with Pearson correlation |

### Python comparison script

A companion Python script reuses the original [wav2vec2aligner](https://github.com/EveryVoiceTTS/wav2vec2aligner) to generate TextGrid files and perf records on the same LibriSpeech data, enabling direct comparison:

```bash
python scripts/wav2vec2aligner_librispeech_textgrids.py \
    --dataset-root test-data/LibriSpeech \
    --device cuda \
    --perf-out target/perf/python-cuda.jsonl \
    --perf-warmup 10 \
    --perf-repeats 30 \
    --perf-aggregate median \
    --perf-append
```

A patch file (`wav2vec2aligner-main.perf-monitoring.patch`) adds profiling instrumentation to the original Python project so that per-stage timings (forward, post, dp, group, conf) are recorded in the same JSONL schema as the Rust implementation.

---

## Test data

Benchmarks and evaluation use the [LibriSpeech](https://www.openslr.org/12) corpus, specifically the **test-clean** and **test-other** subsets. Download them from:

> **https://www.openslr.org/12**

Extract them under `test-data/LibriSpeech/test-clean` and `test-data/LibriSpeech/test-other`.

---

## Architecture

### Pipeline stages

```
Audio [f32] ──► Normalize ──► Forward Pass ──► Log-softmax ──► Viterbi DP ──► Grouping ──► Words
                  (μ=0,σ=1)   (Candle|ORT)    (CPU|GPU)       (CPU|wgpu|CUDA) (expand+score)
```

Each stage is abstracted behind a trait (`RuntimeBackend`, `SequenceAligner`, `Tokenizer`, `WordGrouper`) and can be replaced via the builder. The default implementations are wired in `pipeline::defaults`.

### Module layout

```
src/
├── alignment/
│   ├── viterbi.rs              # Dispatch: CPU → wgpu → CUDA based on T×S threshold
│   ├── cuda/
│   │   ├── viterbi.cu          # CUDA kernels: log_softmax_rows, viterbi_forward, viterbi_backtrace
│   │   └── viterbi_cuda.rs     # cudarc host code, zero-copy + upload variants
│   ├── gpu/
│   │   ├── viterbi.wgsl        # WGSL compute shader: single-workgroup wavefront
│   │   └── viterbi_gpu.rs      # wgpu host code, buffer management, blocking readback
│   ├── tokenization.rs         # Case-aware CTC token sequence builder (blank-interleaved)
│   ├── grouping/
│   │   ├── path_to_words.rs    # Phase 1: walk Viterbi path → raw word boundaries
│   │   ├── blank_expansion.rs  # Phase 2: expand boundaries (Balanced policy)
│   │   └── mod.rs              # Orchestration, quality confidence, calibration
│   └── report.rs               # Evaluation: structural/timing/confidence metrics
├── model/
│   ├── ctc_model.rs            # Wav2Vec2ForCTC (Candle)
│   ├── encoder.rs              # Transformer encoder with positional conv
│   ├── feature_extractor.rs    # Conv1d stack with weight-norm, GroupNorm/LayerNorm
│   ├── feature_projection.rs   # Linear projection to hidden dim
│   └── layers.rs               # LayerNorm, GroupNorm1d (custom for Candle)
├── pipeline/
│   ├── builder.rs              # ForcedAlignerBuilder: wire config → pipeline
│   ├── runtime.rs              # ForcedAligner: align() and align_profiled()
│   ├── model_runtime.rs        # CandleRuntimeBackend, OnnxRuntimeBackend
│   ├── cuda_forward.rs         # CudaLogProbsBuffer: zero-copy device buffer
│   ├── defaults.rs             # Default trait implementations
│   ├── traits.rs               # RuntimeBackend, Tokenizer, SequenceAligner, WordGrouper
│   └── memory_tracker.rs       # Per-stage RSS + GPU memory profiling
├── config.rs                   # Wav2Vec2Config, Wav2Vec2ModelConfig
├── types.rs                    # AlignmentInput, AlignmentOutput, WordTiming, WordConfidenceStats
└── error.rs                    # AlignmentError (Io, Json, Runtime, InvalidInput)

src/bin/
├── alignment_report.rs         # CLI binary: quality reports, TextGrid generation, perf benchmarks
└── alignment_report/
    ├── json_report_formatter.rs      # Quality report JSON serializer
    ├── perf_report_formatter.rs      # Perf benchmark JSON/JSONL serializer
    └── text_grid_report_formatter.rs # TextGrid output writer

scripts/
└── wav2vec2aligner_librispeech_textgrids.py  # Python comparison benchmark script
```

### CTC Viterbi algorithm

The core DP aligns a CTC token sequence `S` (blank-interleaved: `⟨blank, c₁, blank, |, blank, c₂, blank, ...⟩`) against `T` frames of log-probabilities from the acoustic model.

**State transitions** follow CTC constraints: stay on current state (`s → s`), step forward (`s-1 → s`), or skip (`s-2 → s`, only if `tokens[s] ≠ tokens[s-2]` to prevent skipping blanks between repeated characters).

**Reachability band pruning** avoids touching unreachable cells — at each time step `t`, only states in `[curr_start, curr_end]` need to be evaluated. This is significant for long sequences.

**Backpointer storage** uses only 2 bits per cell. Backtrace reconstructs the full path in O(T).

### Three Viterbi backends

All three backends implement identical DP logic and produce bit-identical paths. The dispatch in `viterbi.rs` selects based on T×S product (below 40,000, CPU is faster than GPU launch overhead).

There is no strong performance reason for having both the wgpu and CUDA backends — they achieve comparable throughput on the same hardware. Both exist because building them was a fun exercise in exploring GPU compute from Rust through two very different APIs (portable graphics API vs. vendor-native toolkit).

**CPU** — Scalar DP with ping-pong score arrays. Two `Vec<f32>` of length S are swapped each time step. Reference implementation, always available.

**wgpu** (`wgpu-dp` feature) — A single compute shader dispatch runs the entire T-step DP in one workgroup of 256 threads using `workgroupBarrier()` synchronization. Only the T-length path buffer is copied back to host. Supports Vulkan, DX12, and Metal.

**CUDA** (`cuda-dp` feature) — Three kernels compiled at runtime via NVRTC: `log_softmax_rows` (shared-memory reduction), `viterbi_forward` (wavefront DP in dynamic shared memory), and `viterbi_backtrace` (single-thread O(T) path extraction). When ORT runs on CUDA, the entire log-softmax → Viterbi → backtrace pipeline executes on device with zero-copy — only the final path array transfers to host.

### Tokenization

`build_token_sequence_case_aware` detects vocabulary casing (uppercase-only, lowercase-only, or mixed) and normalizes the transcript accordingly. It produces a blank-interleaved token sequence with a parallel `chars` array mapping each position to its character (or `None` for blanks).

### Word grouping and blank expansion

Grouping happens in three phases:

1. **Path to raw words** — Walks the Viterbi path frame by frame, building tight `[start_frame, end_frame]` boundaries and accumulating emission log-probs and top-2 margins per word.

2. **Blank expansion** — Three policies (Balanced, ConservativeStart, AggressiveTail) expand boundaries into adjacent blank regions with different trade-offs for left expansion, right pullback, and minimum interior silence.

3. **Candidate selection** — All three candidates are scored using boundary blank evidence, shift penalty, and pause plausibility. The best expansion is selected per word.

### Confidence scoring

Each word receives a composite quality confidence score blending geometric mean emission probability, margin, p10 log-prob, and boundary evidence, passed through piecewise-linear calibration to produce a value in [0, 1].

---

## Development / CI

The project uses **Clippy with `-D warnings`** so that lint fixes are required before merge. Formatting is enforced with `cargo fmt --check`. No global Clippy allow flags are used in CI; a few lints are allowed locally where the team prefers readability (e.g. `clippy::needless_range_loop` in DP loops, `clippy::too_many_arguments` on grouping APIs). To run the same checks locally:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --features onnx -- -D warnings
cargo test --features onnx
```

Optional **pre-commit hooks** (format + clippy) are in `githooks/`. Enable with `git config core.hooksPath githooks` from the repo root; see [githooks/README.md](githooks/README.md).

If you add CI (e.g. GitHub Actions), run these three steps as blocking jobs.

---

## License

This project is licensed under the Mozilla Public License Version 2.0 — see the [LICENSE](https://www.mozilla.org/en-US/MPL/2.0/) file for details.