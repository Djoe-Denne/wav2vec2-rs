# Benchmarks

This document describes the performance benchmarking methodology used to compare the Rust implementation (`wav2vec2-align`) against the Python/TorchAudio reference implementation ([wav2vec2aligner](https://github.com/EveryVoiceTTS/wav2vec2aligner)). The benchmarks measure **computation time** and **memory footprint** across the full forced alignment pipeline.

---

## Purpose and scope

The goal of these benchmarks is not to declare a definitive winner, but to provide an **order-of-magnitude comparison** between the two implementations running the same alignment task on the same data. The results should be treated as indicative rather than absolute — there are known biases and caveats documented below.

Both benchmarks measure the same pipeline stages: model forward pass, log-softmax post-processing, Viterbi dynamic programming, word grouping/blank expansion, and confidence scoring.

---

## Test data

All benchmarks use the [LibriSpeech](https://www.openslr.org/12) corpus:

- **test-clean** — 2,620 utterances from clean recording conditions
- **test-other** — 2,939 utterances from noisier conditions

Download from: **https://www.openslr.org/12**

Extract under `test-data/LibriSpeech/test-clean` and `test-data/LibriSpeech/test-other`.

---

## What is measured

### Timing stages

Both implementations break down total alignment time into the same stages:

| Stage | What it measures |
|-------|-----------------|
| `forward_ms` | Model inference (wav2vec2 encoder forward pass) |
| `post_ms` | Log-softmax + tensor extraction (0 for CUDA zero-copy in Rust) |
| `dp_ms` | Viterbi DP + backtrace |
| `group_ms` | Path walking + blank expansion + candidate selection |
| `conf_ms` | Confidence scoring |
| `align_ms` | Sum of dp + group + conf |
| `total_ms` | End-to-end time for one utterance |

### Derived metrics

- `align_ms_per_ts` — alignment time normalized by T×S product (frames × CTC state length), useful for comparing scaling behavior
- `align_ms_per_t` — alignment time normalized by number of frames

### Memory

Memory profiling captures GPU device memory usage (`cudaMemGetInfo` on both sides for comparability) and, where available, per-stage peak memory.

---

## How to reproduce

### Rust benchmark

Build with profiling support and run in perf mode:

```bash
# Build
cargo build --release --features "onnx,cuda-dp,alignment-profiling,report-cli"

# Run
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

For CPU-only benchmarks, use `--device cpu` and drop the `cuda-dp` feature. For the wgpu backend, use `--features "onnx,wgpu-dp,alignment-profiling,report-cli"`.

### Python benchmark

First, apply the perf-monitoring patch to the wav2vec2aligner codebase:

```bash
cd wav2vec2aligner-main
git apply ../wav2vec2aligner-main.perf-monitoring.patch
```

Then run the benchmark script:

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

### Output format

Both produce JSONL files with one record per utterance (when using `--perf-append`), plus a `.summary.json` with aggregate statistics. The schema is shared:

```json
{
  "utterance_id": "1089-134686-0000",
  "duration_ms": 12480,
  "num_frames_t": 624,
  "forward_ms": 5.23,
  "post_ms": 0.01,
  "dp_ms": 0.18,
  "group_ms": 0.04,
  "conf_ms": 0.01,
  "align_ms": 0.23,
  "total_ms": 5.47,
  "forward_ms_repeats": [5.1, 5.3, ...],
  "memory": { ... }
}
```

The Python side also records `forward_ms_repeats`, `post_ms_repeats`, etc. for all timed stages.

### Benchmark configuration defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup` | 10 | Warm-up iterations (first utterance only) to stabilize GPU caches and JIT |
| `repeats` | 30 | Timed repeats per utterance |
| `aggregate` | median | Aggregation mode (median is more robust to outliers than mean) |

---

## Known biases and limitations

The benchmarking infrastructure on both sides was developed rapidly ("vibe coded") and should be taken as a best-effort comparison rather than a rigorous benchmark. The following biases are known:

### 1. The entire benchmarking stack was vibe coded

Both the Rust profiling infrastructure (`alignment-profiling` feature, `perf_report_formatter.rs`) and the Python instrumentation (`wav2vec2aligner-main.perf-monitoring.patch`, `scripts/wav2vec2aligner_librispeech_textgrids.py`) were developed quickly with the primary goal of getting directionally useful numbers. Edge cases in timing, synchronization, or aggregation may exist.

Anyone with benchmarking expertise is warmly invited to review and improve the methodology. Pull requests that address measurement biases or add more rigorous instrumentation are very welcome.

### 2. Time breakdown absolute chart may be misleading

One of the generated visualizations — the **absolute time breakdown chart** — appears to show Python winning in scenarios where the raw per-utterance data indicates otherwise. This is suspected to be a **representation/aggregation issue** in the charting code rather than a genuine result, but the root cause has not been fully diagnosed. When in doubt, refer to the raw JSONL records rather than aggregate charts.

### 3. Memory measurement for the wgpu (generic GPU) backend is incomplete

GPU memory usage for the **wgpu backend** could not be reliably measured. The wgpu API does not expose `cudaMemGetInfo`-equivalent queries, and the underlying GPU driver memory is not easily attributed to a specific wgpu device. As a result, memory comparisons are only available for the CUDA backend (Rust) vs. PyTorch CUDA (Python). The wgpu backend's memory column may show zeros or be absent entirely.

### 4. Implementation differences

The two implementations differ in ways that affect what is being measured:

- **Model architecture**: The Rust side uses `wav2vec2-base-960h` via ONNX Runtime or Candle. The Python side uses `WAV2VEC2_ASR_BASE_960H` via TorchAudio. Same model, different runtimes with different optimization profiles.
- **Viterbi implementation**: The Rust side has a custom CTC Viterbi with reachability-band pruning and three backends. The Python side uses `torchaudio.functional.forced_align`. These are algorithmically equivalent but may have different constant factors.
- **Post-processing**: The Rust side has multi-policy blank expansion with candidate scoring. The Python side uses simpler frame-based grouping. This means `group_ms` and `conf_ms` are not directly comparable — the Rust version does more work in these stages.
- **Synchronization**: Both sides call `torch.cuda.synchronize()` / device sync around timed stages. However, subtle differences in when asynchronous GPU work completes may shift time between stages (e.g., a kernel launched in `forward` might finish during `post`).

### 5. These are indicators, not absolutes

The benchmarks are best used to confirm that both implementations are **in the same order of magnitude** for the same task, and to identify which pipeline stages dominate. They should not be used for precise speed-up claims without addressing the biases above.

---

## Scaling analysis

The Rust benchmark supports a `--perf-scaling-report` flag that prints per-utterance T, S, T×S values alongside dp_ms, and computes the Pearson correlation between T×S and dp_ms. This is useful for verifying that the Viterbi DP scales as expected (linearly in T×S with reachability pruning) and for identifying outlier utterances.

```bash
cargo run --release --features "onnx,cuda-dp,alignment-profiling,report-cli" \
    --bin alignment_report -- \
    --model-dir models/onnx_wav2vec2_base_960h \
    --dataset-root test-data \
    --device cuda \
    --runtime onnx \
    --output-format perf \
    --perf-out target/perf/rust-scaling.jsonl \
    --perf-append \
    --perf-scaling-report
```

---

## Contributing

If you notice measurement biases, synchronization issues, or have suggestions for more rigorous benchmarking practices, contributions are very welcome. The areas that would benefit most from community review are:

- Verifying CUDA synchronization correctness around timed stages (both Rust and Python)
- Investigating the absolute time breakdown chart discrepancy
- Adding wgpu memory measurement (perhaps via Vulkan memory budget extensions)
- Cross-validating aggregate statistics against raw per-utterance data
- Adding statistical significance testing (confidence intervals, effect sizes)

Please open an issue or pull request with your findings.
