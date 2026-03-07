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

## Multilingual and other test data

Additional benchmark data can be placed under **`test-data/`** using the same layout as LibriSpeech: a dataset-specific folder containing **subset folders**, each with `*.trans.txt` (format: `utt_id transcript`) and `{utt_id}.flac` (or `.wav`) audio files. This allows the same benchmark tooling to run on multilingual or accent-specific corpora.

### VoxPopuli (transcribed ASR, per language)

**Source:** [facebookresearch/voxpopuli](https://github.com/facebookresearch/voxpopuli) (repository archived; instructions still valid).

VoxPopuli provides transcribed speech for 16 languages (e.g. English, German, French, Spanish, Polish, Italian, Romanian, Hungarian, Czech, Dutch, Finnish, Croatian, Slovak, Slovene, Estonian, Lithuanian). Data is European Parliament recordings (2009–2020).

**Download steps (per language code, e.g. `en`, `de`, `fr`):**

1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/facebookresearch/voxpopuli.git
   cd voxpopuli
   pip install -r requirements.txt
   ```

2. Download raw ASR audios (shared across languages):
   ```bash
   python -m voxpopuli.download_audios --root [ROOT] --subset asr
   ```
   By default this downloads all years (2009–2020). To download only the latest year (much smaller, one archive):
   ```bash
   python -m voxpopuli.download_audios --root [ROOT] --subset asr --years latest
   ```
   This saves to `[ROOT]/raw_audios/original/[year]/[recording_id].ogg`.

3. Segment and align for a specific language:
   ```bash
   python -m voxpopuli.get_asr_data --root [ROOT] --lang [LANGUAGE]
   ```
   Output:
   - Audios: `[ROOT]/transcribed_data/[language]/[year]/[segment_id].ogg`
   - Manifests: `[ROOT]/transcribed_data/[language]/asr_train.tsv`, `asr_dev.tsv`, `asr_test.tsv` (columns: ID, transcript, speaker ID).

VoxPopuli uses **Ogg** and **TSV**, not the `.trans.txt` + `.flac` layout. To use it with the same benchmark tooling, convert the TSV manifests and Ogg files into subset dirs under `test-data/VoxPopuli/<lang>/<split>/` with `.trans.txt` and `.flac` (e.g. resample to 16 kHz and write one `.trans.txt` per split and one `{id}.flac` per utterance). Use the script **`scripts/convert_voxpopuli_to_librispeech_layout.py`** after running the voxpopuli download and segment steps: `--voxpopuli-root [ROOT] --lang fr --output-dir test-data`. Note: the VoxPopuli repo is archived and its `download_audios` script may require an older `torchaudio` (e.g. `download_url` was removed in newer versions); you may need to patch it or use a separate downloader for the ASR tarballs.

**Target layout under test-data:** `test-data/VoxPopuli/<lang>/<split>/` (e.g. `train`, `dev`, `test`) with `.trans.txt` and `{id}.flac` after conversion.

### Multilingual LibriSpeech (Hugging Face)

**Source:** [facebook/multilingual_librispeech](https://huggingface.co/datasets/facebook/multilingual_librispeech).

Configs: `dutch`, `french`, `german`, `italian`, `polish`, `portuguese`, `spanish`. Splits: `train`, `dev`, `test`, `1_hours`, `9_hours`.

**Download via API:**
```python
from datasets import load_dataset
ds = load_dataset("facebook/multilingual_librispeech", "<config>", split="<split>")
# Example: load_dataset("facebook/multilingual_librispeech", "german", split="test")
```

**Conversion to LibriSpeech-like layout:** Use the script `scripts/export_hf_to_librispeech_layout.py` (see below). It writes `test-data/MultilingualLibriSpeech/<config>/<split>/` with `.trans.txt` and `{id}.flac` (16 kHz, mono).

### African Accented French (Hugging Face)

**Source:** [gigant/african_accented_french](https://huggingface.co/datasets/gigant/african_accented_french).

~22 hours of French speech (Cameroon, Gabon, Niger). Splits: `train`, `test`.

**Download via API:**
```python
from datasets import load_dataset
ds = load_dataset("gigant/african_accented_french", split="train")  # or "test"
```

**Conversion to LibriSpeech-like layout:** Use the same script `scripts/export_hf_to_librispeech_layout.py`. It writes `test-data/AfricanAccentedFrench/train/` and `test-data/AfricanAccentedFrench/test/` with `.trans.txt` and `{id}.flac`.

### Export script for Hugging Face datasets (de-parquet)

The script **`scripts/export_hf_to_librispeech_layout.py`** loads a Hugging Face dataset (e.g. Multilingual LibriSpeech or African Accented French) and exports it into the same layout as LibriSpeech under `test-data/`:

- **Multilingual LibriSpeech:** `--dataset facebook/multilingual_librispeech --config <lang> --splits test --output-dir test-data` → `test-data/MultilingualLibriSpeech/<config>/test/`.
- **African Accented French:** `--dataset gigant/african_accented_french --splits test --output-dir test-data --trust-remote-code` → `test-data/AfricanAccentedFrench/test/`.

See the script’s `--help` and docstring for options (`--output-dir`, `--limit`, etc.). Dependencies: `datasets`, `soundfile`, `librosa` (for audio decode with datasets 3.x); use a venv and `pip install datasets soundfile librosa scipy`.

### Recreating test data (after deleting)

If you delete all benchmark data and want to recreate it from scratch:

1. **LibriSpeech (English):** Download test-clean and test-other from [OpenSLR 12](https://www.openslr.org/12), extract so you have `test-data/LibriSpeech/test-clean` and `test-data/LibriSpeech/test-other`.

2. **Multilingual LibriSpeech (e.g. French test):**
   ```bash
   python scripts/export_hf_to_librispeech_layout.py --dataset facebook/multilingual_librispeech --config french --splits test --output-dir test-data
   ```
   Output: `test-data/MultilingualLibriSpeech/french/test/`.

3. **African Accented French (test):**
   ```bash
   python scripts/export_hf_to_librispeech_layout.py --dataset gigant/african_accented_french --splits test --output-dir test-data --trust-remote-code
   ```
   Output: `test-data/AfricanAccentedFrench/test/`.

4. **VoxPopuli (e.g. French):** Follow the VoxPopuli subsection above (clone repo, `download_audios --subset asr`, `get_asr_data --lang fr`; note torchaudio compatibility). Then run `scripts/convert_voxpopuli_to_librispeech_layout.py --voxpopuli-root [ROOT] --lang fr --output-dir test-data`.

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

## Benchmark assets in CI (GitHub Actions)

Benchmark data (LibriSpeech, Python reference TextGrids, and the model) are prepared and cached by manual workflows so that GPU benchmarks can run without re-downloading on every run.

### How to trigger the workflows

Both workflows use **manual dispatch** (`workflow_dispatch`). They do not run on push or pull request.

**In the GitHub web UI:**

1. Open your repo on GitHub.
2. Go to the **Actions** tab.
3. In the left sidebar, click **Prepare benchmark assets** or **GPU manual (benchmark)**.
4. Click the **Run workflow** dropdown (top right).
5. Choose the branch to run on (usually `main`), set the inputs (checkboxes and model options), then click the green **Run workflow** button.

**From the command line** (GitHub CLI):

```bash
# Prepare all three caches (dataset, model, Python reference)
gh workflow run "Prepare benchmark assets" --ref main \
  -f prepare_dataset=true \
  -f prepare_python_reference=true \
  -f prepare_model=true \
  -f model_source=hf \
  -f model_repo_id=your-org/your-model \
  -f model_revision=main \
  -f model_allow_patterns="*.onnx,config.json,vocab.json"

# After caches are ready, run the GPU benchmark
gh workflow run "GPU manual (benchmark)" --ref main \
  -f model_source=hf \
  -f model_repo_id=your-org/your-model \
  -f model_revision=main \
  -f model_allow_patterns="*.onnx,config.json,vocab.json"
```

Replace `your-org/your-model` and the branch name as needed. The model inputs must match between the two runs so the same cache key is used.

### 1. Prepare benchmark assets

Run the **Prepare benchmark assets** workflow to populate caches:

- **LibriSpeech** — Downloads test-clean and test-other from OpenSLR 12 into `test-data/LibriSpeech` (cache key: `librispeech-slr12-test-v1`).
- **Python reference TextGrids** — Generates reference alignments with wav2vec2aligner and stores them under `ci-assets/python-reference-textgrids` (cache key includes script + patch hash).
- **Model** — Downloads from Hugging Face (or an archive URL) into `ci-assets/model` (cache key is a hash of source/repo/revision/patterns or archive URL).

You can run one, two, or all three steps in a single workflow run. Use the same **model inputs** (repo id, revision, allow patterns, or archive URL) when you later run the GPU manual workflow so that the model cache key matches.

### 2. GPU manual (benchmark)

Run the **GPU manual (benchmark)** workflow (`Actions → GPU manual (benchmark) → Run workflow`) after at least the **model** cache has been prepared. Set the model inputs to the same values you used in Prepare benchmark assets (same `model_repo_id`, `model_revision`, `model_allow_patterns`, or `model_archive_url`). The workflow restores the three caches, builds the Rust binary with GPU features, and runs the alignment benchmark with `--model-dir ci-assets/model` and `--dataset-root test-data`. If the model cache is missing, the job fails with a message to run Prepare benchmark assets first.

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

### Rust TextGrid-only (classic mode)

To generate TextGrid files once (no perf measurement), use `--output-format textgrid` and an optional suffix so outputs do not overwrite existing TextGrids (e.g. from the Python aligner):

```bash
# Build (no alignment-profiling needed)
cargo build --release --bin alignment_report --features "onnx,report-cli"

# African Accented French — writes e.g. test_000000_rust.TextGrid
cargo run --release --bin alignment_report --features "onnx,report-cli" -- \
  --model-dir models/onnx_wav2vec2_base_960h \
  --dataset-root test-data/AfricanAccentedFrench \
  --output-format textgrid --textgrid-suffix _rust --device cpu --runtime onnx

# Multilingual LibriSpeech French — writes e.g. 10179_11051_000000_rust.TextGrid
cargo run --release --bin alignment_report --features "onnx,report-cli" -- \
  --model-dir models/onnx_wav2vec2_base_960h \
  --dataset-root test-data/MultilingualLibrispeech/french \
  --output-format textgrid --textgrid-suffix _rust --device cpu --runtime onnx
```

Requires ONNX Runtime ≥ 1.23 (the `ort` crate will report a version mismatch if an older system DLL is loaded).

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
