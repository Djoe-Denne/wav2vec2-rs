---
title: wav2vec2-rs Model Runtimes
category: concepts
tags:
  - wav2vec2
  - candle
  - onnx
  - wav2vec2-rs
summary: wav2vec2-rs has Candle and ONNX Runtime backends; Candle CPU and a Candle CUDA smoke test work, while ONNX FP32 remains the validated zero-copy CUDA path.
updated: 2026-04-26T18:18:00+02:00
sources:
  - README.md
  - src/pipeline/model_runtime.rs
  - src/model/ctc_model.rs
  - src/model/encoder.rs
  - src/model/feature_extractor.rs
  - src/model/feature_projection.rs
  - src/model/layers.rs
  - Cargo.toml
  - target/candle-test-clean-10-perf.json
  - target/candle-test-clean-1-cuda-perf.json
  - target/reports/fp16-test-clean-cpu.json
  - target/reports/fp32-test-clean-cuda.json
  - target/reports/fp16-test-clean-cuda-rust-error.txt
provenance:
  source_type: source_code_and_local_experiment
  extracted: 0.85
  inferred: 0.1
  ambiguous: 0.05
---

# wav2vec2-rs Model Runtimes

`wav2vec2-rs` has two runtime kinds: `Candle` and `Onnx`. The builder defaults to `RuntimeKind::Candle`; callers can choose ONNX explicitly through `with_runtime_kind(RuntimeKind::Onnx)` or through the report CLI runtime choice.

## Candle Runtime

The Candle runtime is implemented in `src/pipeline/model_runtime.rs` as `CandleRuntimeBackend`. It reads safetensors from `Wav2Vec2Config.model_path`, builds a Candle `VarBuilder`, constructs `Wav2Vec2ForCTC`, runs forward inference, applies log-softmax, and returns host log probabilities.

`Wav2Vec2ForCTC` is implemented in `src/model/ctc_model.rs`. It chains `FeatureExtractor`, `FeatureProjection`, `Encoder`, and an LM head. That confirms the README's Candle statement is accurate for the current source tree.

Current validation covers Candle on CPU and a one-case Candle CUDA smoke test. A local 10-case LibriSpeech `test-clean` perf run using `models/candle_wav2vec2_base_960h/model.safetensors` completed on CPU and reported `dtype = f32`, `device = cpu`, and about 14.39 seconds of total library work.

After upgrading Candle to `0.10.2`, `candle-cuda = ["candle-core/cuda", "candle-nn/cuda"]` builds on the tested Windows CUDA 13.2 environment when `CL=/Zc:preprocessor` is set so NVCC invokes MSVC with the conforming preprocessor. A one-case report run completed with `--runtime candle --device cuda`, reporting `dtype = f32`, `device = cuda`, and about 1.97 seconds total for `1089-134686-0000`.

The current Candle CUDA path is not zero-copy: `CandleRuntimeBackend` still applies log-softmax and converts log-probs to host `Vec<Vec<f32>>` before Viterbi. ONNX CUDA plus `cuda-dp` remains the validated GPU path for keeping model output on device through CUDA Viterbi. ^[inferred]

## ONNX Runtime

The ONNX runtime is feature-gated behind `onnx`. It builds an ORT session from `model.onnx`, supports CPU and CUDA execution providers, and falls back to CPU EP after CUDA EP in the provider list. With `onnx` plus `cuda-dp`, ONNX CUDA output may become `ForwardOutput::CudaDevice` for zero-copy Viterbi.

## Precision Limitation

Model precision is a property of the model artifact, not a separate alignment-pipeline mode. A local comparison tested the FP16 ONNX `wav2vec2-base-960h` graph and the existing FP32 ONNX base model on a 10-case LibriSpeech `test-clean` subset.

- FP16 ONNX on CPU completed successfully and produced comparable timing quality to the FP32 run.
- FP32 ONNX on CUDA completed successfully, around an order of magnitude faster on the small subset.
- FP16 ONNX on CUDA failed inside ONNX Runtime's CUDA provider before logits were produced.

The Rust ONNX runtime now treats output precision explicitly. `f32` CUDA logits remain eligible for the zero-copy CUDA Viterbi path. `f16`, `bf16`, and `f64` logits are converted to host `f32` log-probs when ORT returns CPU-accessible output; non-`f32` CUDA-device logits produce a targeted error because the zero-copy CUDA log-softmax kernel currently reads `f32`.

For CUDA lower-precision artifacts, `scripts/export_ctc_model_to_onnx.py --precision cuda-safe-fp16` keeps wav2vec2 input/output tensors as `f32`, lowers most model compute to FP16, and keeps the positional convolution in FP32 to avoid the known ORT/cuDNN engine-selection failure. See [[wav2vec2-rs Roadmap]].

## Runtime Boundary

Candle and ONNX are model inference backends. They do not replace tokenization, CTC Viterbi, grouping, or reporting; those remain pipeline stages described in [[wav2vec2-rs Forced Alignment Pipeline]].

## Links

- [[wav2vec2-rs]]
- [[wav2vec2-rs Forced Alignment Pipeline]]
- [[wav2vec2-rs CTC Viterbi Backends]]
- [[wav2vec2-rs Roadmap]]
- [[Exporting Wav2Vec2 CTC Models To ONNX]]
