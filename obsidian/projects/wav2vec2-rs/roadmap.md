---
title: wav2vec2-rs Roadmap
category: projects
tags:
  - project/wav2vec2-rs
  - roadmap
  - onnx
  - cuda
summary: Follow-up work for wav2vec2-rs, including FP16 ONNX CUDA and next steps after the Candle CUDA smoke test.
created: 2026-04-26T17:45:00+02:00
updated: 2026-04-26T18:18:00+02:00
sources:
  - Cargo.toml
  - target/candle-test-clean-10-perf.json
  - target/candle-test-clean-1-cuda-perf.json
  - target/reports/fp16-test-clean-cpu.json
  - target/reports/fp32-test-clean-cuda.json
  - target/reports/fp16-test-clean-cuda-rust-error.txt
  - projects/wav2vec2-rs/references/fp16-onnx-cuda-incident.md
provenance:
  source_type: local_experiment
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# wav2vec2-rs Roadmap

This page tracks follow-up work for [[wav2vec2-rs]] that should not be forgotten after local experiments.

## Runtime And Precision

- [x] Smoke-test Candle CUDA model inference.
  - Current finding: Candle `0.10.2` builds with CUDA 13.2 on Windows when `CL=/Zc:preprocessor` is set.
  - A one-case LibriSpeech `test-clean` report completed with `--runtime candle --device cuda`, reporting `dtype = f32` and `device = cuda`.
  - Remaining work: run a larger benchmark and decide whether a Candle device-buffer path is worth implementing, because the current Candle runtime still returns host log-probs before Viterbi. ^[inferred]

- [x] Add FP16 ONNX CUDA mitigation for wav2vec2 positional convolution.
  - Current finding: the tested FP16 ONNX model runs on CPU but fails on CUDA before logits are produced.
  - Incident cause: [[wav2vec2-rs FP16 ONNX CUDA Incident]] records the ONNX Runtime/cuDNN frontend engine-selection failure on the wav2vec2 positional convolution.
  - Implemented mitigation: the ONNX exporter supports `--precision cuda-safe-fp16`, which keeps Rust-facing input/output tensors in FP32, lowers most model compute to FP16, and keeps the positional convolution in FP32.
  - Runtime behavior: ONNX output extraction accepts `f32`, `f16`, `bf16`, and `f64` CPU-accessible logits, converting them to host `f32` log-probs; CUDA zero-copy remains gated to `f32` logits. ^[inferred]

## Related Notes

- [[wav2vec2-rs Model Runtimes]]
- [[wav2vec2-rs FP16 ONNX CUDA Incident]]
- [[wav2vec2-rs Reports And Benchmarks]]
- [[wav2vec2-rs CTC Viterbi Backends]]
