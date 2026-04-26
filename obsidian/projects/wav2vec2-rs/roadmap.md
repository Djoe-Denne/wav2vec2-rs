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

- [ ] Investigate FP16 ONNX CUDA support for wav2vec2 positional convolution.
  - Current finding: the tested FP16 ONNX model runs on CPU but fails on CUDA before logits are produced.
  - Incident cause: [[wav2vec2-rs FP16 ONNX CUDA Incident]] records the ONNX Runtime/cuDNN frontend engine-selection failure on the wav2vec2 positional convolution.
  - Current practical path: use FP32 ONNX with CUDA for fast inference.
  - Candidate approaches: test another ONNX Runtime/CUDA/cuDNN stack, export a mixed-precision ONNX model that keeps the problematic convolution in FP32, or patch the ONNX graph to avoid the failing CUDA provider path. ^[inferred]

## Related Notes

- [[wav2vec2-rs Model Runtimes]]
- [[wav2vec2-rs FP16 ONNX CUDA Incident]]
- [[wav2vec2-rs Reports And Benchmarks]]
- [[wav2vec2-rs CTC Viterbi Backends]]
