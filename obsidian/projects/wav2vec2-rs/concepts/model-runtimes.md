---
title: wav2vec2-rs Model Runtimes
category: concepts
tags:
  - wav2vec2
  - candle
  - onnx
  - wav2vec2-rs
summary: wav2vec2-rs has separate Candle and ONNX Runtime backends; Candle loads safetensors into local Rust model code, while ONNX can use CUDA EP.
updated: 2026-04-26T16:10:03+02:00
sources:
  - README.md
  - src/pipeline/model_runtime.rs
  - src/model/ctc_model.rs
  - src/model/encoder.rs
  - src/model/feature_extractor.rs
  - src/model/feature_projection.rs
  - src/model/layers.rs
provenance:
  source_type: source_code
  extracted: 0.9
  inferred: 0.1
  ambiguous: 0.0
---

# wav2vec2-rs Model Runtimes

`wav2vec2-rs` has two runtime kinds: `Candle` and `Onnx`. The builder defaults to `RuntimeKind::Candle`; callers can choose ONNX explicitly through `with_runtime_kind(RuntimeKind::Onnx)` or through the report CLI runtime choice.

## Candle Runtime

The Candle runtime is implemented in `src/pipeline/model_runtime.rs` as `CandleRuntimeBackend`. It reads safetensors from `Wav2Vec2Config.model_path`, builds a Candle `VarBuilder`, constructs `Wav2Vec2ForCTC`, runs forward inference, applies log-softmax, and returns host log probabilities.

`Wav2Vec2ForCTC` is implemented in `src/model/ctc_model.rs`. It chains `FeatureExtractor`, `FeatureProjection`, `Encoder`, and an LM head. That confirms the README's Candle statement is accurate for the current source tree.

## ONNX Runtime

The ONNX runtime is feature-gated behind `onnx`. It builds an ORT session from `model.onnx`, supports CPU and CUDA execution providers, and falls back to CPU EP after CUDA EP in the provider list. With `onnx` plus `cuda-dp`, ONNX CUDA output may become `ForwardOutput::CudaDevice` for zero-copy Viterbi.

## Runtime Boundary

Candle and ONNX are model inference backends. They do not replace tokenization, CTC Viterbi, grouping, or reporting; those remain pipeline stages described in [[wav2vec2-rs Forced Alignment Pipeline]].

## Links

- [[wav2vec2-rs]]
- [[wav2vec2-rs Forced Alignment Pipeline]]
- [[wav2vec2-rs CTC Viterbi Backends]]
- [[Exporting Wav2Vec2 CTC Models To ONNX]]
