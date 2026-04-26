---
title: wav2vec2-rs
category: projects
tags:
  - project/wav2vec2-rs
  - ml/forced-alignment
  - rust
summary: Rust CTC forced-alignment crate with pluggable runtimes, CTC Viterbi backends, grouping, reporting, and a tracked runtime roadmap.
updated: 2026-04-26T17:45:00+02:00
sources:
  - C:/Users/djden/.cursor/projects/c-Users-djden-source-repos-wav2vec2-rs
  - README.md
  - src/lib.rs
  - src/pipeline/runtime.rs
  - src/pipeline/builder.rs
  - src/pipeline/model_runtime.rs
  - src/alignment/viterbi.rs
  - src/alignment/grouping/mod.rs
provenance:
  source_type: cursor_conversations_and_source
  extracted: 0.75
  inferred: 0.25
  ambiguous: 0.0
---

# wav2vec2-rs

`wav2vec2-rs` is a Rust forced-alignment project for CTC acoustic models. It maps a known transcript onto audio and returns word-level timing boundaries with confidence scores.

## Architecture Map

- [[wav2vec2-rs Forced Alignment Pipeline]] describes the runtime flow from `AlignmentInput` through audio normalization, model inference, tokenization, Viterbi decoding, and grouping.
- [[wav2vec2-rs Builder Component Architecture]] describes the trait-based extension points: `RuntimeBackend`, `Tokenizer`, `SequenceAligner`, and `WordGrouper`.
- [[wav2vec2-rs Model Runtimes]] records the verified runtime split: Candle loads safetensors into a local `Wav2Vec2ForCTC`, while ONNX Runtime loads `model.onnx` and can use CUDA execution providers.
- [[wav2vec2-rs CTC Viterbi Backends]] covers the CPU, wgpu, and CUDA DP implementations.
- [[wav2vec2-rs Tokenization And Word Grouping]] covers case-aware CTC token construction, raw word collection, blank expansion, candidate selection, and confidence calibration.
- [[wav2vec2-rs Reports And Benchmarks]] covers `alignment_report`, TextGrid output, JSON quality reports, perf JSONL, and benchmark caveats.
- [[wav2vec2-rs Roadmap]] tracks follow-up work, including the FP16 ONNX CUDA limitation found in local runtime tests.

## Source Verification Notes

The README's claim that the crate has a Candle implementation is source-backed: `src/pipeline/model_runtime.rs` defines `CandleRuntimeBackend`, loads safetensors with Candle `VarBuilder`, and constructs the local `Wav2Vec2ForCTC` from `src/model/ctc_model.rs`. This should be described as a real Candle runtime, not merely planned documentation. The ONNX runtime is a separate backend selected by `RuntimeKind::Onnx` or the report CLI runtime choice.

French phoneme work remains related but separate: [[French Phoneme CTC Grouping]] is mainly about grouping behavior and pipeline selection, while the base architecture remains CTC-token based. ^[inferred]

## Current Runtime Note

Local model/runtime testing found that FP32 ONNX works with CUDA, while the tested FP16 ONNX model works on CPU but fails in the CUDA provider before logits are produced. The roadmap records this as follow-up work rather than treating FP16 CUDA as supported.

See [[Cursor Conversation Import]] for the conversation inventory used before source ingestion.
