---
title: wav2vec2-rs CTC Viterbi Backends
category: concepts
tags:
  - ctc
  - viterbi
  - cuda
  - wav2vec2-rs
summary: CPU, wgpu, and CUDA implementations decode the best CTC state path; CUDA can operate on ONNX output without host log-prob copies.
updated: 2026-04-26T16:10:03+02:00
sources:
  - README.md
  - src/alignment/viterbi.rs
  - src/alignment/cuda/viterbi_cuda.rs
  - src/alignment/gpu/viterbi_gpu.rs
  - src/pipeline/cuda_forward.rs
provenance:
  source_type: source_code
  extracted: 0.85
  inferred: 0.15
  ambiguous: 0.0
---

# wav2vec2-rs CTC Viterbi Backends

The CTC Viterbi stage aligns a blank-interleaved token sequence against model log probabilities. Its output is a path of `(state, frame)` pairs that downstream grouping interprets.

## CPU Path

`forced_align_viterbi_cpu` is always available. It uses ping-pong score buffers, reachability-band pruning, and a compact backpointer table to recover the path.

## GPU Dispatch

`forced_align_viterbi` checks the `T x S` product and only attempts GPU backends above the launch-overhead threshold. With GPU features enabled, dispatch tries wgpu and CUDA paths before falling back to CPU.

## CUDA Zero-Copy Path

With `onnx` plus `cuda-dp`, `ForwardOutput::CudaDevice` can keep log probabilities on device. The runtime runs CUDA log-softmax and Viterbi before copying only the final state path back to host for grouping.

## Relation To Phoneme Grouping

The Viterbi implementation is token-path decoding and does not encode word semantics. That is why [[French Phoneme CTC Grouping]] belongs primarily in grouping/pipeline behavior, not in this DP layer. ^[inferred]

## Links

- [[wav2vec2-rs]]
- [[wav2vec2-rs Forced Alignment Pipeline]]
- [[wav2vec2-rs Model Runtimes]]
- [[wav2vec2-rs Tokenization And Word Grouping]]
