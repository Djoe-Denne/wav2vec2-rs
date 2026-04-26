---
title: wav2vec2-rs FP16 ONNX CUDA Incident
category: references
tags:
  - project/wav2vec2-rs
  - onnx
  - cuda
  - incident
summary: FP16 ONNX inference fails on CUDA because ONNX Runtime/cuDNN cannot select a valid engine for the wav2vec2 positional convolution.
created: 2026-04-26T18:01:00+02:00
updated: 2026-04-26T18:01:00+02:00
sources:
  - target/reports/fp16-test-clean-cuda-rust-error.txt
  - target/reports/fp16-test-clean-cpu.json
  - target/reports/fp32-test-clean-cuda.json
provenance:
  source_type: local_experiment
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# wav2vec2-rs FP16 ONNX CUDA Incident

The tested FP16 ONNX `wav2vec2-base-960h` model is a valid model artifact: it runs on CPU and returns FP32 logits. The failure appears only when ONNX Runtime executes the graph with the CUDA provider.

## Observed Behavior

- FP16 ONNX on CPU: runs successfully.
- FP32 ONNX on CUDA: runs successfully and is the current fast path.
- FP16 ONNX on CUDA: fails before logits are produced.

## Cause

The CUDA failure occurs inside ONNX Runtime's CUDA provider while running the wav2vec2 encoder positional convolution node:

```text
/wav2vec2/encoder/pos_conv_embed/conv/Conv
```

The provider delegates this FP16 convolution to cuDNN, but cuDNN cannot find an implementable frontend engine for the shape/layout used by the graph:

```text
CUDNN_FE failure 8: HEURISTIC_QUERY_FAILED
No valid engine configs for ConvFwd_
```

The error context shows FP16 tensors around a large padded positional convolution, including half-precision input/output/intermediate data and a weight shape reported as `[768, 48, 128, 1]`. This points to an ONNX Runtime/cuDNN provider limitation for this exported graph rather than an alignment, TextGrid, or Rust extraction bug. ^[inferred]

## Impact

The current practical runtime guidance is to use FP32 ONNX with CUDA for fast inference and treat the tested FP16 ONNX model as CPU-runnable only in this environment.

## Follow-Up

Mitigation work is tracked in [[wav2vec2-rs Roadmap]]. The current exporter supports `--precision cuda-safe-fp16`, which lowers most ONNX model compute to FP16 while keeping the wav2vec2 positional convolution in FP32 and preserving FP32 graph input/output for the Rust runtime. Further work could still test other ONNX Runtime/CUDA/cuDNN stacks. ^[inferred]

## Related Notes

- [[wav2vec2-rs Roadmap]]
- [[wav2vec2-rs Model Runtimes]]
- [[wav2vec2-rs Reports And Benchmarks]]
