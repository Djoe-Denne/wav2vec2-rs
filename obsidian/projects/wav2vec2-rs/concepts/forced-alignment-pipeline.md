---
title: wav2vec2-rs Forced Alignment Pipeline
category: concepts
tags:
  - forced-alignment
  - ctc
  - rust
  - wav2vec2-rs
summary: Runtime flow for wav2vec2-rs from audio input through model inference, CTC Viterbi decoding, grouping, and profiled output.
updated: 2026-04-26T16:10:03+02:00
sources:
  - README.md
  - src/pipeline/runtime.rs
  - src/types.rs
  - src/config.rs
provenance:
  source_type: source_code
  extracted: 0.85
  inferred: 0.15
  ambiguous: 0.0
---

# wav2vec2-rs Forced Alignment Pipeline

The main runtime is `ForcedAligner` in `src/pipeline/runtime.rs`. It owns a runtime backend, vocabulary, blank and word-separator IDs, frame stride, expected sample rate, tokenizer, sequence aligner, and word grouper.

## Runtime Flow

1. `align` rejects empty audio or blank transcript by returning an empty `AlignmentOutput`.
2. The input sample rate is checked against `expected_sample_rate_hz`; a mismatch warns rather than immediately failing.
3. Audio is normalized unless `AlignmentInput.normalized` already provides a reusable normalized buffer.
4. `RuntimeBackend::infer` returns `ForwardOutput`, either host log probabilities or CUDA-device log probabilities.
5. `Tokenizer::tokenize` converts the transcript and vocabulary into a CTC `TokenSequence`.
6. The runtime checks that the model produced enough frames for the blank-interleaved token sequence.
7. `dispatch_viterbi` creates the best CTC state path.
8. `WordGrouper::group_words` converts the path into `WordTiming` values.

## Profiling Flow

`align_profiled` records stage timing fields for forward, post-processing, DP, grouping, confidence, alignment, and total time. With `alignment-profiling`, `align_profiled_with_memory` can also collect per-stage memory through `MemoryTracker`.

## Links

- [[wav2vec2-rs]]
- [[wav2vec2-rs Builder Component Architecture]]
- [[wav2vec2-rs Model Runtimes]]
- [[wav2vec2-rs CTC Viterbi Backends]]
- [[wav2vec2-rs Tokenization And Word Grouping]]
