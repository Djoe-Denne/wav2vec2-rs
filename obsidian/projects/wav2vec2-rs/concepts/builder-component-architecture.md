---
title: wav2vec2-rs Builder Component Architecture
category: concepts
tags:
  - rust
  - architecture
  - forced-alignment
  - wav2vec2-rs
summary: Builder and trait design that lets wav2vec2-rs swap runtime, tokenizer, sequence aligner, and word grouper components.
updated: 2026-04-26T16:10:03+02:00
sources:
  - README.md
  - src/pipeline/builder.rs
  - src/pipeline/traits.rs
  - src/pipeline/defaults.rs
provenance:
  source_type: source_code
  extracted: 0.8
  inferred: 0.2
  ambiguous: 0.0
---

# wav2vec2-rs Builder Component Architecture

`ForcedAlignerBuilder` wires configuration into a `ForcedAligner`. It defaults to `RuntimeKind::Candle`, `CaseAwareTokenizer`, `ViterbiSequenceAligner`, and `DefaultWordGrouper`, while allowing callers to inject custom components.

## Extension Points

- `RuntimeBackend` performs model inference and reports whether output is host-resident or still on CUDA device.
- `Tokenizer` converts a transcript and vocab into a `TokenSequence`.
- `SequenceAligner` maps log probabilities and tokens to a Viterbi state path.
- `WordGrouper` converts that state path into `WordTiming` output, with an optional profiled path.

## Build-Time Behavior

The builder loads `Wav2Vec2ModelConfig`, computes frame stride from convolution stride and sample rate, loads `vocab.json`, derives `blank_id` from `pad_token_id`, and derives `word_sep_id` from the `|` vocabulary entry when present.

## Design Implication

The trait layout is why [[French Phoneme CTC Grouping]] can be modeled as an alternate grouping behavior without changing [[wav2vec2-rs CTC Viterbi Backends]]. ^[inferred]

## Links

- [[wav2vec2-rs]]
- [[wav2vec2-rs Forced Alignment Pipeline]]
- [[wav2vec2-rs Model Runtimes]]
- [[wav2vec2-rs Tokenization And Word Grouping]]
