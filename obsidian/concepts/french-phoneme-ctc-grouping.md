---
title: French Phoneme CTC Grouping
category: concepts
tags:
  - ctc
  - phonemes
  - forced-alignment
  - wav2vec2-rs
summary: French phoneme CTC support is modeled as switchable grouping behavior while keeping Viterbi token decoding unchanged.
updated: 2026-04-26T16:10:03+02:00
sources:
  - C:/Users/djden/.cursor/projects/c-Users-djden-source-repos-wav2vec2-rs
  - README.md
  - src/alignment/viterbi.rs
  - src/alignment/grouping/mod.rs
  - src/alignment/grouping/path_to_words.rs
  - src/pipeline/builder.rs
  - src/pipeline/traits.rs
provenance:
  source_type: cursor_conversations_and_source
  extracted: 0.45
  inferred: 0.55
  ambiguous: 0.0
---

# French Phoneme CTC Grouping

French phoneme CTC support in [[wav2vec2-rs]] is treated as a grouping concern: the Viterbi stage decodes the most likely token path, while grouping decides whether path spans become words or phoneme units. ^[inferred]

## Source-Backed Context

`src/alignment/viterbi.rs` operates on token IDs and CTC state transitions; it does not inspect words. `src/alignment/grouping/path_to_words.rs` is the word-specific layer: it tracks separators, expected normalized words, current word text, emission statistics, and flush boundaries.

`src/pipeline/traits.rs` exposes grouping behind `WordGrouper`, which allows alternate grouping behavior to be injected without changing the runtime path. The current builder defaults to `DefaultWordGrouper`; source ingestion does not show a committed phoneme-mode builder API yet.

## Design Notes

- Viterbi is token-agnostic and does not need word-specific changes for phoneme CTC models. ^[inferred]
- Word grouping depends on separators, expected words, and word-level blank expansion; phoneme grouping would emit timing units for non-blank, non-separator token transitions. ^[inferred]
- A builder-level grouping mode would keep [[wav2vec2-rs Forced Alignment Pipeline]] stable while changing only the injected grouping behavior. ^[inferred]
- Explicit custom grouper injection should keep precedence over default mode selection. ^[inferred]

## Related Implementation Areas

- [[wav2vec2-rs CTC Viterbi Backends]]
- [[wav2vec2-rs Tokenization And Word Grouping]]
- [[wav2vec2-rs Builder Component Architecture]]
- [[wav2vec2-rs Model Runtimes]]
