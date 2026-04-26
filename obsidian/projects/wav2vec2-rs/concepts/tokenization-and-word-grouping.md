---
title: wav2vec2-rs Tokenization And Word Grouping
category: concepts
tags:
  - tokenization
  - grouping
  - confidence
  - wav2vec2-rs
summary: Case-aware tokenization emits CTC tokens, while grouping turns Viterbi paths into word timings through raw collection, blank expansion, selection, and calibration.
updated: 2026-04-26T16:10:03+02:00
sources:
  - README.md
  - src/alignment/tokenization.rs
  - src/alignment/grouping/mod.rs
  - src/alignment/grouping/path_to_words.rs
  - src/alignment/grouping/blank_expansion.rs
  - src/alignment/grouping/candidate_selector.rs
  - src/types.rs
provenance:
  source_type: source_code
  extracted: 0.85
  inferred: 0.15
  ambiguous: 0.0
---

# wav2vec2-rs Tokenization And Word Grouping

Tokenization and grouping are the bridge between transcript text and user-visible word timings.

## Tokenization

`build_token_sequence_case_aware` inspects the vocabulary casing and normalizes the transcript. It emits a blank-initial, blank-interleaved token sequence and keeps a parallel `chars` array. Unknown characters are skipped, and `normalized_words` records the words that actually emitted tokens.

## Grouping

`group_into_words_profiled` orchestrates three phases:

1. `path_to_words::collect_profiled` walks the Viterbi path, gathers raw word spans, emission log probabilities, top-2 margins, and coverage counts.
2. `blank_expansion` produces multiple boundary candidates around raw word spans.
3. `candidate_selector` chooses a candidate using boundary blank evidence, boundary shift penalty, and pause plausibility.

## Confidence

Each `WordTiming` carries a deterministic confidence score. The score blends geometric mean probability, margin, p10 log-probability, and boundary evidence, then applies piecewise calibration to produce `[0, 1]` output.

## Phoneme Implication

Current word grouping is word-separator and expected-word aware. A phoneme grouping mode should bypass word-boundary assumptions while preserving the same token-path and confidence ideas where applicable. ^[inferred]

## Links

- [[wav2vec2-rs]]
- [[wav2vec2-rs Forced Alignment Pipeline]]
- [[wav2vec2-rs CTC Viterbi Backends]]
- [[French Phoneme CTC Grouping]]
