---
title: wav2vec2-rs Reports And Benchmarks
category: concepts
tags:
  - benchmarking
  - reports
  - textgrid
  - wav2vec2-rs
summary: Reporting code produces JSON quality reports, TextGrid output, and perf measurements over LibriSpeech-like datasets.
updated: 2026-04-26T16:10:03+02:00
sources:
  - README.md
  - src/alignment/report.rs
  - src/bin/alignment_report.rs
  - src/bin/alignment_report/json_report_formatter.rs
  - src/bin/alignment_report/text_grid_report_formatter.rs
  - src/bin/alignment_report/perf_report_formatter.rs
provenance:
  source_type: source_code
  extracted: 0.8
  inferred: 0.2
  ambiguous: 0.0
---

# wav2vec2-rs Reports And Benchmarks

`alignment_report` is both a quality-reporting CLI and a benchmark binary. It reads LibriSpeech-style datasets, runs forced alignment, compares predictions to optional TextGrid references, and writes JSON, TextGrid, or perf output.

## Library Report Model

`src/alignment/report.rs` defines serializable report types: `Report`, `SentenceReport`, structural metrics, confidence metrics, timing metrics, aggregate distributions, threshold pass rates, and outlier lists.

## CLI Modes

The report CLI supports:

- JSON quality reports with structural, confidence, timing, aggregate, and outlier data.
- TextGrid generation with predicted word timing tiers.
- Perf output when `alignment-profiling` is enabled.

## Dataset Handling

The CLI recognizes `test-data/LibriSpeech` with `test-clean` and `test-other`, but it can also discover direct dataset roots whose child folders contain `*.trans.txt` files.

## Benchmark Interpretation

Benchmarks should be treated as implementation and stage diagnostics, not absolute speed claims. The README points to benchmark methodology and caveats; the source supports per-stage timings that make it possible to isolate forward, post, DP, grouping, and confidence costs.

## Links

- [[wav2vec2-rs]]
- [[wav2vec2-rs Forced Alignment Pipeline]]
- [[wav2vec2-rs Model Runtimes]]
- [[Cursor Conversation Import]]
