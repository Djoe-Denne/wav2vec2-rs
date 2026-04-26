---
title: Exporting Wav2Vec2 CTC Models To ONNX
category: skills
tags:
  - onnx
  - wav2vec2
  - python
  - model-export
summary: Export Wav2Vec2 CTC models with exporter selection that avoids noisy failed opset down-conversion and validates the output.
sources:
  - C:\Users\djden\.cursor\projects\c-Users-djden-source-repos-wav2vec2-rs
provenance:
  source_type: cursor_conversations
  confidence: medium
---

# Exporting Wav2Vec2 CTC Models To ONNX

The project exports Hugging Face Wav2Vec2 CTC models through `scripts/export_ctc_model_to_onnx.py`. Cursor conversations captured a recurring issue where the modern PyTorch ONNX exporter may export at a newer opset and then fail down-conversion, or fail dynamic shape constraints before falling back to the legacy exporter. ^[inferred]

## Practical Guidance

- Treat `ONNX export complete` as the success signal, but validate the produced file with `onnx.checker` or runtime loading before trusting it. ^[inferred]
- For opset values below 18, the legacy exporter is often the cleaner direct path because it avoids a noisy attempted conversion from a newer opset. ^[inferred]
- For opset 18 or newer, the dynamo exporter can be attempted, but fallback to the legacy exporter is useful when dynamic shape constraints fail. ^[inferred]
- Missing CUDA availability should fall back to CPU export rather than blocking model export. ^[inferred]

## Related Files

- `scripts/export_ctc_model_to_onnx.py`
- `models/phonemizer-wav2vec2-ctc-french-onnx/model.onnx`
