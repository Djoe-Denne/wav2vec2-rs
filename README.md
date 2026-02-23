# wav2vec2-rs

## Prepare Test Data

This project uses LibriSpeech test sets as reference test data.
The setup script downloads archives from OpenSLR, extracts them, and generates
reference word alignments with a PyTorch `Wav2Vec2ForCTC` model.

By default, it loads the local model from `models/wav2vec2-base-960h`.

### 1) Install Python dependencies

CUDA 12.8 (priority):

```bash
pip install -r scripts/requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
```

CUDA 13.0:

```bash
pip install -r scripts/requirements.txt --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

CPU fallback:

```bash
pip install -r scripts/requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

### 2) Run setup script

```bash
python scripts/pytorch_aligner.py
```

The script will:

- download `test-clean.tar.gz` and `test-other.tar.gz` from OpenSLR
- extract data to `test-data/LibriSpeech/`
- generate alignment references:
  - `test-data/alignments/test-clean.json`
  - `test-data/alignments/test-other.json`
- run emissions with PyTorch Wav2Vec2 (`Wav2Vec2ForCTC`)
- decode FLAC audio with `soundfile` (no TorchCodec requirement)

### 3) Optional flags

Regenerate everything:

```bash
python scripts/pytorch_aligner.py --overwrite
```

Run a subset only:

```bash
python scripts/pytorch_aligner.py --subsets test-clean
```

Limit aligned utterances per subset:

```bash
python scripts/pytorch_aligner.py --limit 100
```

Use a different local model directory:

```bash
python scripts/pytorch_aligner.py --model-dir models/wav2vec2-base-960h
```
