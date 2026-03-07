#!/usr/bin/env python3
"""
Convert VoxPopuli transcribed_data (TSV + Ogg) to LibriSpeech-like layout.

Run after you have run:
  python -m voxpopuli.download_audios --root [ROOT] --subset asr
  python -m voxpopuli.get_asr_data --root [ROOT] --lang fr

Then:
  python scripts/convert_voxpopuli_to_librispeech_layout.py \
    --voxpopuli-root test-data/voxpopuli_root \
    --lang fr \
    --output-dir test-data

Output: test-data/VoxPopuli/fr/{train,dev,test}/ with .trans.txt and .flac (16 kHz mono).

Dependencies: soundfile, scipy (optional for resampling).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VoxPopuli transcribed_data to LibriSpeech-like layout.",
    )
    parser.add_argument("--voxpopuli-root", type=Path, required=True, help="Root used for voxpopuli (contains transcribed_data/<lang>/).")
    parser.add_argument("--lang", required=True, help="Language code (e.g. fr, de).")
    parser.add_argument("--output-dir", type=Path, default=Path("test-data"), help="Output root (default: test-data).")
    return parser.parse_args()


def sanitize_id(raw: str) -> str:
    return re.sub(r"[^\w\-.]", "_", raw)


def main() -> int:
    args = parse_args()
    import soundfile as sf
    import numpy as np
    from tqdm import tqdm

    root = args.voxpopuli_root.resolve()
    lang = args.lang.strip().lower()
    out_root = args.output_dir.resolve()
    transcribed = root / "transcribed_data" / lang
    if not transcribed.is_dir():
        print(f"Error: {transcribed} not found. Run voxpopuli get_asr_data first.", file=sys.stderr)
        return 1

    target_sr = 16000
    splits = ["train", "dev", "test"]
    total = 0

    for split in splits:
        tsv_path = transcribed / f"asr_{split}.tsv"
        if not tsv_path.is_file():
            print(f"Skip {split}: {tsv_path} not found.", file=sys.stderr)
            continue
        out_split = out_root / "VoxPopuli" / lang / split
        out_split.mkdir(parents=True, exist_ok=True)
        trans_path = out_split / f"{split}.trans.txt"
        lines = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            header = f.readline()
            col_idx = {name: i for i, name in enumerate(header.strip().split("\t"))}
            id_idx = col_idx.get("id", 0)
            text_idx = col_idx.get("normalized_text", col_idx.get("raw_text", 1))
            # Count lines for progress bar (data lines only; header already read)
            line_count = sum(1 for _ in f)
            f.seek(0)
            f.readline()  # skip header again
            for line in tqdm(f, total=line_count, desc=f"Convert {split}", unit="utt"):
                parts = line.strip().split("\t")
                if len(parts) <= max(id_idx, text_idx):
                    continue
                utt_id = sanitize_id(parts[id_idx])
                transcript = parts[text_idx].strip()
                if not transcript:
                    continue
                ogg_name = f"{parts[id_idx]}.ogg"
                ogg_path = None
                for year_dir in transcribed.iterdir():
                    if year_dir.is_dir() and year_dir.name.isdigit():
                        cand = year_dir / ogg_name
                        if cand.is_file():
                            ogg_path = cand
                            break
                if ogg_path is None:
                    print(f"Warning: audio not found for {utt_id}", file=sys.stderr)
                    continue
                arr, sr = sf.read(str(ogg_path), dtype="float32", always_2d=False)
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                if sr != target_sr:
                    try:
                        from scipy import signal as scipy_signal
                        num = int(round(len(arr) * target_sr / sr))
                        arr = scipy_signal.resample(arr, num).astype(np.float32)
                    except ImportError:
                        pass
                flac_path = out_split / f"{utt_id}.flac"
                sf.write(flac_path, arr, target_sr, subtype="PCM_16")
                lines.append(f"{utt_id} {transcript}\n")
        if lines:
            trans_path.write_text("".join(lines), encoding="utf-8")
            n = len(lines)
            total += n
            print(f"{split}: wrote {n} utterances to {out_split}")
    print(f"Total: {total} utterances under {out_root}/VoxPopuli/{lang}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
