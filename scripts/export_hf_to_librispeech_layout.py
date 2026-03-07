#!/usr/bin/env python3
"""
Export a Hugging Face speech dataset to LibriSpeech-like layout under test-data.

Writes subset directories containing:
  - *.trans.txt  (one line per utterance: "utt_id transcript")
  - {utt_id}.flac (16 kHz mono FLAC)

Supported datasets:
  - facebook/multilingual_librispeech  (--config required, e.g. german, french)
  - gigant/african_accented_french      (no config; splits: train, test)

Dependencies: pip install datasets soundfile (optional: scipy for resampling to 16 kHz)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Hugging Face speech dataset to LibriSpeech-like layout (.trans.txt + .flac).",
        epilog="Example: python export_hf_to_librispeech_layout.py --dataset facebook/multilingual_librispeech --config german --splits test --limit 10",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Hugging Face dataset id (e.g. facebook/multilingual_librispeech, gigant/african_accented_french).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset config name (required for multilingual_librispeech; omit for african_accented_french).",
    )
    parser.add_argument(
        "--splits",
        default="test",
        help="Comma-separated split names (default: test only). Use e.g. train,test,dev to get more splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root output directory (default: test-data, i.e. parent of this script).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max utterances per split (0 = all). Useful for testing.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom dataset loading code (required for some HF datasets, e.g. gigant/african_accented_french).",
    )
    return parser.parse_args()


def get_dataset_slug(dataset_id: str) -> str:
    """Map dataset id to a short folder name."""
    name = dataset_id.split("/")[-1]
    slug = name.replace("_", " ").title().replace(" ", "")
    return slug


def get_transcript_key(columns: list[str]) -> str:
    # Try common names; order matters for datasets that have multiple (prefer transcript text).
    for key in ("transcript", "text", "sentence", "normalized_text", "raw_text", "transcription"):
        if key in columns:
            return key
    raise KeyError(f"No transcript column found; columns: {columns}")


def get_id_key(columns: list[str]) -> str | None:
    for key in ("id", "utt_id", "utterance_id", "audio_id"):
        if key in columns:
            return key
    return None


def sanitize_utt_id(raw: str) -> str:
    """Replace characters that are unsafe in filenames."""
    return re.sub(r"[^\w\-.]", "_", raw)


def export_split(
    dataset_id: str,
    config: str | None,
    split: str,
    output_dir: Path,
    limit: int,
    trust_remote_code: bool = False,
) -> int:
    from datasets import load_dataset
    import numpy as np
    import soundfile as sf
    from tqdm import tqdm

    slug = get_dataset_slug(dataset_id)
    if config:
        out_split_dir = output_dir / slug / config / split
    else:
        out_split_dir = output_dir / slug / split

    out_split_dir.mkdir(parents=True, exist_ok=True)
    trans_path = out_split_dir / f"{split}.trans.txt"

    load_kwargs: dict = {"path": dataset_id, "split": split}
    if config:
        load_kwargs["name"] = config
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    ds = load_dataset(**load_kwargs)
    columns = ds.column_names
    transcript_key = get_transcript_key(columns)
    id_key = get_id_key(columns)
    # Fallback keys if the primary transcript column is empty for a row (exclude id-like columns)
    id_like = {id_key} if id_key else set()
    transcript_fallback_keys = [k for k in ("transcript", "text", "sentence", "normalized_text", "raw_text", "transcription") if k in columns and k != transcript_key and k not in id_like]

    target_sr = 16000
    lines: list[str] = []
    n = 0
    iterator = tqdm(ds, desc=f"{split}", total=len(ds), unit="utt", leave=True)

    for idx, row in enumerate(iterator):
        if limit > 0 and n >= limit:
            break

        if id_key and row.get(id_key) is not None:
            utt_id = str(row[id_key])
        else:
            utt_id = f"{split}_{idx:06d}"
        utt_id = sanitize_utt_id(utt_id)

        transcript = row.get(transcript_key)
        if transcript is None or (isinstance(transcript, str) and not str(transcript).strip()):
            for k in transcript_fallback_keys:
                v = row.get(k)
                if v is not None and str(v).strip():
                    transcript = v
                    break
        if transcript is None or not str(transcript).strip():
            continue
        if isinstance(transcript, str):
            transcript = transcript.strip()
        else:
            transcript = str(transcript).strip()

        audio = row.get("audio")
        if audio is None:
            continue
        if isinstance(audio, dict):
            array = audio.get("array")
            sr = audio.get("sampling_rate")
        else:
            array = audio
            sr = target_sr

        if array is None:
            continue
        arr = np.asarray(array, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != target_sr and sr is not None:
            try:
                from scipy import signal as scipy_signal
                num_samples = int(round(len(arr) * target_sr / sr))
                arr = scipy_signal.resample(arr, num_samples).astype(np.float32)
            except ImportError:
                if sr != target_sr:
                    print(f"Warning: scipy not installed; skipping resample {sr} -> {target_sr} Hz for {utt_id}", file=sys.stderr)
        elif sr is None:
            sr = target_sr

        flac_path = out_split_dir / f"{utt_id}.flac"
        sf.write(flac_path, arr, target_sr, subtype="PCM_16")

        line = f"{utt_id} {transcript}\n"
        lines.append(line)
        n += 1

    if lines:
        trans_path.write_text("".join(lines), encoding="utf-8")
    return n


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (Path(__file__).resolve().parent)
    output_dir = output_dir.resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        print("No splits specified.", file=sys.stderr)
        return 1

    dataset_id = args.dataset.strip()
    if "multilingual_librispeech" in dataset_id and not args.config:
        print("--config is required for facebook/multilingual_librispeech (e.g. german, french).", file=sys.stderr)
        return 1

    try:
        total = 0
        for split in splits:
            n = export_split(
                dataset_id,
                args.config,
                split,
                output_dir,
                args.limit,
                trust_remote_code=args.trust_remote_code,
            )
            total += n
            print(f"{split}: wrote {n} utterances")
        print(f"Total: {total} utterances under {output_dir}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
