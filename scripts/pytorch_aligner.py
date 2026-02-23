#!/usr/bin/env python3
"""Download LibriSpeech test sets and generate reference word alignments.

This script does everything in one place:
1) Download test-clean and test-other archives from OpenSLR.
2) Extract into test-data/LibriSpeech.
3) Run wav2vec2-base-960h emissions with PyTorch Wav2Vec2.
4) Run the same CTC Viterbi/word-grouping strategy used by src/lib.rs.
5) Save JSON alignment references under test-data/alignments.
"""

from __future__ import annotations

import argparse
import json
import math
import tarfile
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC


DATASET_URLS = {
    "test-clean": "https://openslr.trmal.net/resources/12/test-clean.tar.gz",
    "test-other": "https://openslr.trmal.net/resources/12/test-other.tar.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LibriSpeech test sets and build reference alignments."
    )
    parser.add_argument("--data-dir", default="test-data")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--subsets", default="test-clean,test-other")
    parser.add_argument("--model-dir", default="models/wav2vec2-base-960h")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of utterances to align per subset (0 = no limit).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_device(arg_device: str) -> torch.device:
    if arg_device == "cpu":
        return torch.device("cpu")
    if arg_device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = min(block_num * block_size, total_size)
    pct = int(downloaded * 100 / total_size)
    print(f"\r  {pct:3d}% ({downloaded}/{total_size} bytes)", end="", flush=True)


def download_archive(url: str, out_path: Path, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        print(f"Archive exists, skip download: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    urlretrieve(url, out_path, reporthook=print_progress)
    print("\nDone.")


def safe_extract(tar: tarfile.TarFile, output_dir: Path) -> None:
    output_dir_abs = output_dir.resolve()
    for member in tar.getmembers():
        member_path = (output_dir / member.name).resolve()
        if output_dir_abs not in member_path.parents and member_path != output_dir_abs:
            raise RuntimeError(f"Blocked unsafe path in tar: {member.name}")
    try:
        tar.extractall(output_dir, **{"filter": "data"})
    except TypeError:
        tar.extractall(output_dir)


def extract_archive(archive_path: Path, output_dir: Path, subset: str, overwrite: bool) -> None:
    subset_dir = output_dir / "LibriSpeech" / subset
    if subset_dir.exists() and not overwrite:
        print(f"Subset already extracted, skip extract: {subset_dir}")
        return
    print(f"Extracting: {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        safe_extract(tar, output_dir)
    print(f"Extracted: {subset_dir}")


def build_token_sequence(
    transcript: str,
    vocab: dict[str, int],
    blank_id: int,
    word_sep_id: int,
) -> tuple[list[int], list[str | None]]:
    # Match transcript casing to model vocabulary to avoid dropping letters.
    alpha_tokens = [t for t in vocab if len(t) == 1 and t.isalpha()]
    has_upper = any(t.isupper() for t in alpha_tokens)
    has_lower = any(t.islower() for t in alpha_tokens)
    if has_upper and not has_lower:
        cleaned = transcript.upper()
    else:
        cleaned = transcript.lower()

    tokens: list[int] = [blank_id]
    chars: list[str | None] = [None]

    for wi, word in enumerate(cleaned.split()):
        if wi > 0:
            tokens.append(word_sep_id)
            chars.append("|")
            tokens.append(blank_id)
            chars.append(None)
        for c in word:
            if c in vocab:
                tokens.append(vocab[c])
                chars.append(c)
                tokens.append(blank_id)
                chars.append(None)
    return tokens, chars


def normalize_audio(samples: torch.Tensor) -> torch.Tensor:
    mean = samples.mean()
    var = (samples - mean).pow(2).mean()
    std = torch.sqrt(torch.clamp(var, min=1e-7))
    return (samples - mean) / std


def _best_prev_state(
    dp_row: list[float], tokens: list[int], state_index: int
) -> tuple[float, int]:
    best = dp_row[state_index]
    from_s = state_index
    if state_index >= 1 and dp_row[state_index - 1] > best:
        best = dp_row[state_index - 1]
        from_s = state_index - 1
    if (
        state_index >= 2
        and tokens[state_index] != tokens[state_index - 2]
        and dp_row[state_index - 2] > best
    ):
        best = dp_row[state_index - 2]
        from_s = state_index - 2
    return best, from_s


def forced_align_viterbi(log_probs: list[list[float]], tokens: list[int]) -> list[tuple[int, int]]:
    t_len = len(log_probs)
    s_len = len(tokens)
    if t_len == 0 or s_len == 0:
        return []

    dp = [[float("-inf")] * s_len for _ in range(t_len)]
    bp = [[0] * s_len for _ in range(t_len)]

    dp[0][0] = log_probs[0][tokens[0]]
    if s_len > 1:
        dp[0][1] = log_probs[0][tokens[1]]

    for t in range(1, t_len):
        for s in range(s_len):
            emit = log_probs[t][tokens[s]]
            best, from_s = _best_prev_state(dp[t - 1], tokens, s)
            dp[t][s] = best + emit
            bp[t][s] = from_s

    s = s_len - 1
    if s_len >= 2 and dp[t_len - 1][s_len - 2] > dp[t_len - 1][s_len - 1]:
        s = s_len - 2

    path: list[tuple[int, int]] = [(s, t_len - 1)]
    for t in range(t_len - 1, 0, -1):
        s = bp[t][s]
        path.append((s, t - 1))
    path.reverse()
    return path


def group_into_words(
    path: list[tuple[int, int]],
    tokens: list[int],
    chars: list[str | None],
    log_probs: list[list[float]],
    blank_id: int,
    word_sep_id: int,
    stride_ms: float,
) -> list[dict]:
    words: list[dict] = []
    cur_word = ""
    start_frame: int | None = None
    end_frame = 0
    lp_accum: list[float] = []
    prev_state: int | None = None

    def flush() -> None:
        nonlocal cur_word, start_frame, end_frame, lp_accum
        if not cur_word:
            return
        conf = 0.0 if not lp_accum else math.exp(sum(lp_accum) / len(lp_accum))
        words.append(
            {
                "word": cur_word,
                "start_ms": int((start_frame or 0) * stride_ms),
                "end_ms": int((end_frame + 1) * stride_ms),
                "confidence": conf,
            }
        )
        cur_word = ""
        start_frame = None
        lp_accum = []

    for s, frame in path:
        tid = tokens[s]
        if tid == blank_id:
            prev_state = s
            continue
        if tid == word_sep_id:
            flush()
            prev_state = s
            continue
        c = chars[s]
        if c is not None:
            if start_frame is None:
                start_frame = frame
            end_frame = frame
            # CTC emits the same token over multiple consecutive frames.
            # Only append when entering a new token state.
            if prev_state != s:
                cur_word += c
                lp_accum.append(log_probs[frame][tid])
        prev_state = s

    flush()
    return words


def parse_transcriptions(subset_dir: Path) -> Iterable[tuple[str, Path, str]]:
    for trans_file in sorted(subset_dir.rglob("*.trans.txt")):
        base_dir = trans_file.parent
        with trans_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                utt_id, transcript = parts
                audio_path = base_dir / f"{utt_id}.flac"
                yield utt_id, audio_path, transcript


def align_utterance(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    transcript: str,
    vocab: dict[str, int],
    blank_id: int,
    sep_id: int,
    stride_ms: float,
    device: torch.device,
) -> list[dict]:
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    waveform = normalize_audio(waveform)

    with torch.inference_mode():
        logits = model(waveform.to(device)).logits
        log_probs = torch.log_softmax(logits, dim=-1)[0].cpu().tolist()

    tokens, chars = build_token_sequence(transcript, vocab, blank_id, sep_id)
    if not tokens:
        return []

    t_len = len(log_probs)
    min_frames = (len(tokens) + 1) // 2
    if t_len < min_frames:
        return []

    path = forced_align_viterbi(log_probs, tokens)
    return group_into_words(path, tokens, chars, log_probs, blank_id, sep_id, stride_ms)


def align_subset(
    model: torch.nn.Module,
    data_dir: Path,
    subset: str,
    vocab: dict[str, int],
    blank_id: int,
    sep_id: int,
    stride_ms: float,
    device: torch.device,
    limit: int,
) -> list[dict]:
    subset_dir = data_dir / "LibriSpeech" / subset
    if not subset_dir.exists():
        raise SystemExit(f"Subset directory not found: {subset_dir}")

    rows: list[dict] = []
    count = 0
    for utt_id, audio_path, transcript in parse_transcriptions(subset_dir):
        if not audio_path.exists():
            continue
        audio_np, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        waveform = torch.from_numpy(audio_np)
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=1)
        waveform = waveform.unsqueeze(0)
        if sample_rate != 16000:
            continue
        words = align_utterance(
            model=model,
            waveform=waveform,
            transcript=transcript,
            vocab=vocab,
            blank_id=blank_id,
            sep_id=sep_id,
            stride_ms=stride_ms,
            device=device,
        )
        rows.append(
            {
                "id": utt_id,
                "audio_path": audio_path.relative_to(data_dir).as_posix(),
                "transcript": transcript,
                "words": words,
            }
        )
        count += 1
        if count % 100 == 0:
            print(f"  aligned {count} utterances in {subset}...")
        if limit > 0 and count >= limit:
            print(f"  reached limit ({limit}) for {subset}, stopping early.")
            break

    return rows


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    alignments_dir = data_dir / "alignments"
    alignments_dir.mkdir(parents=True, exist_ok=True)

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    unknown = [s for s in subsets if s not in DATASET_URLS]
    if unknown:
        raise SystemExit(f"Unknown subset(s): {unknown}. Valid: {list(DATASET_URLS)}")

    for subset in subsets:
        archive_name = f"{subset}.tar.gz"
        archive_path = data_dir / archive_name
        download_archive(DATASET_URLS[subset], archive_path, overwrite=args.overwrite)
        extract_archive(archive_path, data_dir, subset, overwrite=args.overwrite)

    device = choose_device(args.device)
    print(f"Using device: {device}")
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}")
    vocab_path = model_dir / "vocab.json"
    if not vocab_path.exists():
        raise SystemExit(f"vocab.json not found in model directory: {vocab_path}")

    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    with vocab_path.open("r", encoding="utf-8") as vf:
        raw_vocab = json.load(vf)
    vocab = {k: int(v) for k, v in raw_vocab.items() if isinstance(k, str) and len(k) == 1}
    blank_id = int(raw_vocab.get("<pad>", 0))
    sep_id = vocab.get("|", 4)
    stride_ms = 20.0

    for subset in subsets:
        output_path = alignments_dir / f"{subset}.json"
        if output_path.exists() and not args.overwrite:
            print(f"Alignment exists, skip: {output_path}")
            continue
        print(f"Aligning subset: {subset}")
        rows = align_subset(
            model=model,
            data_dir=data_dir,
            subset=subset,
            vocab=vocab,
            blank_id=blank_id,
            sep_id=sep_id,
            stride_ms=stride_ms,
            device=device,
            limit=args.limit,
        )
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Saved: {output_path} ({len(rows)} utterances)")


if __name__ == "__main__":
    main()
