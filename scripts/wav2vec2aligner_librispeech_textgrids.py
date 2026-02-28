#!/usr/bin/env python3
"""Generate LibriSpeech TextGrid files with wav2vec2aligner in batch mode."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator, TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-generate TextGrid files for LibriSpeech by reusing one "
            "wav2vec2aligner model instance."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("test-data/LibriSpeech"),
        help="Root directory that contains subset folders like test-clean/test-other.",
    )
    parser.add_argument(
        "--subsets",
        default="test-clean,test-other",
        help="Comma-separated subset names under --dataset-root.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
        help="Execution device (default: cuda, fails fast if unavailable).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate expected by wav2vec2aligner.",
    )
    parser.add_argument(
        "--word-padding",
        type=int,
        default=0,
        help="Frame padding to apply around words.",
    )
    parser.add_argument(
        "--sentence-padding",
        type=int,
        default=0,
        help="Frame padding to apply around sentences.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of utterances to process across all subsets (0 = all).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N utterances.",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing .TextGrid files (default).",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip utterances that already have a .TextGrid file.",
    )
    parser.add_argument(
        "--perf-out",
        type=Path,
        default=None,
        help="Optional perf report output path (JSON or JSONL when --perf-append is set).",
    )
    parser.add_argument(
        "--perf-warmup",
        type=int,
        default=10,
        help="Warm-up iterations for the first measured utterance only.",
    )
    parser.add_argument(
        "--perf-repeats",
        type=int,
        default=30,
        help="Measured repeats per utterance for perf timing.",
    )
    parser.add_argument(
        "--perf-aggregate",
        choices=["median", "mean"],
        default="median",
        help="How to aggregate per-repeat timings into one value per utterance.",
    )
    parser.add_argument(
        "--perf-append",
        action="store_true",
        help="Append one JSON record per utterance to --perf-out as JSONL.",
    )
    return parser.parse_args()


def ensure_aligner_importable() -> None:
    try:
        import aligner  # noqa: F401

        return
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        local_pkg_root = repo_root / "wav2vec2aligner-main"
        if local_pkg_root.exists():
            sys.path.insert(0, str(local_pkg_root))
    try:
        import aligner  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Cannot import 'aligner'. Install it with "
            "'pip install -e wav2vec2aligner-main' or run this script from the repo root."
        ) from exc


def choose_device(requested: str):
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iter_librispeech_rows(
    subset_dir: Path,
) -> Iterator[tuple[str, Path, str]]:
    for transcript_file in sorted(subset_dir.rglob("*.trans.txt")):
        with transcript_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                utt_id, transcript = parts
                yield utt_id, transcript_file.parent / f"{utt_id}.flac", transcript


def load_and_prepare_audio(audio_path: Path, sample_rate: int):
    import torch
    import soundfile as sf
    import torchaudio

    audio_np, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    wav = torch.from_numpy(audio_np)
    if wav.dim() == 2:
        # soundfile returns [samples, channels] for multi-channel audio
        wav = wav.mean(dim=1)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if wav.size(0) != 1:
        wav = torch.mean(wav, dim=0).unsqueeze(0)
    return wav


def format_elapsed(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def aggregate_measurements(values: list[float], mode: str) -> float:
    if not values:
        return 0.0
    if mode == "mean":
        return sum(values) / len(values)
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    return sorted_values[mid]


def summarize_metric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "median": aggregate_measurements(values, "median"),
        "min": min(values),
        "max": max(values),
    }


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def summary_path_for(path: Path) -> Path:
    return Path(f"{path}.summary.json")


def maybe_sync_cuda(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_perf_json_report(
    path: Path,
    config: dict[str, object],
    records: list[dict[str, object]],
    aggregate: dict[str, object],
) -> None:
    ensure_parent_dir(path)
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config,
        "records": records,
        "aggregate": aggregate,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_perf_summary_report(
    path: Path,
    config: dict[str, object],
    aggregate: dict[str, object],
) -> None:
    ensure_parent_dir(path)
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config,
        "aggregate": aggregate,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    if args.perf_repeats < 1:
        raise SystemExit("--perf-repeats must be >= 1.")
    if args.perf_warmup < 0:
        raise SystemExit("--perf-warmup must be >= 0.")
    if args.perf_out is None and (
        args.perf_append
        or args.perf_warmup != 10
        or args.perf_repeats != 30
        or args.perf_aggregate != "median"
    ):
        print("[WARN] perf flags are ignored unless --perf-out is set.")
    ensure_aligner_importable()

    import aligner.heavy as heavy
    from aligner.heavy import (
    align_speech_file,
    align_speech_file_profiled,
    collect_per_stage_memory,
    load_model,
)
    from aligner.utils import TextHash, create_text_grid_from_segments, create_transducer

    dataset_root = args.dataset_root
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    subsets = [x.strip() for x in args.subsets.split(",") if x.strip()]
    if not subsets:
        raise SystemExit("No subsets specified.")

    started_at = time.perf_counter()
    device = choose_device(args.device)
    heavy.DEVICE = device
    print(f"Using device: {device}")
    print("Loading model once for entire batch...")
    model, labels = load_model()
    perf_enabled = args.perf_out is not None
    perf_out_path = args.perf_out
    perf_config = {
        "warmup": args.perf_warmup,
        "repeats": args.perf_repeats,
        "aggregate": args.perf_aggregate,
        "append": args.perf_append,
    }
    perf_records: list[dict[str, object]] = []
    perf_forward_samples: list[float] = []
    perf_post_samples: list[float] = []
    perf_align_samples: list[float] = []
    perf_total_samples: list[float] = []
    perf_jsonl_handle: TextIO | None = None
    perf_jsonl_since_flush = 0
    perf_warmup_done = False
    if perf_enabled and args.perf_append:
        assert perf_out_path is not None
        ensure_parent_dir(perf_out_path)
        perf_jsonl_handle = perf_out_path.open(
            "a",
            encoding="utf-8",
            buffering=1024 * 1024,
        )

    seen = 0
    success = 0
    skipped = 0
    missing_audio = 0
    failed = 0
    lib_work_elapsed_s = 0.0
    lib_work_calls = 0

    for subset in subsets:
        subset_dir = dataset_root / subset
        if not subset_dir.exists():
            print(f"[WARN] subset directory not found, skipping: {subset_dir}")
            continue

        print(f"Processing subset: {subset}")
        for utt_id, audio_path, transcript in iter_librispeech_rows(subset_dir):
            if args.limit > 0 and seen >= args.limit:
                break
            seen += 1

            tg_path = audio_path.with_suffix(".TextGrid")
            if tg_path.exists() and not args.overwrite:
                skipped += 1
                continue

            if not audio_path.exists():
                missing_audio += 1
                print(f"[WARN] missing audio for {utt_id}: {audio_path}")
                continue

            try:
                sentence_list = [transcript.strip()]
                if not sentence_list[0]:
                    skipped += 1
                    continue

                wav = load_and_prepare_audio(audio_path, args.sample_rate)
                lib_work_calls += 1
                transcript_concat = "".join(sentence_list)

                def build_text_hash():
                    transducer = create_transducer(transcript_concat, labels)
                    return TextHash(sentence_list, transducer)

                if perf_enabled:
                    if not perf_warmup_done:
                        for _ in range(args.perf_warmup):
                            align_speech_file_profiled(
                                wav,
                                build_text_hash(),
                                model,
                                labels,
                                args.word_padding,
                                args.sentence_padding,
                            )
                            maybe_sync_cuda(device)
                        if device.type == "cuda":
                            import torch

                            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                                torch.cuda.reset_peak_memory_stats(device)
                        perf_warmup_done = True

                    forward_ms_repeats: list[float] = []
                    post_ms_repeats: list[float] = []
                    align_ms_repeats: list[float] = []
                    total_ms_repeats: list[float] = []
                    dtype = "f32"
                    perf_device = str(device)
                    vocab_size = 0
                    characters = None
                    words = None
                    sentences = None
                    num_frames = 0
                    for repeat_idx in range(args.perf_repeats):
                        (
                            rep_characters,
                            rep_words,
                            rep_sentences,
                            rep_num_frames,
                            perf_sample,
                        ) = align_speech_file_profiled(
                            wav,
                            build_text_hash(),
                            model,
                            labels,
                            args.word_padding,
                            args.sentence_padding,
                        )
                        forward_ms_repeats.append(float(perf_sample["forward_ms"]))
                        post_ms_repeats.append(float(perf_sample["post_ms"]))
                        align_ms_repeats.append(float(perf_sample["align_ms"]))
                        total_ms_repeats.append(float(perf_sample["total_ms"]))

                        if repeat_idx == 0:
                            characters = rep_characters
                            words = rep_words
                            sentences = rep_sentences
                            num_frames = rep_num_frames
                            dtype = str(perf_sample["dtype"])
                            perf_device = str(perf_sample["device"])
                            vocab_size = int(perf_sample["vocab_size"])

                    if characters is None or words is None or sentences is None:
                        raise RuntimeError("Missing profiled alignment output.")

                    forward_ms = aggregate_measurements(forward_ms_repeats, args.perf_aggregate)
                    post_ms = aggregate_measurements(post_ms_repeats, args.perf_aggregate)
                    align_ms = aggregate_measurements(align_ms_repeats, args.perf_aggregate)
                    total_ms = aggregate_measurements(total_ms_repeats, args.perf_aggregate)
                    lib_work_elapsed_s += total_ms / 1000.0

                    memory_result = collect_per_stage_memory(
                        wav,
                        build_text_hash(),
                        model,
                        labels,
                        args.word_padding,
                        args.sentence_padding,
                    )

                    duration_ms = (
                        int((wav.size(1) * 1000) / args.sample_rate)
                        if args.sample_rate > 0
                        else 0
                    )
                    frame_stride_ms = (
                        ((wav.size(1) / num_frames) * 1000.0 / args.sample_rate)
                        if num_frames > 0 and args.sample_rate > 0
                        else 0.0
                    )
                    perf_record = {
                        "utterance_id": utt_id,
                        "audio_path": str(audio_path),
                        "duration_ms": duration_ms,
                        "num_frames_t": int(num_frames),
                        "vocab_size": vocab_size,
                        "dtype": dtype,
                        "device": perf_device,
                        "frame_stride_ms": frame_stride_ms,
                        "warmup": args.perf_warmup,
                        "repeats": args.perf_repeats,
                        "aggregate": args.perf_aggregate,
                        "forward_ms": forward_ms,
                        "post_ms": post_ms,
                        "align_ms": align_ms,
                        "total_ms": total_ms,
                        "forward_ms_repeats": forward_ms_repeats,
                        "post_ms_repeats": post_ms_repeats,
                        "align_ms_repeats": align_ms_repeats,
                        "total_ms_repeats": total_ms_repeats,
                        "memory": memory_result,
                    }
                    perf_forward_samples.append(forward_ms)
                    perf_post_samples.append(post_ms)
                    perf_align_samples.append(align_ms)
                    perf_total_samples.append(total_ms)
                    if args.perf_append:
                        if perf_jsonl_handle is None:
                            raise RuntimeError("Missing perf JSONL file handle.")
                        json.dump(perf_record, perf_jsonl_handle)
                        perf_jsonl_handle.write("\n")
                        perf_jsonl_since_flush += 1
                        if perf_jsonl_since_flush >= 32:
                            perf_jsonl_handle.flush()
                            perf_jsonl_since_flush = 0
                    else:
                        perf_records.append(perf_record)
                else:
                    lib_started_at = time.perf_counter()
                    try:
                        characters, words, sentences, num_frames = align_speech_file(
                            wav,
                            build_text_hash(),
                            model,
                            labels,
                            args.word_padding,
                            args.sentence_padding,
                        )
                    finally:
                        lib_work_elapsed_s += time.perf_counter() - lib_started_at
                waveform_to_frame_ratio = wav.size(1) / num_frames
                tg = create_text_grid_from_segments(
                    characters,
                    "characters",
                    waveform_to_frame_ratio,
                    sample_rate=args.sample_rate,
                )
                words_tg = create_text_grid_from_segments(
                    words,
                    "words",
                    waveform_to_frame_ratio,
                    sample_rate=args.sample_rate,
                )
                sentences_tg = create_text_grid_from_segments(
                    sentences,
                    "sentences",
                    waveform_to_frame_ratio,
                    sample_rate=args.sample_rate,
                )
                tg.tiers += words_tg.get_tiers()
                tg.tiers += sentences_tg.get_tiers()
                tg.to_file(tg_path)
                success += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[FAIL] {utt_id}: {exc}")

            if args.progress_every > 0 and seen % args.progress_every == 0:
                print(
                    f"  progress={seen} success={success} failed={failed} "
                    f"skipped={skipped} missing_audio={missing_audio}"
                )

        if args.limit > 0 and seen >= args.limit:
            break

    print("\nDone.")
    print(f"  seen:         {seen}")
    print(f"  success:      {success}")
    print(f"  failed:       {failed}")
    print(f"  skipped:      {skipped}")
    print(f"  missing_audio:{missing_audio}")
    elapsed_s = time.perf_counter() - started_at
    print(f"  elapsed:      {elapsed_s:.2f}s ({format_elapsed(elapsed_s)})")
    print(
        f"  lib_work_elapsed: {lib_work_elapsed_s:.2f}s "
        f"({format_elapsed(lib_work_elapsed_s)})"
    )
    if lib_work_calls > 0:
        print(
            "  lib_avg_per_call: "
            f"{(lib_work_elapsed_s * 1000.0) / lib_work_calls:.2f}ms"
        )
    if perf_enabled and perf_out_path is not None:
        aggregate = {
            "utterance_count": len(perf_total_samples),
            "forward_ms": summarize_metric(perf_forward_samples),
            "post_ms": summarize_metric(perf_post_samples),
            "align_ms": summarize_metric(perf_align_samples),
            "total_ms": summarize_metric(perf_total_samples),
        }
        if args.perf_append:
            if perf_jsonl_handle is not None:
                perf_jsonl_handle.flush()
                perf_jsonl_handle.close()
            summary_path = summary_path_for(perf_out_path)
            write_perf_summary_report(summary_path, perf_config, aggregate)
            print(str(perf_out_path))
            print(str(summary_path))
        else:
            write_perf_json_report(perf_out_path, perf_config, perf_records, aggregate)
            print(str(perf_out_path))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
