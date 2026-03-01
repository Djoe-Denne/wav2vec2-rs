#!/usr/bin/env python3
"""
Compare baseline and Rust-generated TextGrid files under a LibriSpeech root.
Takes the path to the LibriSpeech root; recursively finds every directory
containing baseline .TextGrid and *_rust_*.TextGrid pairs. Aggregates all
word-pair diffs and outputs a single global median for each metric per mode:
  - median_start_diff_ms: median of (xmin_rust - xmin_ref) over all words, in ms
  - median_end_diff_ms: median of (xmax_rust - xmax_ref) over all words, in ms
  - median_word_middle_diff_ms: median of (middle_rust - middle_ref) over all words, in ms

Exit codes (unique, for CI):
  0  Success; all metrics within thresholds.
  1  No baseline + *_rust_*.TextGrid pairs found, or path is not a directory.
  2  At least one |metric| > 5 ms for some mode (alignment drift threshold).
  3  Same metric differs by more than 0.01 ms between modes (cross-mode consistency).
  4  One or more baseline/rust pair failed to load or compare (parse/word-count error).
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

EXIT_SUCCESS = 0
EXIT_NO_PAIRS = 1
EXIT_ABS_THRESHOLD = 2   # |metric| > 5 ms
EXIT_MODE_DRIFT = 3     # same metric differs > 0.01 ms between modes
EXIT_LOAD_ERROR = 4     # one or more pair failed to load/compare
ABS_THRESHOLD_MS = 5.0
MODE_DRIFT_THRESHOLD_MS = 0.01


def parse_textgrid_words(path: Path) -> list[tuple[float, float, str]]:
    """
    Parse a TextGrid file and return all intervals from the tier named "words"
    as a list of (start_sec, end_sec, text).
    Raises ValueError if the file has no "words" tier or parse fails.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Find the tier with name = "words"
    in_words_tier = False
    intervals_size = 0
    intervals: list[tuple[float, float, str]] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        # Start of a new item (tier)
        if re.match(r"\s*item\s*\[\d+\]\s*:", line):
            in_words_tier = False
        if re.search(r'name\s*=\s*"words"', line):
            in_words_tier = True
        if in_words_tier:
            m = re.search(r"intervals:\s*size\s*=\s*(\d+)", line)
            if m:
                intervals_size = int(m.group(1))
                i += 1
                # Parse subsequent interval blocks until we have intervals_size
                while len(intervals) < intervals_size and i < len(lines):
                    if re.search(r"\s*intervals\s*\[\d+\]\s*:", lines[i]):
                        i += 1
                        xmin = xmax = float("nan")
                        text_val = ""
                        while i < len(lines) and re.match(
                            r"\s*(xmin|xmax|text)\s*=", lines[i]
                        ):
                            kv = lines[i]
                            xm = re.search(r"xmin\s*=\s*([\d.]+)", kv)
                            if xm:
                                xmin = float(xm.group(1))
                            xm = re.search(r"xmax\s*=\s*([\d.]+)", kv)
                            if xm:
                                xmax = float(xm.group(1))
                            tm = re.search(r'text\s*=\s*"([^"]*)"', kv)
                            if tm:
                                text_val = tm.group(1)
                            i += 1
                        if not (xmin != xmin and xmax != xmax):  # nan check
                            intervals.append((xmin, xmax, text_val))
                        continue
                    i += 1
                break
        i += 1

    if not in_words_tier:
        raise ValueError(f'No tier named "words" in {path}')
    if intervals_size and len(intervals) != intervals_size:
        raise ValueError(
            f"Expected {intervals_size} intervals in words tier of {path}, got {len(intervals)}"
        )
    return intervals


def get_word_intervals(
    intervals: list[tuple[float, float, str]],
) -> list[tuple[float, float]]:
    """Return only intervals with non-empty text, as (start_sec, end_sec)."""
    return [(s, e) for s, e, t in intervals if (t or "").strip()]


def compute_metrics(
    ref_words: list[tuple[float, float]],
    rust_words: list[tuple[float, float]],
) -> tuple[float, float, float]:
    """
    Compute median_start_diff_ms, median_end_diff_ms, median_word_middle_diff_ms.
    ref_words and rust_words must be same length (pairs by index).
    All values in milliseconds (seconds from TextGrid * 1000).
    """
    if len(ref_words) != len(rust_words):
        raise ValueError(
            f"Word count mismatch: ref={len(ref_words)}, rust={len(rust_words)}"
        )
    if not ref_words:
        raise ValueError("No word intervals to compute metrics")

    # Per-word-pair differences: rust - ref, converted to ms
    start_diffs_ms = [
        (rust_words[i][0] - ref_words[i][0]) * 1000.0 for i in range(len(ref_words))
    ]
    end_diffs_ms = [
        (rust_words[i][1] - ref_words[i][1]) * 1000.0 for i in range(len(ref_words))
    ]
    middle_diffs_ms = [
        ((rust_words[i][0] + rust_words[i][1]) / 2.0 - (ref_words[i][0] + ref_words[i][1]) / 2.0)
        * 1000.0
        for i in range(len(ref_words))
    ]

    median_start_diff_ms = statistics.median(start_diffs_ms)
    median_end_diff_ms = statistics.median(end_diffs_ms)
    median_word_middle_diff_ms = statistics.median(middle_diffs_ms)

    return (
        median_start_diff_ms,
        median_end_diff_ms,
        median_word_middle_diff_ms,
    )


def collect_diffs_ms(
    ref_words: list[tuple[float, float]],
    rust_words: list[tuple[float, float]],
) -> tuple[list[float], list[float], list[float]]:
    """Return (start_diffs_ms, end_diffs_ms, middle_diffs_ms) for aggregation."""
    if len(ref_words) != len(rust_words) or not ref_words:
        return ([], [], [])
    start_diffs_ms = [
        (rust_words[i][0] - ref_words[i][0]) * 1000.0 for i in range(len(ref_words))
    ]
    end_diffs_ms = [
        (rust_words[i][1] - ref_words[i][1]) * 1000.0 for i in range(len(ref_words))
    ]
    middle_diffs_ms = [
        ((rust_words[i][0] + rust_words[i][1]) / 2.0 - (ref_words[i][0] + ref_words[i][1]) / 2.0)
        * 1000.0
        for i in range(len(ref_words))
    ]
    return (start_diffs_ms, end_diffs_ms, middle_diffs_ms)


def discover_baseline_rust_pairs(directory: Path) -> list[tuple[Path, Path, str]]:
    """
    In `directory` (non-recursive), find baseline TextGrid files and matching
    *_rust_{mode}.TextGrid files. Return list of (baseline_path, rust_path, mode).
    """
    if not directory.is_dir():
        directory = directory.parent
    textgrids = list(directory.glob("*.TextGrid"))
    baselines = [p for p in textgrids if "_rust_" not in p.name]
    rust_files = [p for p in textgrids if "_rust_" in p.name]
    pairs: list[tuple[Path, Path, str]] = []
    for baseline in baselines:
        stem = baseline.stem
        prefix = f"{stem}_rust_"
        for rust_path in rust_files:
            if rust_path.name.startswith(prefix) and rust_path.suffix == ".TextGrid":
                mode = rust_path.name[len(prefix) : -len(".TextGrid")]
                pairs.append((baseline, rust_path, mode))
    return pairs


def directories_with_textgrids(root: Path) -> list[Path]:
    """Recursively find all directories that contain at least one .TextGrid file."""
    out: list[Path] = []
    for path in root.rglob("*.TextGrid"):
        out.append(path.parent)
    return sorted(set(out))


def load_pair_words(
    baseline_path: Path, rust_path: Path
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
    """Load both TextGrids and return (ref_words, rust_words) or None on error."""
    try:
        ref_intervals = parse_textgrid_words(baseline_path)
        rust_intervals = parse_textgrid_words(rust_path)
    except (OSError, ValueError):
        return None
    ref_words = get_word_intervals(ref_intervals)
    rust_words = get_word_intervals(rust_intervals)
    if not ref_words or not rust_words or len(ref_words) != len(rust_words):
        return None
    return (ref_words, rust_words)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline and *_rust_* TextGrid files under LibriSpeech root; output global median of three metrics."
    )
    parser.add_argument(
        "librispeech_path",
        type=Path,
        help="Path to LibriSpeech root (e.g. test-data/LibriSpeech); scans recursively for baseline + *_rust_*.TextGrid pairs",
    )
    args = parser.parse_args()

    root = args.librispeech_path.resolve()
    if not root.is_dir():
        print("Error: librispeech_path must be a directory.", file=sys.stderr)
        return EXIT_NO_PAIRS

    # Aggregate per mode: mode -> (start_diffs, end_diffs, middle_diffs)
    by_mode: dict[str, tuple[list[float], list[float], list[float]]] = {}

    dirs = directories_with_textgrids(root)
    any_failed = False
    for directory in dirs:
        pairs = discover_baseline_rust_pairs(directory)
        for baseline_path, rust_path, mode in pairs:
            words = load_pair_words(baseline_path, rust_path)
            if words is None:
                print(
                    f"Error: failed to compare {baseline_path} vs {rust_path}",
                    file=sys.stderr,
                )
                any_failed = True
                continue
            if mode not in by_mode:
                by_mode[mode] = ([], [], [])
            ref_words, rust_words = words
            s, e, m = collect_diffs_ms(ref_words, rust_words)
            by_mode[mode][0].extend(s)
            by_mode[mode][1].extend(e)
            by_mode[mode][2].extend(m)

    if any_failed:
        return EXIT_LOAD_ERROR

    if not by_mode:
        print(
            "Error: no baseline + *_rust_*.TextGrid pairs found under path.",
            file=sys.stderr,
        )
        return EXIT_NO_PAIRS

    # Per-mode medians: mode -> (median_start, median_end, median_middle)
    medians: dict[str, tuple[float, float, float]] = {}
    for mode in sorted(by_mode.keys()):
        start_diffs, end_diffs, middle_diffs = by_mode[mode]
        if not start_diffs:
            continue
        medians[mode] = (
            statistics.median(start_diffs),
            statistics.median(end_diffs),
            statistics.median(middle_diffs),
        )

    for mode in sorted(medians.keys()):
        ms, me, mm = medians[mode]
        print(f"[{mode}] median_start_diff_ms={ms}")
        print(f"[{mode}] median_end_diff_ms={me}")
        print(f"[{mode}] median_word_middle_diff_ms={mm}")

    # Validation: |metric| <= 5 ms for every metric and mode
    for mode in medians:
        ms, me, mm = medians[mode]
        if abs(ms) > ABS_THRESHOLD_MS or abs(me) > ABS_THRESHOLD_MS or abs(mm) > ABS_THRESHOLD_MS:
            print(
                f"Error: at least one |metric| > {ABS_THRESHOLD_MS} ms (mode={mode}).",
                file=sys.stderr,
            )
            return EXIT_ABS_THRESHOLD

    # Validation: same metric must not differ by > 0.01 ms between modes
    if len(medians) >= 2:
        modes_sorted = sorted(medians.keys())
        for i, name in enumerate(("median_start_diff_ms", "median_end_diff_ms", "median_word_middle_diff_ms")):
            vals = [medians[m][i] for m in modes_sorted]
            drift = max(vals) - min(vals)
            if drift > MODE_DRIFT_THRESHOLD_MS:
                print(
                    f"Error: {name} differs by {drift} ms between modes (max allowed {MODE_DRIFT_THRESHOLD_MS}).",
                    file=sys.stderr,
                )
                return EXIT_MODE_DRIFT

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
