#!/usr/bin/env python3
"""
Compare baseline and Rust-generated TextGrid files under a dataset root.
Takes the path to a dataset root; recursively finds every directory containing
baseline .TextGrid and Rust variant TextGrid pairs. By default, it compares
only plain `*_rust.TextGrid` files. Optionally, it can also include legacy
`*_rust_*.TextGrid` mode files.

Aggregates all word-pair diffs and outputs a single global median for each
metric per mode:
  - median_start_diff_ms: median of (xmin_rust - xmin_ref) over all words, in ms
  - median_end_diff_ms: median of (xmax_rust - xmax_ref) over all words, in ms
  - median_word_middle_diff_ms: median of (middle_rust - middle_ref) over all words, in ms

Exit codes (unique, for CI):
  0  Success.
  1  No baseline + Rust TextGrid pairs found, or path is not a directory.
  2  (Optional) At least one |metric| > threshold for some mode.
  3  (Optional) Same metric differs by more than threshold between modes.
  4  One or more baseline/rust pair failed to load or compare (strict mode only).
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from pathlib import Path

EXIT_SUCCESS = 0
EXIT_NO_PAIRS = 1
EXIT_ABS_THRESHOLD = 2  # |metric| > threshold (optional)
EXIT_MODE_DRIFT = 3  # same metric differs > threshold between modes (optional)
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
                        if not (math.isnan(xmin) or math.isnan(xmax)):
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


def discover_baseline_rust_pairs(
    directory: Path, include_rust_modes: bool = False
) -> list[tuple[Path, Path, str]]:
    """
    In `directory` (non-recursive), find baseline TextGrid files and matching
    Rust variants. Return list of (baseline_path, rust_path, mode).

    Always considered:
      - baseline: stem.TextGrid
      - rust default: stem_rust.TextGrid   -> mode="rust"

    Optional (include_rust_modes=True):
      - rust modes: stem_rust_{mode}.TextGrid -> mode="rust_{mode}"
    """
    if not directory.is_dir():
        directory = directory.parent
    textgrids = sorted(directory.glob("*.TextGrid"))
    baselines = [
        p
        for p in textgrids
        if "_rust_" not in p.stem and not p.stem.endswith("_rust")
    ]
    pairs: list[tuple[Path, Path, str]] = []
    for baseline in baselines:
        stem = baseline.stem

        # Plain rust output: stem_rust.TextGrid
        plain_rust = baseline.with_name(f"{stem}_rust.TextGrid")
        if plain_rust.exists():
            pairs.append((baseline, plain_rust, "rust"))

        # Legacy/extra rust modes: stem_rust_{mode}.TextGrid
        if include_rust_modes:
            prefix = f"{stem}_rust_"
            for rust_path in textgrids:
                if rust_path.name.startswith(prefix) and rust_path.suffix == ".TextGrid":
                    mode = rust_path.name[len(prefix) : -len(".TextGrid")]
                    pairs.append((baseline, rust_path, f"rust_{mode}"))
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
    """Load both TextGrids and return (ref_words, rust_words) or None on parse/shape error."""
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
        description=(
            "Compare baseline and Rust TextGrid files under a dataset root; "
            "output global medians for start/end/middle deltas."
        )
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help=(
            "Path to dataset root (e.g. test-data/MultilingualLibrispeech or "
            "test-data/AfricanAccentedFrench). Scans recursively."
        ),
    )
    parser.add_argument(
        "--include-rust-modes",
        action="store_true",
        help="Also include legacy variants named stem_rust_{mode}.TextGrid.",
    )
    parser.add_argument(
        "--strict-load-errors",
        action="store_true",
        help="Exit with code 4 when any pair fails to load/compare.",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help=(
            "Fail with CI-style exit codes when thresholds are exceeded "
            "(default: report thresholds as warnings only)."
        ),
    )
    parser.add_argument(
        "--max-error-lines",
        type=int,
        default=0,
        help="Maximum number of per-pair mismatch lines to print (default: 0).",
    )
    args = parser.parse_args()

    root = args.dataset_path.resolve()
    if not root.is_dir():
        print("Error: dataset_path must be a directory.", file=sys.stderr)
        return EXIT_NO_PAIRS

    # Aggregate per mode: mode -> (start_diffs, end_diffs, middle_diffs)
    by_mode: dict[str, tuple[list[float], list[float], list[float]]] = {}

    dirs = directories_with_textgrids(root)
    failed_count = 0
    printed_errors = 0
    for directory in dirs:
        pairs = discover_baseline_rust_pairs(
            directory, include_rust_modes=args.include_rust_modes
        )
        for baseline_path, rust_path, mode in pairs:
            words = load_pair_words(baseline_path, rust_path)
            if words is None:
                failed_count += 1
                if printed_errors < args.max_error_lines:
                    print(
                        f"Skip: failed to compare {baseline_path} vs {rust_path}",
                        file=sys.stderr,
                    )
                    printed_errors += 1
                continue
            if mode not in by_mode:
                by_mode[mode] = ([], [], [])
            ref_words, rust_words = words
            s, e, m = collect_diffs_ms(ref_words, rust_words)
            by_mode[mode][0].extend(s)
            by_mode[mode][1].extend(e)
            by_mode[mode][2].extend(m)

    if args.strict_load_errors and failed_count > 0:
        return EXIT_LOAD_ERROR

    if not by_mode:
        print(
            "Error: no baseline + Rust TextGrid pairs found under path.",
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

    if failed_count > 0 and not args.strict_load_errors:
        suppressed = max(0, failed_count - printed_errors)
        if suppressed > 0:
            print(
                f"Warning: {failed_count} pair(s) failed to compare ({suppressed} not shown).",
                file=sys.stderr,
            )
        else:
            print(
                f"Warning: {failed_count} pair(s) failed to compare.",
                file=sys.stderr,
            )

    # Optional CI-style validation.
    # By default, thresholds are informational to keep this script useful for manual exploration.
    threshold_violations: list[str] = []
    for mode in medians:
        ms, me, mm = medians[mode]
        if (
            abs(ms) > ABS_THRESHOLD_MS
            or abs(me) > ABS_THRESHOLD_MS
            or abs(mm) > ABS_THRESHOLD_MS
        ):
            threshold_violations.append(
                f"{mode}: |metric| > {ABS_THRESHOLD_MS} ms "
                f"(start={ms}, end={me}, middle={mm})"
            )

    drift_violations: list[str] = []
    if len(medians) >= 2:
        modes_sorted = sorted(medians.keys())
        metric_names = (
            "median_start_diff_ms",
            "median_end_diff_ms",
            "median_word_middle_diff_ms",
        )
        for i, name in enumerate(metric_names):
            vals = [medians[m][i] for m in modes_sorted]
            drift = max(vals) - min(vals)
            if drift > MODE_DRIFT_THRESHOLD_MS:
                drift_violations.append(
                    f"{name}: drift={drift} ms (max allowed {MODE_DRIFT_THRESHOLD_MS})"
                )

    if threshold_violations:
        for msg in threshold_violations:
            print(f"Warning: {msg}", file=sys.stderr)
        if args.fail_on_threshold:
            return EXIT_ABS_THRESHOLD

    if drift_violations:
        for msg in drift_violations:
            print(f"Warning: {msg}", file=sys.stderr)
        if args.fail_on_threshold:
            return EXIT_MODE_DRIFT

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
