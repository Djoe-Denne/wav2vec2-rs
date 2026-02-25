#!/usr/bin/env python3
"""Run selected alignment integration tests listed in a text file.

Input file formats supported:
- One utterance ID per line, e.g. "4507-16021-0037"
- Full test name per line, e.g.
  "pytorch_alignment_reference_matches_within_delta::audio::4507-16021-0037"
- Lines copied from logs (IDs are extracted from the line)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


LINE_NUMBER_PREFIX_RE = re.compile(r"^L\d+:\s*")
TEST_ID_RE = re.compile(r"\b\d+-\d+-\d+\b")
FULL_TEST_RE = re.compile(r"\b[A-Za-z0-9_]+(?:::[A-Za-z0-9_]+)*::audio::\d+-\d+-\d+\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run alignment integration tests listed in a file "
            "(ID-only lines or full test names)."
        )
    )
    parser.add_argument(
        "tests_file",
        type=Path,
        help="Path to file containing IDs or test names.",
    )

    parser.add_argument(
        "--name-template",
        default="pytorch_alignment_reference_matches_within_delta::audio::{id}",
        help="Template used when a line only contains an utterance ID.",
    )
    parser.add_argument(
        "--test-target",
        default="alignment_reference",
        help="Cargo integration test target (passed to --test).",
    )
    parser.add_argument(
        "--features",
        default="cuda",
        help="Cargo features to enable (empty string disables --features).",
    )
    parser.add_argument(
        "--no-include-ignored",
        action="store_true",
        help="Do not pass --include-ignored to the test harness.",
    )

    parser.add_argument("--device", choices=["cpu", "cuda"], help="WAV2VEC2_IT_DEVICE")
    parser.add_argument(
        "--full",
        choices=["0", "1"],
        help='WAV2VEC2_IT_FULL (0=sample mode, 1=all utterances loaded).',
    )
    parser.add_argument("--seed", type=int, help="WAV2VEC2_IT_SEED")
    parser.add_argument("--delta-ms", type=float, help="WAV2VEC2_IT_DELTA_MS")
    parser.add_argument("--model-dir", help="WAV2VEC2_IT_MODEL_DIR")
    parser.add_argument("--rust-log", help="RUST_LOG")

    parser.add_argument(
        "--max-tests",
        type=int,
        default=0,
        help="Run at most N tests from the file (0 = all).",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop at first failure.",
    )
    parser.add_argument(
        "--single-process",
        action="store_true",
        help=(
            "Run one cargo test process by setting WAV2VEC2_IT_ONLY_IDS_FILE. "
            "Requires alignment_reference support for this env var."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )

    return parser.parse_args()


def normalize_line(line: str) -> str:
    line = line.strip()
    line = LINE_NUMBER_PREFIX_RE.sub("", line)
    return line.strip()


def parse_test_filters(tests_file: Path, name_template: str) -> list[str]:
    if not tests_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {tests_file}")

    filters: list[str] = []
    seen: set[str] = set()

    for raw_line in tests_file.read_text(encoding="utf-8").splitlines():
        line = normalize_line(raw_line)
        if not line or line.startswith("#"):
            continue

        full_tests = FULL_TEST_RE.findall(line)
        if full_tests:
            for test_name in full_tests:
                if test_name not in seen:
                    seen.add(test_name)
                    filters.append(test_name)
            continue

        for test_id in TEST_ID_RE.findall(line):
            test_name = name_template.format(id=test_id)
            if test_name not in seen:
                seen.add(test_name)
                filters.append(test_name)

    return filters


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()

    if args.device is not None:
        env["WAV2VEC2_IT_DEVICE"] = args.device
    if args.full is not None:
        env["WAV2VEC2_IT_FULL"] = args.full
    if args.seed is not None:
        env["WAV2VEC2_IT_SEED"] = str(args.seed)
    if args.delta_ms is not None:
        env["WAV2VEC2_IT_DELTA_MS"] = str(args.delta_ms)
    if args.model_dir is not None:
        env["WAV2VEC2_IT_MODEL_DIR"] = args.model_dir
    if args.rust_log is not None:
        env["RUST_LOG"] = args.rust_log

    return env


def build_command(
    test_filter: str,
    test_target: str,
    features: str,
    include_ignored: bool,
) -> list[str]:
    cmd = ["cargo", "test"]
    if features.strip():
        cmd.extend(["--features", features])
    cmd.extend(["--test", test_target, test_filter, "--", "--nocapture"])
    if include_ignored:
        cmd.append("--include-ignored")
    return cmd


def main() -> int:
    args = parse_args()

    try:
        test_filters = parse_test_filters(args.tests_file, args.name_template)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to parse input file: {exc}", file=sys.stderr)
        return 2

    if not test_filters:
        print(
            f"No tests found in {args.tests_file}. "
            "Expected IDs like 4507-16021-0037 or full test names.",
            file=sys.stderr,
        )
        return 2

    if args.max_tests > 0:
        test_filters = test_filters[: args.max_tests]

    env = build_env(args)
    include_ignored = not args.no_include_ignored

    print(f"Loaded {len(test_filters)} test(s) from {args.tests_file}")
    print(f"Cargo target: --test {args.test_target}")
    print(f"Include ignored: {include_ignored}")
    if args.dry_run:
        print("Mode: dry-run (commands are not executed)")

    if args.single_process:
        if args.max_tests > 0:
            print(
                "Note: --max-tests is ignored in --single-process mode; "
                "the full file is passed through WAV2VEC2_IT_ONLY_IDS_FILE."
            )
        env["WAV2VEC2_IT_ONLY_IDS_FILE"] = str(args.tests_file.resolve())
        cmd = ["cargo", "test"]
        if args.features.strip():
            cmd.extend(["--features", args.features])
        cmd.extend(["--test", args.test_target, "--", "--nocapture"])
        if include_ignored:
            print(
                "Note: --include-ignored is disabled in --single-process mode "
                "to avoid running the full suite."
            )

        command_preview = " ".join(f'"{part}"' if " " in part else part for part in cmd)
        print("\nSingle-process mode enabled")
        print(f"$ {command_preview}")

        if args.dry_run:
            return 0

        completed = subprocess.run(cmd, env=env, check=False)
        return completed.returncode

    failed: list[str] = []
    for idx, test_filter in enumerate(test_filters, start=1):
        cmd = build_command(
            test_filter=test_filter,
            test_target=args.test_target,
            features=args.features,
            include_ignored=include_ignored,
        )
        command_preview = " ".join(f'"{part}"' if " " in part else part for part in cmd)
        print(f"\n[{idx}/{len(test_filters)}] {test_filter}")
        print(f"$ {command_preview}")

        if args.dry_run:
            continue

        completed = subprocess.run(cmd, env=env, check=False)
        if completed.returncode != 0:
            failed.append(test_filter)
            if args.stop_on_fail:
                break

    passed_count = len(test_filters) - len(failed)
    print(f"\nDone. Passed: {passed_count}, Failed: {len(failed)}")

    if failed:
        print("Failed tests:")
        for test_name in failed:
            print(f"- {test_name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
