#!/usr/bin/env python3
"""
Export a CTC speech model from local path or Hugging Face to ONNX.

Examples:
  # Export from HF
  python scripts/export_ctc_model_to_onnx.py \
      --model-source hf \
      --model-id-or-path facebook/wav2vec2-base-960h \
      --output-path models/onnx_wav2vec2_base_960h/model.onnx

  # Export from local model directory
  python scripts/export_ctc_model_to_onnx.py \
      --model-source local \
      --model-id-or-path ./my-local-model \
      --output-path ./model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a local or Hugging Face CTC model to ONNX.",
    )
    parser.add_argument(
        "--model-source",
        choices=("auto", "local", "hf"),
        default="auto",
        help=(
            "Model source type. "
            "'auto' treats existing local paths as local, otherwise Hugging Face."
        ),
    )
    parser.add_argument(
        "--model-id-or-path",
        required=True,
        help="Local model directory/file or Hugging Face repo id.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Local ONNX output file path.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Export device (default: cpu).",
    )
    parser.add_argument(
        "--dummy-seconds",
        type=float,
        default=10.0,
        help="Duration of dummy audio input used during export (default: 10.0s).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Dummy input sample rate (default: 16000).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision/branch/tag.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token (otherwise uses local auth/env).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code when loading model from Hugging Face.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated ONNX if 'onnx' package is installed.",
    )
    return parser.parse_args()


def resolve_model_ref(model_source: str, model_id_or_path: str) -> tuple[str, str]:
    candidate = Path(model_id_or_path)
    exists_locally = candidate.exists()

    if model_source == "local":
        if not exists_locally:
            raise SystemExit(f"Local model path does not exist: {candidate}")
        return "local", str(candidate.resolve())

    if model_source == "hf":
        return "hf", model_id_or_path

    # auto
    if exists_locally:
        return "local", str(candidate.resolve())
    return "hf", model_id_or_path


def choose_device(requested: str):
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(requested)


class _CtcForwardWrapper:  # pylint: disable=too-few-public-methods
    def __init__(self, model):
        self.model = model

    def __call__(self, input_values):
        return self.model(input_values=input_values).logits


def export_onnx(args: argparse.Namespace) -> Path:
    import torch
    from transformers import AutoModelForCTC

    source_kind, model_ref = resolve_model_ref(args.model_source, args.model_id_or_path)
    device = choose_device(args.device)
    print(f"Model source: {source_kind} ({model_ref})")
    print(f"Export device: {device}")

    load_kwargs: dict = {}
    if args.revision:
        load_kwargs["revision"] = args.revision
    if args.hf_token:
        load_kwargs["token"] = args.hf_token
    if args.cache_dir:
        load_kwargs["cache_dir"] = str(args.cache_dir.resolve())
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    model = AutoModelForCTC.from_pretrained(model_ref, **load_kwargs)
    model.eval()
    model.to(device)

    # Dynamic time axis export: [batch, num_samples] -> [batch, num_frames, vocab]
    input_samples = max(1, int(args.sample_rate * args.dummy_seconds))
    dummy_input = torch.zeros((1, input_samples), dtype=torch.float32, device=device)
    wrapper = _CtcForwardWrapper(model)

    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "Exporting ONNX...",
        f"opset={args.opset}, input_samples={input_samples}, output={output_path}",
    )
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input_values"],
            output_names=["logits"],
            dynamic_axes={
                "input_values": {0: "batch_size", 1: "num_samples"},
                "logits": {0: "batch_size", 1: "num_frames"},
            },
        )
    print(f"ONNX export complete: {output_path}")
    return output_path


def maybe_validate_onnx(path: Path) -> None:
    try:
        import onnx
    except ModuleNotFoundError:
        print("Validation skipped: 'onnx' package not installed.")
        return

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    print("ONNX validation passed.")


def main() -> int:
    args = parse_args()
    try:
        output = export_onnx(args)
        if args.validate:
            maybe_validate_onnx(output)
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
