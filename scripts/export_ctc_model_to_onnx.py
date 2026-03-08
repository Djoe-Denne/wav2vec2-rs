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
import shutil
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


def copy_supporting_files(
    source_kind: str,
    model_ref: str,
    output_dir: Path,
    load_kwargs: dict,
) -> None:
    """
    Copy/export non-ONNX files needed to run the exported model.
    Writes config/tokenizer/processor files into output_dir.
    """
    from transformers import AutoConfig

    output_dir.mkdir(parents=True, exist_ok=True)
    copied_labels: list[str] = []

    # Always persist config.json.
    cfg = AutoConfig.from_pretrained(model_ref, **load_kwargs)
    cfg.save_pretrained(str(output_dir))
    copied_labels.append("config")

    # Try tokenizer (writes vocab.json for wav2vec2 CTC models).
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_ref, **load_kwargs)
        tok.save_pretrained(str(output_dir))
        copied_labels.append("tokenizer")
    except Exception as exc:
        print(f"Warning: tokenizer export skipped ({exc})", file=sys.stderr)

    # Try feature extractor / processor for completeness.
    try:
        from transformers import AutoFeatureExtractor

        fe = AutoFeatureExtractor.from_pretrained(model_ref, **load_kwargs)
        fe.save_pretrained(str(output_dir))
        copied_labels.append("feature_extractor")
    except Exception:
        pass

    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_ref, **load_kwargs)
        processor.save_pretrained(str(output_dir))
        copied_labels.append("processor")
    except Exception:
        pass

    # If source is local, also copy common sidecar files not always emitted by save_pretrained.
    if source_kind == "local":
        src_dir = Path(model_ref)
        common_files = (
            "vocab.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
            "feature_extractor_config.json",
            "merges.txt",
            "added_tokens.json",
            "spiece.model",
            "sentencepiece.bpe.model",
        )
        copied_sidecars = 0
        for name in common_files:
            src = src_dir / name
            dst = output_dir / name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                copied_sidecars += 1
        if copied_sidecars:
            copied_labels.append(f"{copied_sidecars}_sidecar_file(s)")

    print(f"Supporting files exported: {', '.join(copied_labels)}")

    required_for_wav2vec2_rs = ("config.json", "vocab.json")
    missing = [name for name in required_for_wav2vec2_rs if not (output_dir / name).exists()]
    if missing:
        print(
            f"Warning: missing expected file(s) for wav2vec2-rs runtime: {', '.join(missing)}",
            file=sys.stderr,
        )


def choose_device(requested: str):
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print(
            "CUDA requested but torch.cuda.is_available() is false. Falling back to CPU.",
            file=sys.stderr,
        )
        requested = "cpu"
    return torch.device(requested)


class _CtcForwardWrapper:  # pylint: disable=too-few-public-methods
    """nn.Module wrapper returning only logits for ONNX export."""

    def __init__(self, model):
        import torch.nn as nn

        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(model)!r}")
        self._impl = model

    def as_module(self):
        import torch.nn as nn

        class _Wrapper(nn.Module):
            def __init__(self, impl):
                super().__init__()
                self.impl = impl

            def forward(self, input_values):
                return self.impl(input_values=input_values).logits

        return _Wrapper(self._impl)


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
    wrapper = _CtcForwardWrapper(model).as_module()
    wrapper.eval()

    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "Exporting ONNX...",
        f"opset={args.opset}, input_samples={input_samples}, output={output_path}",
    )
    common_export_kwargs = dict(
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
    with torch.inference_mode():
        try:
            # Prefer modern exporter first.
            torch.onnx.export(
                wrapper,
                dummy_input,
                str(output_path),
                dynamo=True,
                **common_export_kwargs,
            )
        except Exception as exc:
            # Fallback for models/pathways that still fail with torch.export.
            print(
                f"Modern ONNX export failed ({exc}). Retrying with legacy exporter...",
                file=sys.stderr,
            )
            torch.onnx.export(
                wrapper,
                dummy_input,
                str(output_path),
                dynamo=False,
                **common_export_kwargs,
            )
    print(f"ONNX export complete: {output_path}")

    copy_supporting_files(
        source_kind=source_kind,
        model_ref=model_ref,
        output_dir=output_path.parent,
        load_kwargs=load_kwargs,
    )
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
