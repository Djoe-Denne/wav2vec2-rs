# Git hooks

These hooks run before each commit to keep the codebase formatted and clippy-clean.

## Enable (one-time)

From the repository root:

```bash
git config core.hooksPath githooks
```

On Unix/macOS, ensure the hook is executable: `chmod +x githooks/pre-commit`.

This makes Git use the `githooks/` directory in this repo instead of `.git/hooks/`.

## What runs on commit

- **cargo fmt --check** — Commit is rejected if code is not formatted. Run `cargo fmt` and try again.
- **cargo clippy** — Same flags as CI (`--all-targets --features "onnx,report-cli,alignment-profiling" -- -D warnings`). Commit is rejected if clippy reports warnings or errors.

## Disable

To use the default Git hooks again:

```bash
git config --unset core.hooksPath
```
