#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_1_7b_correctness.sh /path/to/Qwen3-1.7B "Hello" 8 cpu float32 1e-2 1e-2
MODEL_PATH=${1:?model path required}
PROMPT=${2:?prompt required}
MAX_NEW_TOKENS=${3:-8}
DEVICE=${4:-cpu}
DTYPE=${5:-float32}
ABS_TOL=${6:-1e-2}
REL_TOL=${7:-1e-2}

uv run python -m eval.correctness \
  --model-path "$MODEL_PATH" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --abs-tol "$ABS_TOL" \
  --rel-tol "$REL_TOL"
