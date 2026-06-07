#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${EDGEINFER_QWEN3_1_7B:-${1:-}}
OUTPUT_ROOT=${EDGEINFER_OUTPUT_DIR:-outputs/required}
PROMPT=${EDGEINFER_CORRECTNESS_PROMPT:-"Explain why layer-wise offloading can reduce GPU memory pressure in one short paragraph."}
MAX_NEW_TOKENS=${EDGEINFER_CORRECTNESS_MAX_NEW_TOKENS:-16}
DEVICE=${EDGEINFER_CORRECTNESS_DEVICE:-cuda}
DTYPE=${EDGEINFER_CORRECTNESS_DTYPE:-float16}
ABS_TOL=${EDGEINFER_CORRECTNESS_ABS_TOL:-1e-2}
REL_TOL=${EDGEINFER_CORRECTNESS_REL_TOL:-1e-2}

if [[ -z "$MODEL_PATH" ]]; then
  echo "EDGEINFER_QWEN3_1_7B or positional model path is required" >&2
  exit 1
fi

OUTDIR="$OUTPUT_ROOT/1_7b_correctness"
mkdir -p "$OUTDIR"
LOG_PATH="$OUTDIR/correctness.log"

PYTHONPATH=. uv run python -m eval.correctness \
  --model-path "$MODEL_PATH" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --abs-tol "$ABS_TOL" \
  --rel-tol "$REL_TOL" | tee "$LOG_PATH"
