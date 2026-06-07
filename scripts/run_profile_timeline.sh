#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${EDGEINFER_QWEN3_8B:-${1:-}}
OUTPUT_ROOT=${EDGEINFER_OUTPUT_DIR:-outputs/required}
PROMPT=${EDGEINFER_PROFILE_PROMPT:-"Explain in two short sentences why overlapping H2D copies with compute can reduce visible offload stall."}
MAX_NEW_TOKENS=${EDGEINFER_PROFILE_MAX_NEW_TOKENS:-8}
PREFETCH_BUDGET=${EDGEINFER_PROFILE_GPU_LAYER_BUDGET:-2}

if [[ -z "$MODEL_PATH" ]]; then
  echo "EDGEINFER_QWEN3_8B or positional model path is required" >&2
  exit 1
fi

OUTDIR="$OUTPUT_ROOT/profile"
mkdir -p "$OUTDIR"

PYTHONPATH=. uv run python eval/profile_offload_timeline.py \
  --model-path "$MODEL_PATH" \
  --prompt "$PROMPT" \
  --mode naive \
  --device cuda \
  --dtype float16 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output-dir "$OUTDIR"

PYTHONPATH=. uv run python eval/profile_offload_timeline.py \
  --model-path "$MODEL_PATH" \
  --prompt "$PROMPT" \
  --mode prefetch \
  --device cuda \
  --dtype float16 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --gpu-layer-budget "$PREFETCH_BUDGET" \
  --output-dir "$OUTDIR"
