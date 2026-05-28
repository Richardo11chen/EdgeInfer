#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:?Usage: $0 <model-path> <prompt> [max-new-tokens]}
PROMPT=${2:?Usage: $0 <model-path> <prompt> [max-new-tokens]}
MAX_NEW_TOKENS=${3:-1}

OUTDIR="outputs/acceptance_2026_05_28"
mkdir -p "$OUTDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

uv run python -m eval.run_generation \
  --model-path "$MODEL_PATH" \
  --prompt "$PROMPT" \
  --mode prefetch \
  --device cuda \
  --dtype float16 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --gpu-layer-budget 2 \
  --output-csv "$OUTDIR/8b_prefetch_${TIMESTAMP}.csv" \
  --output-log "$OUTDIR/8b_prefetch_${TIMESTAMP}.log"
