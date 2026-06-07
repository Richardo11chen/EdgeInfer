#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${EDGEINFER_QWEN3_8B:-${1:-}}
PROMPTS_FILE=${EDGEINFER_PROMPTS:-${2:-eval/prompts/required_prompts.jsonl}}
OUTPUT_ROOT=${EDGEINFER_OUTPUT_DIR:-outputs/required}
MAX_NEW_TOKENS=${EDGEINFER_8B_MAX_NEW_TOKENS:-256}

if [[ -z "$MODEL_PATH" ]]; then
  echo "EDGEINFER_QWEN3_8B or positional model path is required" >&2
  exit 1
fi

OUTDIR="$OUTPUT_ROOT/8b_naive_offload"
mkdir -p "$OUTDIR"

PYTHONPATH=. uv run python eval/run_edgeinfer_batch.py \
  --model-path "$MODEL_PATH" \
  --prompts-file "$PROMPTS_FILE" \
  --mode naive \
  --device cuda \
  --dtype float16 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output-dir "$OUTDIR"
