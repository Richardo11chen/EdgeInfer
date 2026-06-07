#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${EDGEINFER_QWEN3_8B:-${1:-}}
PROMPTS_FILE=${EDGEINFER_PROMPTS:-${2:-eval/prompts/required_prompts.jsonl}}
OUTPUT_ROOT=${EDGEINFER_OUTPUT_DIR:-outputs/required}
CPU_OFFLOAD_GB=${EDGEINFER_VLLM_CPU_OFFLOAD_GB:-${3:-24}}
MAX_NEW_TOKENS=${EDGEINFER_8B_MAX_NEW_TOKENS:-256}

if [[ -z "$MODEL_PATH" ]]; then
  echo "EDGEINFER_QWEN3_8B or positional model path is required" >&2
  exit 1
fi

OUTDIR="$OUTPUT_ROOT/8b_vllm_offload"
mkdir -p "$OUTDIR"

PYTHONPATH=. uv run --group vllm python eval/run_vllm.py \
  --model-path "$MODEL_PATH" \
  --prompts-file "$PROMPTS_FILE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype float16 \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --output-dir "$OUTDIR"
