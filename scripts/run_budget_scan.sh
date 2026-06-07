#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${EDGEINFER_QWEN3_8B:-${1:-}}
PROMPTS_FILE=${EDGEINFER_PROMPTS:-${2:-eval/prompts/required_prompts.jsonl}}
OUTPUT_ROOT=${EDGEINFER_OUTPUT_DIR:-outputs/required}
BUDGETS=${EDGEINFER_BUDGET_SCAN_VALUES:-"1 2 4 8 16"}
MAX_NEW_TOKENS=${EDGEINFER_8B_MAX_NEW_TOKENS:-256}

if [[ -z "$MODEL_PATH" ]]; then
  echo "EDGEINFER_QWEN3_8B or positional model path is required" >&2
  exit 1
fi

OUTDIR="$OUTPUT_ROOT/8b_budget_scan"
mkdir -p "$OUTDIR"
SUMMARY_CSV="$OUTDIR/budget_scan_summary.csv"

echo "framework,mode,gpu_layer_budget,avg_prefill_latency_ms,avg_ttft_ms,avg_decode_tokens_per_sec,peak_memory_mib,total_copy_ms,total_compute_ms,measurement_notes" > "$SUMMARY_CSV"

for budget in $BUDGETS; do
  RUN_DIR="$OUTDIR/budget_${budget}"
  echo "Running prefetch budget=$budget"
  PYTHONPATH=. uv run python eval/run_edgeinfer_batch.py \
    --model-path "$MODEL_PATH" \
    --prompts-file "$PROMPTS_FILE" \
    --mode prefetch \
    --device cuda \
    --dtype float16 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --gpu-layer-budget "$budget" \
    --output-dir "$RUN_DIR"
  PYTHONPATH=. uv run python - <<PY >> "$SUMMARY_CSV"
import json
from pathlib import Path
data = json.loads(Path("$RUN_DIR/edgeinfer_result.json").read_text(encoding="utf-8"))
print(",".join([
    data["framework"],
    data["mode"],
    str(data["gpu_layer_budget"]),
    str(data["avg_prefill_latency_ms"]),
    str(data["avg_ttft_ms"]),
    str(data["avg_decode_tokens_per_sec"]),
    str(data["peak_memory_mib"]),
    str(data["total_copy_ms"]),
    str(data["total_compute_ms"]),
    "",
]))
PY
done

PYTHONPATH=. uv run python eval/plot_budget_scan.py \
  --summary-csv "$SUMMARY_CSV" \
  --output-dir "$OUTPUT_ROOT/figures"

cp "$SUMMARY_CSV" "$OUTDIR/plot_data.csv"
