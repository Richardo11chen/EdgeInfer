#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=""
PROMPT=""
MAX_NEW_TOKENS=128
DEVICE="cuda"
DTYPE="float16"
OUTDIR=""

usage() {
    cat <<EOF
Usage: $0 --model-path <path> --prompt <prompt> [options]

Options:
  --model-path <path>       Path to model directory (required)
  --prompt <prompt>         Prompt text (required)
  --max-new-tokens <N>      Max tokens to generate (default: 128)
  --device <device>         Torch device (default: cuda)
  --dtype <dtype>           Torch dtype (default: float16)
  --output-dir <dir>        Output directory (default: outputs/budget_scan/)
  -h, --help                Show this help
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --output-dir) OUTDIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$MODEL_PATH" || -z "$PROMPT" ]]; then
    echo "Error: --model-path and --prompt are required"
    usage
fi

OUTDIR="${OUTDIR:-outputs/budget_scan}"
mkdir -p "$OUTDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG="$OUTDIR/run_${TIMESTAMP}.log"
SUMMARY_CSV="$OUTDIR/summary_${TIMESTAMP}.csv"

echo "=== Budget scan started at $(date) ===" | tee "$RUN_LOG"
echo "Model: $MODEL_PATH" | tee -a "$RUN_LOG"
echo "Prompt: $PROMPT" | tee -a "$RUN_LOG"
echo "Max new tokens: $MAX_NEW_TOKENS" | tee -a "$RUN_LOG"
echo "Device: $DEVICE, Dtype: $DTYPE" | tee -a "$RUN_LOG"
echo "Output dir: $OUTDIR" | tee -a "$RUN_LOG"
echo "" | tee -a "$RUN_LOG"

run_one() {
    local mode="$1"
    local label="$2"
    local budget="${3:-}"

    local budget_arg=()
    [[ -n "$budget" ]] && budget_arg=(--gpu-layer-budget "$budget")

    echo "--- $label ---" | tee -a "$RUN_LOG"
    uv run python -m eval.run_generation \
        --model-path "$MODEL_PATH" \
        --prompt "$PROMPT" \
        --mode "$mode" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        "${budget_arg[@]}" \
        --output-csv "$OUTDIR/${label}.csv" \
        --output-log "$OUTDIR/${label}.json" \
        2>&1 | tee -a "$RUN_LOG"
    echo "" | tee -a "$RUN_LOG"
}

# Prefetch mode with various GPU layer budgets
for budget in 1 2 4 8 16 28; do
    run_one prefetch "prefetch_${budget}" "$budget"
done

# Baselines: naive and resident (no budget)
run_one naive "naive"
run_one resident "resident"

echo "=== Aggregating results ===" | tee -a "$RUN_LOG"

echo "mode,gpu_layer_budget,prefill_latency_ms,ttft_ms,decode_tokens_per_sec,peak_memory_mib,total_copy_ms,total_compute_ms" > "$SUMMARY_CSV"

for label in prefetch_1 prefetch_2 prefetch_4 prefetch_8 prefetch_16 prefetch_28 naive resident; do
    json_file="$OUTDIR/${label}.json"
    if [[ -f "$json_file" ]]; then
        mode=$(python3 -c "import json; d=json.load(open('$json_file')); print(d['mode'])")
        budget=""
        [[ "$mode" == "prefetch" ]] && budget=$(echo "$label" | sed 's/prefetch_//')
        [[ -z "$budget" ]] && budget=""

        prefill=$(python3 -c "import json; d=json.load(open('$json_file')); print(d['prefill_latency_ms'])")
        ttft=$(python3 -c "import json; d=json.load(open('$json_file')); print(d['ttft_ms'])")
        decode=$(python3 -c "import json; d=json.load(open('$json_file')); print(d['decode_tokens_per_sec'])")
        mem=$(python3 -c "import json; d=json.load(open('$json_file')); print(d['peak_memory_mib'])")
        copy=$(python3 -c "import json; d=json.load(open('$json_file')); print(d['total_copy_ms'])")
        compute=$(python3 -c "import json; d=json.load(open('$json_file')); print(d['total_compute_ms'])")

        echo "$mode,$budget,$prefill,$ttft,$decode,$mem,$copy,$compute" >> "$SUMMARY_CSV"
    else
        echo "WARNING: $json_file not found, skipping" | tee -a "$RUN_LOG"
    fi
done

echo "" | tee -a "$RUN_LOG"
echo "=== Summary ===" | tee -a "$RUN_LOG"
column -t -s, "$SUMMARY_CSV" | tee -a "$RUN_LOG"

echo "" | tee -a "$RUN_LOG"
echo "=== Budget scan completed at $(date) ===" | tee -a "$RUN_LOG"
echo "Summary CSV: $SUMMARY_CSV" | tee -a "$RUN_LOG"
echo "Per-run logs: $OUTDIR/" | tee -a "$RUN_LOG"
