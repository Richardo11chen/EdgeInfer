#!/usr/bin/env bash
set -euo pipefail

: "${EDGEINFER_QWEN3_1_7B:?EDGEINFER_QWEN3_1_7B is required}"
: "${EDGEINFER_QWEN3_8B:?EDGEINFER_QWEN3_8B is required}"
: "${EDGEINFER_PROMPTS:=eval/prompts/required_prompts.jsonl}"
: "${EDGEINFER_OUTPUT_DIR:=outputs/required}"

if [[ ! -f "$EDGEINFER_OUTPUT_DIR/1_7b_correctness/correctness.log" ]]; then
  echo "[1/6] 1.7B correctness"
  bash scripts/run_1_7b_correctness.sh
else
  echo "[1/6] 1.7B correctness (skip existing)"
fi

if [[ ! -f "$EDGEINFER_OUTPUT_DIR/1_7b_baseline/edgeinfer_resident/edgeinfer_result.json" || ! -f "$EDGEINFER_OUTPUT_DIR/1_7b_baseline/vllm/vllm_result.json" ]]; then
  echo "[2/6] 1.7B resident baseline"
  bash scripts/run_1_7b_baseline.sh
else
  echo "[2/6] 1.7B resident baseline (skip existing)"
fi

if [[ ! -f "$EDGEINFER_OUTPUT_DIR/8b_naive_offload/edgeinfer_result.json" ]]; then
  echo "[3/6] 8B naive offload"
  bash scripts/run_8b_naive_offload.sh
else
  echo "[3/6] 8B naive offload (skip existing)"
fi

if [[ ! -f "$EDGEINFER_OUTPUT_DIR/8b_prefetch_offload/edgeinfer_result.json" ]]; then
  echo "[4/6] 8B prefetch offload"
  bash scripts/run_8b_prefetch_offload.sh
else
  echo "[4/6] 8B prefetch offload (skip existing)"
fi

if [[ ! -f "$EDGEINFER_OUTPUT_DIR/8b_vllm_offload/vllm_result.json" ]]; then
  echo "[5/6] 8B vLLM cpu offload"
  bash scripts/run_8b_vllm_offload.sh
else
  echo "[5/6] 8B vLLM cpu offload (skip existing)"
fi

if [[ ! -f "$EDGEINFER_OUTPUT_DIR/8b_budget_scan/budget_scan_summary.csv" ]]; then
  echo "[6/6] 8B budget scan"
  bash scripts/run_budget_scan.sh
else
  echo "[6/6] 8B budget scan (skip existing)"
fi

echo "Aggregating required results"
PYTHONPATH=. uv run python eval/aggregate_results.py --base-dir "$EDGEINFER_OUTPUT_DIR"
