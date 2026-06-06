from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.benchmark import BenchmarkHarness
from eval.correctness import compare_logits, compare_tokens
from model.kv_cache import KVCache
from model.qwen3 import Qwen3Model
from runtime.generation import GenerationRuntime
from runtime.weight_provider import create_weight_provider
from weights.model_bundle import open_model_bundle


def _parse_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def _hf_greedy_generate(
    hf_model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    if max_new_tokens == 0:
        return input_ids

    generated = input_ids
    outputs = hf_model(input_ids=input_ids, use_cache=True)
    logits = outputs.logits
    past_key_values = outputs.past_key_values

    for step in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if step == max_new_tokens - 1:
            break
        outputs = hf_model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits
        past_key_values = outputs.past_key_values

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="EdgeInfer vs HuggingFace comparison")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--mode", default="prefetch", choices=["resident", "naive", "prefetch"])
    parser.add_argument("--gpu-layer-budget", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/comparison")
    args = parser.parse_args()

    torch_device = torch.device(args.device)
    torch_dtype = _parse_dtype(args.dtype)

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Loading HuggingFace reference model ...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(torch_device)
    hf_model.eval()

    print("Loading EdgeInfer runtime ...")
    bundle = open_model_bundle(args.model_path)
    model = Qwen3Model(bundle.config)
    rt_benchmark = BenchmarkHarness()
    provider = create_weight_provider(
        model_bundle=bundle,
        mode=args.mode,
        device=torch_device,
        dtype=torch_dtype,
        gpu_layer_budget=args.gpu_layer_budget,
        benchmark=rt_benchmark,
    )
    runtime = GenerationRuntime(
        model=model,
        provider=provider,
        config=bundle.config,
        benchmark=rt_benchmark,
    )

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(torch_device)
    prompt_len = input_ids.shape[1]
    max_len = prompt_len + args.max_new_tokens

    print(f"Prompt length: {prompt_len} tokens, max new tokens: {args.max_new_tokens}")

    # --- Correctness: prefill logits ---
    print("\n=== Correctness: Prefill Logits ===")
    kv_cache = KVCache(
        config=bundle.config,
        batch_size=input_ids.shape[0],
        max_sequence_length=max_len,
        device=torch_device,
        dtype=torch_dtype,
    )
    with torch.no_grad():
        hf_outputs = hf_model(input_ids=input_ids, use_cache=True)
        hf_logits = hf_outputs.logits
        rt_logits = runtime.prefill(input_ids, kv_cache)
    logits_result = compare_logits(rt_logits, hf_logits)
    print(
        f"  passed={logits_result.passed} "
        f"max_abs_error={logits_result.max_abs_error} "
        f"max_rel_error={logits_result.max_rel_error}"
    )

    # --- Correctness: generated tokens ---
    print("\n=== Correctness: Generated Tokens ===")
    with torch.no_grad():
        hf_tokens = _hf_greedy_generate(hf_model, input_ids, args.max_new_tokens)
        rt_tokens = runtime.generate(input_ids=input_ids, max_new_tokens=args.max_new_tokens)
    tokens_result = compare_tokens(rt_tokens, hf_tokens)
    print(f"  passed={tokens_result.passed} mismatch_count={tokens_result.mismatch_count}")

    # --- Performance comparison ---
    print("\n=== Performance ===")
    rt_summary = rt_benchmark.get_summary()

    hf_start = torch.cuda.Event(enable_timing=True)
    hf_end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        hf_start.record()
        _hf_greedy_generate(hf_model, input_ids, args.max_new_tokens)
        hf_end.record()
    torch.cuda.synchronize()
    hf_total_ms = hf_start.elapsed_time(hf_end)
    hf_tokens_per_sec = (args.max_new_tokens / hf_total_ms) * 1000 if hf_total_ms > 0 else float("inf")

    print(f"  HF total time:        {hf_total_ms:.2f} ms")
    print(f"  HF tokens/sec:        {hf_tokens_per_sec:.2f}")
    print(f"  RT prefill latency:   {rt_summary['prefill_latency_ms']:.2f} ms")
    print(f"  RT TTFT:              {rt_summary['ttft_ms']:.2f} ms")
    print(f"  RT decode tokens/sec: {rt_summary['decode_tokens_per_sec']:.2f}")
    print(f"  RT peak memory:       {rt_summary['peak_memory_mib']:.2f} MiB")
    print(f"  RT total copy:        {rt_summary['total_copy_ms']:.2f} ms")
    print(f"  RT total compute:     {rt_summary['total_compute_ms']:.2f} ms")

    # --- Output ---
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_json = outdir / "comparison_result.json"
    result = {
        "model_path": args.model_path,
        "mode": args.mode,
        "device": args.device,
        "dtype": args.dtype,
        "prompt_length": prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "correctness": {
            "logits_passed": logits_result.passed,
            "max_abs_error": logits_result.max_abs_error,
            "max_rel_error": logits_result.max_rel_error,
            "tokens_passed": tokens_result.passed,
            "token_mismatch_count": tokens_result.mismatch_count,
        },
        "performance": {
            "hf_total_ms": hf_total_ms,
            "hf_tokens_per_sec": hf_tokens_per_sec,
            "rt_prefill_latency_ms": rt_summary["prefill_latency_ms"],
            "rt_ttft_ms": rt_summary["ttft_ms"],
            "rt_decode_tokens_per_sec": rt_summary["decode_tokens_per_sec"],
            "rt_peak_memory_mib": rt_summary["peak_memory_mib"],
            "rt_total_copy_ms": rt_summary["total_copy_ms"],
            "rt_total_compute_ms": rt_summary["total_compute_ms"],
        },
    }
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nResults written to {output_json}")

    provider.close()

    if not tokens_result.passed:
        print("\nWARNING: Token mismatch detected!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
