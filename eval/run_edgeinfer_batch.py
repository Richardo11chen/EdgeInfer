from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from eval.benchmark import BenchmarkHarness
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


def main() -> None:
    parser = argparse.ArgumentParser(description="EdgeInfer batch benchmark runner")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts-file", required=True, help="JSONL file with 'prompt' field")
    parser.add_argument("--mode", required=True, choices=["resident", "naive", "prefetch"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu-layer-budget", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/comparison/vllm_1.7B")
    args = parser.parse_args()

    prompts = []
    with open(args.prompts_file) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])

    print(f"Loaded {len(prompts)} prompts")

    torch_device = torch.device(args.device)
    torch_dtype = _parse_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    bundle = open_model_bundle(args.model_path)
    model = Qwen3Model(bundle.config)

    print(f"Mode: {args.mode}, device: {args.device}, dtype: {args.dtype}")
    print(f"max_new_tokens: {args.max_new_tokens}")

    results = []
    peak_memory_mib = 0.0

    for i, prompt_text in enumerate(prompts):
        encoded = tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(torch_device)

        benchmark = BenchmarkHarness()
        provider = create_weight_provider(
            model_bundle=bundle,
            mode=args.mode,
            device=torch_device,
            dtype=torch_dtype,
            gpu_layer_budget=args.gpu_layer_budget,
            benchmark=benchmark,
        )
        runtime = GenerationRuntime(
            model=model,
            provider=provider,
            config=bundle.config,
            benchmark=benchmark,
        )

        with torch.no_grad():
            generated = runtime.generate(input_ids=input_ids, max_new_tokens=args.max_new_tokens)

        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        summary = benchmark.get_summary()

        results.append({
            "prompt_length": input_ids.shape[1],
            "generated_tokens": generated.shape[1] - input_ids.shape[1],
            "ttft_ms": round(summary["ttft_ms"], 2),
            "decode_tokens_per_sec": round(summary["decode_tokens_per_sec"], 2),
            "prefill_latency_ms": round(summary["prefill_latency_ms"], 2),
            "total_copy_ms": round(summary["total_copy_ms"], 2),
            "total_compute_ms": round(summary["total_compute_ms"], 2),
            "peak_memory_mib": round(summary["peak_memory_mib"], 2),
            "generated_text": generated_text,
        })

        peak_memory_mib = max(peak_memory_mib, summary["peak_memory_mib"])

        print(f"  [{i+1}/{len(prompts)}] prompt_len={input_ids.shape[1]} "
              f"gen={generated.shape[1] - input_ids.shape[1]} "
              f"ttft={summary['ttft_ms']:.0f}ms "
              f"decode={summary['decode_tokens_per_sec']:.2f}tok/s")

        provider.close()

    avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
    avg_decode = sum(r["decode_tokens_per_sec"] for r in results) / len(results)
    avg_tokens = sum(r["generated_tokens"] for r in results) / len(results)
    avg_prompt_len = sum(r["prompt_length"] for r in results) / len(results)
    avg_copy = sum(r["total_copy_ms"] for r in results) / len(results)
    avg_compute = sum(r["total_compute_ms"] for r in results) / len(results)

    print(f"\n=== EdgeInfer {args.mode} Benchmark Summary ===")
    print(f"  Prompts:              {len(results)}")
    print(f"  Avg prompt length:    {avg_prompt_len:.1f} tokens")
    print(f"  Avg generated tokens: {avg_tokens:.1f}")
    print(f"  Avg TTFT:             {avg_ttft:.2f} ms")
    print(f"  Avg decode:           {avg_decode:.2f} tok/s")
    print(f"  Avg copy time:        {avg_copy:.0f} ms")
    print(f"  Avg compute time:     {avg_compute:.0f} ms")
    print(f"  Peak memory:          {peak_memory_mib:.2f} MiB")

    result_json = {
        "model_path": args.model_path,
        "framework": "edgeinfer",
        "mode": args.mode,
        "dtype": args.dtype,
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "gpu_layer_budget": args.gpu_layer_budget,
        },
        "results": results,
        "summary": {
            "num_prompts": len(results),
            "avg_prompt_length": round(avg_prompt_len, 1),
            "avg_generated_tokens": round(avg_tokens, 1),
            "avg_ttft_ms": round(avg_ttft, 2),
            "avg_decode_tokens_per_sec": round(avg_decode, 2),
            "avg_copy_ms": round(avg_copy, 0),
            "avg_compute_ms": round(avg_compute, 0),
            "peak_memory_mib": round(peak_memory_mib, 2),
        },
    }

    outdir = Path(args.output_dir) / args.mode
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "edgeinfer_result.json"
    outpath.write_text(json.dumps(result_json, indent=2), encoding="utf-8")
    print(f"\nResults written to {outpath}")


if __name__ == "__main__":
    main()
