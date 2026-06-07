from __future__ import annotations

import argparse
import csv
import json
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


def _load_prompts(prompts_file: str) -> list[str]:
    prompts: list[str] = []
    with open(prompts_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_file}")
    return prompts


def _write_prompt_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_index",
                "prompt_length",
                "generated_tokens",
                "prefill_latency_ms",
                "ttft_ms",
                "decode_tokens_per_sec",
                "peak_memory_mib",
                "total_copy_ms",
                "total_compute_ms",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def _write_summary_csv(path: Path, result_json: dict[str, object]) -> None:
    summary = result_json["summary"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "value"])
        writer.writerow(["framework", result_json["framework"]])
        writer.writerow(["mode", result_json["mode"]])
        writer.writerow(["model_path", result_json["model_path"]])
        writer.writerow(["dtype", result_json["dtype"]])
        writer.writerow(["gpu_layer_budget", result_json["gpu_layer_budget"]])
        writer.writerow(["max_new_tokens", result_json["max_new_tokens"]])
        writer.writerow(["num_prompts", summary["num_prompts"]])
        writer.writerow(["avg_prompt_length", summary["avg_prompt_length"]])
        writer.writerow(["avg_prefill_latency_ms", summary["avg_prefill_latency_ms"]])
        writer.writerow(["avg_ttft_ms", summary["avg_ttft_ms"]])
        writer.writerow(["avg_decode_tokens_per_sec", summary["avg_decode_tokens_per_sec"]])
        writer.writerow(["peak_memory_mib", summary["peak_memory_mib"]])
        writer.writerow(["total_copy_ms", summary["total_copy_ms"]])
        writer.writerow(["total_compute_ms", summary["total_compute_ms"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="EdgeInfer batch benchmark runner")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts-file", required=True, help="JSONL file with 'prompt' field")
    parser.add_argument("--mode", required=True, choices=["resident", "naive", "prefetch"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu-layer-budget", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    prompts = _load_prompts(args.prompts_file)
    torch_device = torch.device(args.device)
    torch_dtype = _parse_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    bundle = open_model_bundle(args.model_path)
    model = Qwen3Model(bundle.config)
    provider = create_weight_provider(
        model_bundle=bundle,
        mode=args.mode,
        device=torch_device,
        dtype=torch_dtype,
        gpu_layer_budget=args.gpu_layer_budget,
        benchmark=None,
    )

    print(f"Loaded {len(prompts)} prompts")
    print(f"Mode: {args.mode}, device: {args.device}, dtype: {args.dtype}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    if args.gpu_layer_budget is not None:
        print(f"gpu_layer_budget: {args.gpu_layer_budget}")

    prompt_results: list[dict[str, object]] = []
    peak_memory_mib = 0.0
    total_copy_ms = 0.0
    total_compute_ms = 0.0

    for i, prompt_text in enumerate(prompts):
        encoded = tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(torch_device)

        benchmark = BenchmarkHarness()
        if hasattr(provider, "benchmark"):
            provider.benchmark = benchmark
        runtime = GenerationRuntime(
            model=model,
            provider=provider,
            config=bundle.config,
            benchmark=benchmark,
        )

        try:
            with torch.no_grad():
                generated = runtime.generate(input_ids=input_ids, max_new_tokens=args.max_new_tokens)
        finally:
            if hasattr(provider, "benchmark"):
                provider.benchmark = None

        summary = benchmark.get_summary()
        generated_tokens = generated.shape[1] - input_ids.shape[1]
        prompt_result = {
            "prompt_index": i,
            "prompt_length": int(input_ids.shape[1]),
            "generated_tokens": int(generated_tokens),
            "prefill_latency_ms": round(summary["prefill_latency_ms"], 4),
            "ttft_ms": round(summary["ttft_ms"], 4),
            "decode_tokens_per_sec": round(summary["decode_tokens_per_sec"], 4),
            "peak_memory_mib": round(summary["peak_memory_mib"], 4),
            "total_copy_ms": round(summary["total_copy_ms"], 4),
            "total_compute_ms": round(summary["total_compute_ms"], 4),
            "generated_text": tokenizer.decode(generated[0], skip_special_tokens=True),
        }
        prompt_results.append(prompt_result)

        peak_memory_mib = max(peak_memory_mib, summary["peak_memory_mib"])
        total_copy_ms += summary["total_copy_ms"]
        total_compute_ms += summary["total_compute_ms"]

        print(
            f"  [{i + 1}/{len(prompts)}] prompt_len={input_ids.shape[1]} "
            f"gen={generated_tokens} ttft={summary['ttft_ms']:.2f}ms "
            f"decode={summary['decode_tokens_per_sec']:.4f}tok/s"
        )

    avg_prompt_len = sum(r["prompt_length"] for r in prompt_results) / len(prompt_results)
    avg_prefill = sum(r["prefill_latency_ms"] for r in prompt_results) / len(prompt_results)
    avg_ttft = sum(r["ttft_ms"] for r in prompt_results) / len(prompt_results)
    avg_decode = sum(r["decode_tokens_per_sec"] for r in prompt_results) / len(prompt_results)

    result_json = {
        "framework": "edgeinfer",
        "model_path": args.model_path,
        "mode": args.mode,
        "dtype": args.dtype,
        "gpu_layer_budget": args.gpu_layer_budget,
        "max_new_tokens": args.max_new_tokens,
        "num_prompts": len(prompt_results),
        "avg_prompt_length": round(avg_prompt_len, 4),
        "avg_prefill_latency_ms": round(avg_prefill, 4),
        "avg_ttft_ms": round(avg_ttft, 4),
        "avg_decode_tokens_per_sec": round(avg_decode, 4),
        "peak_memory_mib": round(peak_memory_mib, 4),
        "total_copy_ms": round(total_copy_ms, 4),
        "total_compute_ms": round(total_compute_ms, 4),
        "results": prompt_results,
        "summary": {
            "num_prompts": len(prompt_results),
            "avg_prompt_length": round(avg_prompt_len, 4),
            "avg_prefill_latency_ms": round(avg_prefill, 4),
            "avg_ttft_ms": round(avg_ttft, 4),
            "avg_decode_tokens_per_sec": round(avg_decode, 4),
            "peak_memory_mib": round(peak_memory_mib, 4),
            "total_copy_ms": round(total_copy_ms, 4),
            "total_compute_ms": round(total_compute_ms, 4),
        },
    }

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "edgeinfer_result.json").write_text(json.dumps(result_json, indent=2), encoding="utf-8")
    _write_prompt_csv(outdir / "prompt_results.csv", prompt_results)
    _write_summary_csv(outdir / "summary.csv", result_json)
    print(f"Results written to {outdir}")
    provider.close()


if __name__ == "__main__":
    main()
