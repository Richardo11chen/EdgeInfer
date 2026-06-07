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
    parser = argparse.ArgumentParser(description="EdgeInfer generation acceptance runner")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mode", required=True, choices=["resident", "naive", "prefetch"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--gpu-layer-budget", type=int, default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-log", default=None)
    args = parser.parse_args()

    torch_device = torch.device(args.device)
    torch_dtype = _parse_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(torch_device)

    bundle = open_model_bundle(args.model_path)
    model = Qwen3Model(bundle.config)
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

    try:
        with torch.no_grad():
            generated = runtime.generate(input_ids=input_ids, max_new_tokens=args.max_new_tokens)
    finally:
        provider.close()

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    summary = benchmark.get_summary()

    log_entry = {
        "model_path": args.model_path,
        "mode": args.mode,
        "device": args.device,
        "dtype": args.dtype,
        "prompt_length": input_ids.shape[1],
        "max_new_tokens": args.max_new_tokens,
        "generated_token_ids": generated[0].tolist(),
        "generated_text": generated_text,
        "prefill_latency_ms": summary["prefill_latency_ms"],
        "ttft_ms": summary["ttft_ms"],
        "decode_tokens_per_sec": summary["decode_tokens_per_sec"],
        "peak_memory_mib": summary["peak_memory_mib"],
        "total_copy_ms": summary["total_copy_ms"],
        "total_compute_ms": summary["total_compute_ms"],
    }

    if args.output_log is not None:
        Path(args.output_log).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_log).write_text(json.dumps(log_entry, indent=2), encoding="utf-8")

    if args.output_csv is not None:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        benchmark.export_csv(args.output_csv)

    print(json.dumps(log_entry, indent=2))

    if not generated_text:
        print("WARNING: generated text is empty", file=sys.stderr)


if __name__ == "__main__":
    main()
