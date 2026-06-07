from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

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


def _build_runtime(
    model_path: str,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    gpu_layer_budget: int | None,
) -> tuple[AutoTokenizer, GenerationRuntime]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    bundle = open_model_bundle(model_path)
    model = Qwen3Model(bundle.config)
    provider = create_weight_provider(
        model_bundle=bundle,
        mode=mode,
        device=device,
        dtype=dtype,
        gpu_layer_budget=gpu_layer_budget,
        benchmark=None,
    )
    runtime = GenerationRuntime(
        model=model,
        provider=provider,
        config=bundle.config,
        benchmark=None,
    )
    return tokenizer, runtime


def _write_summary(path: Path, prof: torch.profiler.profile) -> None:
    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    summary = prof.key_averages().table(
        sort_by=sort_key,
        row_limit=120,
    )
    path.write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile naive/prefetch offload timeline")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mode", required=True, choices=["naive", "prefetch"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--gpu-layer-budget", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)

    if device.type != "cuda":
        raise ValueError("profile_offload_timeline.py requires --device cuda for timeline evidence")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    tokenizer, runtime = _build_runtime(
        model_path=args.model_path,
        mode=args.mode,
        device=device,
        dtype=dtype,
        gpu_layer_budget=args.gpu_layer_budget,
    )

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{args.mode}_trace.json"
    summary_path = output_dir / f"{args.mode}_profiler_summary.txt"

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.no_grad():
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            with_stack=False,
            acc_events=True,
        ) as prof:
            runtime.generate(input_ids=input_ids, max_new_tokens=args.max_new_tokens)
            torch.cuda.synchronize(device)

    prof.export_chrome_trace(str(trace_path))
    _write_summary(summary_path, prof)
    runtime.provider.close()

    print(f"Wrote {trace_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
