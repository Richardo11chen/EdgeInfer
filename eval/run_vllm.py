from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM benchmark runner")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts-file", required=True, help="JSONL file with 'prompt' field")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cpu-offload-gb", type=float, default=None)
    parser.add_argument("--output-dir", default="outputs/comparison/vllm")
    args = parser.parse_args()

    prompts = []
    with open(args.prompts_file) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])

    print(f"Loaded {len(prompts)} prompts")

    try:
        import pynvml
        pynvml.nvmlInit()
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        _baseline_mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).used / (1024 * 1024)
        print(f"Baseline GPU memory: {_baseline_mem:.0f} MiB")
    except Exception:
        nvml_handle = None
        _baseline_mem = 0.0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    llm_kwargs: dict = {
        "model": args.model_path,
        "dtype": args.dtype,
        "enforce_eager": True,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.70,
        "max_model_len": 2048,
    }
    if args.cpu_offload_gb is not None:
        llm_kwargs["cpu_offload_gb"] = args.cpu_offload_gb

    print(f"Loading vLLM model: {args.model_path}")
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    # Warmup with first prompt to avoid first-run overhead
    warmup_sp = SamplingParams(temperature=0.0, max_tokens=4)
    llm.generate(prompts[:1], sampling_params=warmup_sp)

    # Benchmark each prompt individually for accurate per-prompt timing
    print(f"Running generation with max_new_tokens={args.max_new_tokens} ...")
    results = []
    peak_mem_mib = 0.0

    for i, prompt_text in enumerate(prompts):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = llm.generate([prompt_text], sampling_params=sampling_params)[0]
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t0

        prompt_len = len(output.prompt_token_ids)
        generated_tokens = len(output.outputs[0].token_ids)
        total_time_ms = elapsed_s * 1000.0
        tokens_per_sec = generated_tokens / elapsed_s if elapsed_s > 0 else 0.0

        if nvml_handle is not None:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            mem_used_mib = mem_info.used / (1024 * 1024)
            peak_mem_mib = max(peak_mem_mib, mem_used_mib)

        results.append({
            "prompt_length": prompt_len,
            "generated_tokens": generated_tokens,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "total_time_ms": round(total_time_ms, 2),
            "generated_text": output.outputs[0].text,
        })

        print(f"  [{i+1}/{len(prompts)}] prompt_len={prompt_len} "
              f"gen={generated_tokens} tok/s={tokens_per_sec:.2f} "
              f"total={total_time_ms:.0f}ms")

    avg_tokens = sum(r["generated_tokens"] for r in results) / len(results)
    avg_prompt_len = sum(r["prompt_length"] for r in results) / len(results)
    avg_tok_per_sec = sum(r["tokens_per_sec"] for r in results) / len(results)
    avg_total_time = sum(r["total_time_ms"] for r in results) / len(results)

    if nvml_handle is not None:
        peak_delta_mib = peak_mem_mib - _baseline_mem
    else:
        peak_delta_mib = 0.0

    print(f"\n=== vLLM Benchmark Summary ===")
    print(f"  Prompts:              {len(results)}")
    print(f"  Avg prompt length:    {avg_prompt_len:.1f} tokens")
    print(f"  Avg generated tokens: {avg_tokens:.1f}")
    print(f"  Avg tokens/sec:       {avg_tok_per_sec:.2f}")
    print(f"  Avg total time:       {avg_total_time:.0f} ms")
    print(f"  Peak memory (delta):  {peak_delta_mib:.1f} MiB")

    result_json = {
        "model_path": args.model_path,
        "framework": "vllm",
        "dtype": args.dtype,
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "cpu_offload_gb": args.cpu_offload_gb,
        },
        "results": results,
        "summary": {
            "num_prompts": len(results),
            "avg_prompt_length": round(avg_prompt_len, 1),
            "avg_generated_tokens": round(avg_tokens, 1),
            "avg_tokens_per_sec": round(avg_tok_per_sec, 2),
            "avg_total_time_ms": round(avg_total_time, 0),
            "peak_memory_mib_delta": round(peak_delta_mib, 1),
        },
    }

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "vllm_result.json"
    outpath.write_text(json.dumps(result_json, indent=2), encoding="utf-8")
    print(f"\nResults written to {outpath}")

    if nvml_handle is not None:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
