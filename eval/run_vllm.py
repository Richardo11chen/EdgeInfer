from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


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


def _maybe_init_nvml():
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml, handle
    except Exception:
        return None, None


def _sample_peak_memory_mib(pynvml_module, handle) -> float:
    if pynvml_module is None or handle is None:
        return 0.0
    mem_info = pynvml_module.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used / (1024 * 1024)


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
        for key in (
            "framework",
            "model_path",
            "dtype",
            "cpu_offload_gb",
            "max_new_tokens",
            "num_prompts",
            "avg_prompt_length",
            "avg_prefill_latency_ms",
            "avg_ttft_ms",
            "avg_decode_tokens_per_sec",
            "peak_memory_mib",
            "measurement_notes",
        ):
            value = result_json.get(key, summary.get(key))
            writer.writerow([key, value])


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM benchmark runner")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts-file", required=True, help="JSONL file with 'prompt' field")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cpu-offload-gb", type=float, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    prompts = _load_prompts(args.prompts_file)
    pynvml_module, nvml_handle = _maybe_init_nvml()

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    llm_kwargs: dict[str, object] = {
        "model": args.model_path,
        "dtype": args.dtype,
        "enforce_eager": True,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.7,
        "max_model_len": 2048,
    }
    if args.cpu_offload_gb is not None:
        llm_kwargs["cpu_offload_gb"] = args.cpu_offload_gb

    print(f"Loaded {len(prompts)} prompts")
    print(f"Loading vLLM model: {args.model_path}")
    llm = LLM(**llm_kwargs)

    warmup_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=min(args.max_new_tokens, 4))
    llm.generate(prompts[:1], sampling_params=warmup_params)

    greedy_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
    ttft_probe_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1)
    peak_memory_mib = 0.0
    prompt_results: list[dict[str, object]] = []

    for i, prompt_text in enumerate(prompts):
        torch.cuda.synchronize()
        ttft_start = time.perf_counter()
        ttft_output = llm.generate([prompt_text], sampling_params=ttft_probe_params)[0]
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

        torch.cuda.synchronize()
        full_start = time.perf_counter()
        output = llm.generate([prompt_text], sampling_params=greedy_params)[0]
        torch.cuda.synchronize()
        total_elapsed_s = time.perf_counter() - full_start

        generated_tokens = len(output.outputs[0].token_ids)
        decode_tokens = max(generated_tokens - 1, 0)
        decode_elapsed_s = max(total_elapsed_s - (ttft_ms / 1000.0), 0.0)
        decode_tokens_per_sec = decode_tokens / decode_elapsed_s if decode_tokens > 0 and decode_elapsed_s > 0 else 0.0
        prompt_length = len(output.prompt_token_ids)

        prompt_peak_mib = max(
            _sample_peak_memory_mib(pynvml_module, nvml_handle),
            float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0,
        )
        peak_memory_mib = max(peak_memory_mib, prompt_peak_mib)

        prompt_results.append(
            {
                "prompt_index": i,
                "prompt_length": prompt_length,
                "generated_tokens": generated_tokens,
                "prefill_latency_ms": round(ttft_ms, 4),
                "ttft_ms": round(ttft_ms, 4),
                "decode_tokens_per_sec": round(decode_tokens_per_sec, 4),
                "peak_memory_mib": round(prompt_peak_mib, 4),
                "generated_text": output.outputs[0].text,
                "ttft_probe_generated_tokens": len(ttft_output.outputs[0].token_ids),
            }
        )

        print(
            f"  [{i + 1}/{len(prompts)}] prompt_len={prompt_length} gen={generated_tokens} "
            f"ttft={ttft_ms:.2f}ms decode={decode_tokens_per_sec:.4f}tok/s"
        )

    avg_prompt_length = sum(r["prompt_length"] for r in prompt_results) / len(prompt_results)
    avg_prefill_latency_ms = sum(r["prefill_latency_ms"] for r in prompt_results) / len(prompt_results)
    avg_ttft_ms = sum(r["ttft_ms"] for r in prompt_results) / len(prompt_results)
    avg_decode_tokens_per_sec = sum(r["decode_tokens_per_sec"] for r in prompt_results) / len(prompt_results)
    measurement_notes = (
        "TTFT/prefill latency measured via a separate greedy max_tokens=1 run per prompt; "
        "peak_memory_mib is sampled from NVML or torch peak allocation when available."
    )

    result_json = {
        "framework": "vllm",
        "model_path": args.model_path,
        "dtype": args.dtype,
        "cpu_offload_gb": args.cpu_offload_gb,
        "max_new_tokens": args.max_new_tokens,
        "num_prompts": len(prompt_results),
        "avg_prompt_length": round(avg_prompt_length, 4),
        "avg_prefill_latency_ms": round(avg_prefill_latency_ms, 4),
        "avg_ttft_ms": round(avg_ttft_ms, 4),
        "avg_decode_tokens_per_sec": round(avg_decode_tokens_per_sec, 4),
        "peak_memory_mib": round(peak_memory_mib, 4),
        "measurement_notes": measurement_notes,
        "results": prompt_results,
        "summary": {
            "num_prompts": len(prompt_results),
            "avg_prompt_length": round(avg_prompt_length, 4),
            "avg_prefill_latency_ms": round(avg_prefill_latency_ms, 4),
            "avg_ttft_ms": round(avg_ttft_ms, 4),
            "avg_decode_tokens_per_sec": round(avg_decode_tokens_per_sec, 4),
            "peak_memory_mib": round(peak_memory_mib, 4),
        },
    }

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "vllm_result.json").write_text(json.dumps(result_json, indent=2), encoding="utf-8")
    _write_prompt_csv(outdir / "prompt_results.csv", prompt_results)
    _write_summary_csv(outdir / "summary.csv", result_json)
    print(f"Results written to {outdir}")

    if pynvml_module is not None:
        pynvml_module.nvmlShutdown()


if __name__ == "__main__":
    main()
