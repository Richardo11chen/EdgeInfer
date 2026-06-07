from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.kv_cache import KVCache
from model.qwen3 import Qwen3Model
from runtime.generation import GenerationRuntime
from runtime.weight_provider import create_weight_provider
from weights.model_bundle import open_model_bundle


@dataclass(frozen=True)
class CorrectnessResult:
    passed: bool
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    mismatch_count: int | None = None
    message: str = ""


def compare_logits(
    actual: torch.Tensor,
    expected: torch.Tensor,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-5,
) -> CorrectnessResult:
    if actual.shape != expected.shape:
        return CorrectnessResult(False, message=f"shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")

    abs_diff = (actual - expected).abs()
    max_abs_error = float(abs_diff.max().item())

    denom = expected.abs().clamp_min(1e-12)
    rel_diff = abs_diff / denom
    max_rel_error = float(rel_diff.max().item())

    return CorrectnessResult(
        passed=(max_abs_error <= abs_tol) and (max_rel_error <= rel_tol),
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        message="ok",
    )


def compare_tokens(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> CorrectnessResult:
    if actual.shape != expected.shape:
        return CorrectnessResult(False, message=f"shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")

    mismatches = int((actual != expected).sum().item())
    return CorrectnessResult(
        passed=mismatches == 0,
        mismatch_count=mismatches,
        message="ok" if mismatches == 0 else "token mismatch",
    )


def _parse_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def _build_runtime(model_dir: str, device: torch.device, dtype: torch.dtype) -> GenerationRuntime:
    bundle = open_model_bundle(model_dir)
    model = Qwen3Model(bundle.config)
    provider = create_weight_provider(
        model_bundle=bundle,
        mode="resident",
        device=device,
        dtype=dtype,
        gpu_layer_budget=None,
    )
    return GenerationRuntime(model=model, provider=provider, config=bundle.config)


def _hf_fixed_greedy_generate(
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


def run_alignment(
    model_dir: str,
    prompt: str,
    max_new_tokens: int,
    device: str,
    dtype: str,
    abs_tol: float,
    rel_tol: float,
) -> None:
    torch_device = torch.device(device)
    torch_dtype = _parse_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(torch_device)
    hf_model.eval()

    runtime = _build_runtime(model_dir, torch_device, torch_dtype)

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(torch_device)
    kv_cache = KVCache(
        config=runtime.config,
        batch_size=input_ids.shape[0],
        max_sequence_length=input_ids.shape[1] + max_new_tokens,
        device=torch_device,
        dtype=torch_dtype,
    )

    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids).logits
        rt_logits = runtime.prefill(input_ids, kv_cache)
        hf_tokens = _hf_fixed_greedy_generate(hf_model, input_ids, max_new_tokens)
        rt_tokens = runtime.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)

    logits_result = compare_logits(rt_logits, hf_logits, abs_tol=abs_tol, rel_tol=rel_tol)
    tokens_result = compare_tokens(rt_tokens, hf_tokens)

    print(
        "[logits] "
        f"passed={logits_result.passed} "
        f"max_abs_error={logits_result.max_abs_error} "
        f"max_rel_error={logits_result.max_rel_error} "
        f"abs_tol={abs_tol} rel_tol={rel_tol}"
    )
    print(f"[tokens] passed={tokens_result.passed} mismatch_count={tokens_result.mismatch_count}")
    if not tokens_result.passed:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-1.7B runtime correctness check")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--abs-tol", type=float, default=1e-2)
    parser.add_argument("--rel-tol", type=float, default=1e-2)
    args = parser.parse_args()

    run_alignment(
        model_dir=args.model_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
    )


if __name__ == "__main__":
    main()
