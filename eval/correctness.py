from __future__ import annotations

from dataclasses import dataclass

import torch


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
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> CorrectnessResult:
    if actual.shape != expected.shape:
        return CorrectnessResult(
            passed=False,
            message=f"shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}",
        )

    diff = actual.float() - expected.float()
    abs_diff = diff.abs()
    max_abs = abs_diff.max().item()

    rel_diff = abs_diff / (expected.float().abs() + 1e-12)
    max_rel = rel_diff.max().item()

    passed = max_abs <= atol and max_rel <= rtol
    msg = "logits match within tolerance" if passed else "logits exceed tolerance"

    return CorrectnessResult(
        passed=passed,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        message=msg,
    )


def compare_tokens(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> CorrectnessResult:
    if actual.shape != expected.shape:
        return CorrectnessResult(
            passed=False,
            message=f"shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}",
        )

    mismatches = (actual != expected).sum().item()

    if mismatches == 0:
        return CorrectnessResult(
            passed=True,
            mismatch_count=0,
            message="all tokens match",
        )

    total = actual.numel()
    return CorrectnessResult(
        passed=False,
        mismatch_count=mismatches,
        message=f"{mismatches}/{total} tokens mismatch",
    )
