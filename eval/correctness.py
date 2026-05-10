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
) -> CorrectnessResult:
    raise NotImplementedError


def compare_tokens(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> CorrectnessResult:
    raise NotImplementedError
