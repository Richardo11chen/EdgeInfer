from __future__ import annotations

import math

import pytest
import torch

from eval.correctness import CorrectnessResult, compare_logits, compare_tokens


class TestCompareLogits:
    def test_identical(self):
        x = torch.randn(4, 8)
        r = compare_logits(x, x.clone())
        assert r.passed
        assert r.max_abs_error == pytest.approx(0.0, abs=1e-10)
        assert r.max_rel_error == pytest.approx(0.0, abs=1e-10)

    def test_small_difference_within_tolerance(self):
        x = torch.randn(4, 8)
        y = x + 1e-6 * torch.randn(4, 8)
        r = compare_logits(x, y, abs_tol=1e-4, rel_tol=1e-4)
        assert r.passed

    def test_large_difference_exceeds_tolerance(self):
        x = torch.randn(4, 8)
        y = x + 10.0
        r = compare_logits(x, y)
        assert not r.passed
        assert r.max_abs_error > 0
        assert r.max_rel_error > 0

    def test_shape_mismatch(self):
        x = torch.randn(4, 8)
        y = torch.randn(4, 9)
        r = compare_logits(x, y)
        assert not r.passed
        assert "shape mismatch" in r.message

    def test_3d_tensors(self):
        x = torch.randn(2, 3, 4)
        y = x.clone()
        r = compare_logits(x, y)
        assert r.passed

    def test_max_abs_error_exact(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.1, 3.0])
        r = compare_logits(x, y)
        assert r.max_abs_error == pytest.approx(0.1, rel=1e-5)

    def test_zero_tensor(self):
        x = torch.zeros(4, 4)
        y = torch.zeros(4, 4)
        r = compare_logits(x, y)
        assert r.passed

    def test_nan_in_actual(self):
        x = torch.tensor([float("nan")])
        y = torch.tensor([1.0])
        r = compare_logits(x, y)
        assert not r.passed
        assert not math.isfinite(r.max_abs_error)

    def test_nan_in_both(self):
        x = torch.tensor([float("nan")])
        y = torch.tensor([float("nan")])
        r = compare_logits(x, y)
        assert not r.passed
        assert not math.isfinite(r.max_abs_error)


class TestCompareTokens:
    def test_identical(self):
        x = torch.tensor([1, 2, 3, 4])
        r = compare_tokens(x, x.clone())
        assert r.passed
        assert r.mismatch_count == 0

    def test_one_mismatch(self):
        x = torch.tensor([1, 2, 3, 4])
        y = torch.tensor([1, 2, 0, 4])
        r = compare_tokens(x, y)
        assert not r.passed
        assert r.mismatch_count == 1

    def test_all_mismatch(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        r = compare_tokens(x, y)
        assert not r.passed
        assert r.mismatch_count == 3

    def test_shape_mismatch(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([1, 2])
        r = compare_tokens(x, y)
        assert not r.passed
        assert "shape mismatch" in r.message

    def test_2d_batch(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[1, 2], [3, 4]])
        r = compare_tokens(x, y)
        assert r.passed
        assert r.mismatch_count == 0


class TestCorrectnessResult:
    def test_frozen(self):
        r = CorrectnessResult(passed=True, message="ok")
        with pytest.raises(Exception):
            r.passed = False

    def test_defaults(self):
        r = CorrectnessResult(passed=False)
        assert r.max_abs_error is None
        assert r.max_rel_error is None
        assert r.mismatch_count is None
        assert r.message == ""
