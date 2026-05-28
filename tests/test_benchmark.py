from __future__ import annotations

import csv
import time

import torch

from eval.benchmark import BenchmarkHarness

CUDA_AVAILABLE = torch.cuda.is_available()


class TestPrefillTiming:
    def test_simple(self):
        bm = BenchmarkHarness()
        bm.on_prefill_start()
        bm.on_prefill_end()
        s = bm.get_summary()
        assert s["prefill_latency_ms"] >= 0
        assert s["ttft_ms"] == s["prefill_latency_ms"]

    def test_no_prefill_returns_zero(self):
        bm = BenchmarkHarness()
        s = bm.get_summary()
        assert s["prefill_latency_ms"] == 0.0
        assert s["ttft_ms"] == 0.0


class TestDecodeTiming:
    def test_simple(self):
        bm = BenchmarkHarness()
        bm.on_decode_start()
        bm.on_decode_token_end(0)
        time.sleep(0.01)
        bm.on_decode_token_end(1)
        s = bm.get_summary()
        assert s["decode_tokens_per_sec"] > 0

    def test_no_decode_returns_zero(self):
        bm = BenchmarkHarness()
        s = bm.get_summary()
        assert s["decode_tokens_per_sec"] == 0.0

    def test_single_token(self):
        bm = BenchmarkHarness()
        bm.on_decode_start()
        bm.on_decode_token_end(0)
        s = bm.get_summary()
        assert s["decode_tokens_per_sec"] >= 0


class TestLayerCopyTiming:
    def test_single_layer(self):
        bm = BenchmarkHarness()
        bm.on_layer_copy_start(0)
        bm.on_layer_copy_end(0)
        s = bm.get_summary()
        assert s["total_copy_ms"] >= 0
        assert s["avg_copy_ms"] >= 0

    def test_multiple_layers(self):
        bm = BenchmarkHarness()
        for i in range(4):
            bm.on_layer_copy_start(i)
        for i in range(4):
            bm.on_layer_copy_end(i)
        s = bm.get_summary()
        assert s["total_copy_ms"] >= 0
        assert s["avg_copy_ms"] >= 0

    def test_no_copy_returns_zero(self):
        bm = BenchmarkHarness()
        s = bm.get_summary()
        assert s["total_copy_ms"] == 0.0
        assert s["avg_copy_ms"] == 0.0

    def test_repeated_layer_accumulates(self):
        bm = BenchmarkHarness()
        bm.on_layer_copy_start(0)
        bm.on_layer_copy_end(0)
        bm.on_layer_copy_start(0)
        bm.on_layer_copy_end(0)
        s = bm.get_summary()
        assert s["total_copy_ms"] >= 0

    def test_end_without_start_is_noop(self):
        bm = BenchmarkHarness()
        bm.on_layer_copy_end(0)


class TestLayerComputeTiming:
    def test_single_layer(self):
        bm = BenchmarkHarness()
        bm.on_layer_compute_start(0)
        bm.on_layer_compute_end(0)
        s = bm.get_summary()
        assert s["total_compute_ms"] >= 0
        assert s["avg_compute_ms"] >= 0

    def test_multiple_layers(self):
        bm = BenchmarkHarness()
        for i in range(4):
            bm.on_layer_compute_start(i)
        for i in range(4):
            bm.on_layer_compute_end(i)
        s = bm.get_summary()
        assert s["total_compute_ms"] >= 0
        assert s["avg_compute_ms"] >= 0

    def test_no_compute_returns_zero(self):
        bm = BenchmarkHarness()
        s = bm.get_summary()
        assert s["total_compute_ms"] == 0.0
        assert s["avg_compute_ms"] == 0.0

    def test_end_without_start_is_noop(self):
        bm = BenchmarkHarness()
        bm.on_layer_compute_end(0)


class TestMemoryMetrics:
    def test_peak_memory_key_present(self):
        bm = BenchmarkHarness()
        bm.on_prefill_start()
        bm.on_prefill_end()
        s = bm.get_summary()
        assert "peak_memory_mib" in s
        assert isinstance(s["peak_memory_mib"], float)
        assert s["peak_memory_mib"] >= 0.0

    def test_no_cuda_reset_does_not_crash(self):
        bm = BenchmarkHarness()
        s = bm.get_summary()
        assert s["peak_memory_mib"] == 0.0 if not CUDA_AVAILABLE else True


class TestExportCsv:
    def test_writes_file(self, tmp_path):
        bm = BenchmarkHarness()
        output = tmp_path / "bench.csv"
        bm.export_csv(str(output))
        assert output.exists()

    def test_header_and_structure(self, tmp_path):
        bm = BenchmarkHarness()
        bm.on_prefill_start()
        bm.on_prefill_end()
        bm.on_decode_start()
        bm.on_decode_token_end(0)
        bm.on_layer_copy_start(0)
        bm.on_layer_copy_end(0)
        bm.on_layer_compute_start(0)
        bm.on_layer_compute_end(0)
        output = tmp_path / "bench.csv"
        bm.export_csv(str(output))

        with open(output, newline="") as f:
            reader = list(csv.reader(f))

        assert reader[0] == ["metric", "value"]
        assert len(reader) > 3

    def test_per_layer_detail_section(self, tmp_path):
        bm = BenchmarkHarness()
        for i in range(3):
            bm.on_layer_copy_start(i)
            bm.on_layer_copy_end(i)
            bm.on_layer_compute_start(i)
            bm.on_layer_compute_end(i)
        output = tmp_path / "bench.csv"
        bm.export_csv(str(output))

        with open(output, newline="") as f:
            reader = list(csv.reader(f))

        blank_idx = None
        for i, row in enumerate(reader):
            if len(row) == 0:
                blank_idx = i
                break
        assert blank_idx is not None
        assert reader[blank_idx + 1] == ["layer_id", "copy_time_ms", "compute_time_ms"]
        assert len(reader) > blank_idx + 4  # 3 layers worth


class TestFullWorkflow:
    def test_prefill_decode_workflow(self):
        bm = BenchmarkHarness()
        bm.on_prefill_start()
        for i in range(4):
            bm.on_layer_copy_start(i)
            bm.on_layer_copy_end(i)
            bm.on_layer_compute_start(i)
            bm.on_layer_compute_end(i)
        bm.on_prefill_end()

        bm.on_decode_start()
        for t in range(5):
            for i in range(4):
                bm.on_layer_copy_start(i)
                bm.on_layer_copy_end(i)
                bm.on_layer_compute_start(i)
                bm.on_layer_compute_end(i)
            bm.on_decode_token_end(t)

        s = bm.get_summary()
        assert s["prefill_latency_ms"] > 0
        assert s["ttft_ms"] > 0
        assert s["decode_tokens_per_sec"] > 0
        assert s["total_copy_ms"] > 0
        assert s["total_compute_ms"] > 0


class TestSummaryKeys:
    def test_all_expected_keys_present(self):
        bm = BenchmarkHarness()
        s = bm.get_summary()
        expected = {
            "prefill_latency_ms",
            "ttft_ms",
            "decode_tokens_per_sec",
            "total_copy_ms",
            "avg_copy_ms",
            "total_compute_ms",
            "avg_compute_ms",
            "peak_memory_mib",
        }
        assert set(s.keys()) == expected
        for v in s.values():
            assert isinstance(v, float)
