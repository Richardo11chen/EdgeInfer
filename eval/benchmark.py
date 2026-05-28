from __future__ import annotations

import csv
import time

import torch

from runtime.memory import MemoryTracker, bytes_to_mib


class BenchmarkHarness:
    def __init__(self) -> None:
        self._prefill_start: float | None = None
        self._prefill_end: float | None = None
        self._decode_start: float | None = None
        self._decode_token_times: list[float] = []
        self._layer_copy_start: dict[int, float] = {}
        self._layer_copy_totals: dict[int, float] = {}
        self._layer_compute_start: dict[int, float] = {}
        self._layer_compute_totals: dict[int, float] = {}
        self._memory_tracker = MemoryTracker()

    def on_prefill_start(self) -> None:
        if torch.cuda.is_available():
            self._memory_tracker.reset_peak()
        self._prefill_start = time.perf_counter()

    def on_prefill_end(self) -> None:
        self._prefill_end = time.perf_counter()

    def on_decode_start(self) -> None:
        self._decode_start = time.perf_counter()

    def on_decode_token_end(self, token_index: int) -> None:
        self._decode_token_times.append(time.perf_counter())

    def on_layer_copy_start(self, layer_id: int) -> None:
        self._layer_copy_start[layer_id] = time.perf_counter()

    def on_layer_copy_end(self, layer_id: int) -> None:
        start = self._layer_copy_start.pop(layer_id, None)
        if start is not None:
            elapsed = (time.perf_counter() - start) * 1000.0
            self._layer_copy_totals[layer_id] = self._layer_copy_totals.get(layer_id, 0.0) + elapsed

    def on_layer_compute_start(self, layer_id: int) -> None:
        self._layer_compute_start[layer_id] = time.perf_counter()

    def on_layer_compute_end(self, layer_id: int) -> None:
        start = self._layer_compute_start.pop(layer_id, None)
        if start is not None:
            elapsed = (time.perf_counter() - start) * 1000.0
            self._layer_compute_totals[layer_id] = self._layer_compute_totals.get(layer_id, 0.0) + elapsed

    def get_summary(self) -> dict[str, float]:
        result: dict[str, float] = {}

        if self._prefill_start is not None and self._prefill_end is not None:
            result["prefill_latency_ms"] = (self._prefill_end - self._prefill_start) * 1000.0
            result["ttft_ms"] = result["prefill_latency_ms"]
        else:
            result["prefill_latency_ms"] = 0.0
            result["ttft_ms"] = 0.0

        if self._decode_token_times and self._decode_start is not None:
            decode_elapsed = self._decode_token_times[-1] - self._decode_start
            result["decode_tokens_per_sec"] = len(self._decode_token_times) / decode_elapsed if decode_elapsed > 0 else 0.0
        else:
            result["decode_tokens_per_sec"] = 0.0

        if self._layer_copy_totals:
            total_copy = sum(self._layer_copy_totals.values())
            result["total_copy_ms"] = total_copy
            result["avg_copy_ms"] = total_copy / len(self._layer_copy_totals)
        else:
            result["total_copy_ms"] = 0.0
            result["avg_copy_ms"] = 0.0

        if self._layer_compute_totals:
            total_compute = sum(self._layer_compute_totals.values())
            result["total_compute_ms"] = total_compute
            result["avg_compute_ms"] = total_compute / len(self._layer_compute_totals)
        else:
            result["total_compute_ms"] = 0.0
            result["avg_compute_ms"] = 0.0

        if torch.cuda.is_available():
            result["peak_memory_mib"] = bytes_to_mib(self._memory_tracker.get_peak_memory_bytes())
        else:
            result["peak_memory_mib"] = 0.0

        return result

    def export_csv(self, output_path: str) -> None:
        summary = self.get_summary()
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, value in summary.items():
                writer.writerow([key, f"{value:.4f}"])

            writer.writerow([])
            writer.writerow(["layer_id", "copy_time_ms", "compute_time_ms"])
            all_layers = sorted(
                set(self._layer_copy_totals.keys()) | set(self._layer_compute_totals.keys())
            )
            for layer_id in all_layers:
                writer.writerow([
                    layer_id,
                    f"{self._layer_copy_totals.get(layer_id, 0.0):.4f}",
                    f"{self._layer_compute_totals.get(layer_id, 0.0):.4f}",
                ])
