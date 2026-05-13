from __future__ import annotations

import torch


class MemoryTracker:
    def reset_peak(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    def get_peak_memory_bytes(self) -> int:
        return torch.cuda.max_memory_allocated()

    def get_allocated_bytes(self) -> int:
        return torch.cuda.memory_allocated()


def bytes_to_mib(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)
