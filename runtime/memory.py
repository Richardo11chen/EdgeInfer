from __future__ import annotations


class MemoryTracker:
    def reset_peak(self) -> None:
        raise NotImplementedError

    def get_peak_memory_bytes(self) -> int:
        raise NotImplementedError

    def get_allocated_bytes(self) -> int:
        raise NotImplementedError


def bytes_to_mib(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)
