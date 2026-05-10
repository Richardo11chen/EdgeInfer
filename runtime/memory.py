from __future__ import annotations


def bytes_to_mib(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def query_peak_cuda_memory_bytes() -> int:
    raise NotImplementedError
