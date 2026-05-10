from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ExperimentConfig:
    model_dir: str
    device: str
    dtype: str
    offload_mode: Literal["resident", "naive", "prefetch"]
    gpu_layer_budget: int | None
    batch_size: int
    prompt_length: int
    generate_length: int
    max_sequence_length: int
    output_dir: str
    seed: int

    def __post_init__(self) -> None:
        if self.offload_mode not in {"resident", "naive", "prefetch"}:
            raise ValueError(f"Unsupported offload_mode: {self.offload_mode}")
