from __future__ import annotations

import torch

from weights.model_config import ModelConfig


class KVCache:
    def __init__(
        self,
        config: ModelConfig,
        batch_size: int,
        max_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.config = config
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.dtype = dtype

    def append(
        self,
        layer_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int,
    ) -> None:
        raise NotImplementedError

    def get(self, layer_id: int, end_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
