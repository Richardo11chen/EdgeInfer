from __future__ import annotations

import torch

from weights.model_config import ModelConfig


class RotaryEmbedding:
    def __init__(self, config: ModelConfig, device: torch.device):
        self.config = config
        self.device = device

    def get_cos_sin(
        self,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
