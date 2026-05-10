from __future__ import annotations

import torch

from model.kv_cache import KVCache
from weights.model_config import ModelConfig
from weights.weight_spec import LayerWeights


class Qwen3DecoderLayer:
    def __init__(self, config: ModelConfig):
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        layer_weights: LayerWeights,
        kv_cache: KVCache,
        layer_id: int,
    ) -> torch.Tensor:
        raise NotImplementedError
