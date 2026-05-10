from __future__ import annotations

import torch

from model.kv_cache import KVCache
from model.layers import Qwen3DecoderLayer
from weights.model_config import ModelConfig
from weights.weight_spec import GlobalWeights, LayerWeights


class Qwen3Model:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.layers = [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]

    def embed(self, input_ids: torch.Tensor, global_weights: GlobalWeights) -> torch.Tensor:
        raise NotImplementedError

    def forward_layer(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        layer_weights: LayerWeights,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        return self.layers[layer_id].forward(
            hidden_states=hidden_states,
            position_ids=position_ids,
            layer_weights=layer_weights,
            kv_cache=kv_cache,
            layer_id=layer_id,
        )

    def final_logits(self, hidden_states: torch.Tensor, global_weights: GlobalWeights) -> torch.Tensor:
        raise NotImplementedError
