from __future__ import annotations

import torch
import torch.nn.functional as F

from model.kv_cache import KVCache
from model.layers import Qwen3DecoderLayer
from weights.model_config import ModelConfig
from weights.weight_spec import GlobalWeights, LayerWeights


class Qwen3Model:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.layers = [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]

    def embed(self, input_ids: torch.Tensor, global_weights: GlobalWeights) -> torch.Tensor:
        return F.embedding(input_ids, global_weights.embed_tokens)

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
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.float()
        variance = hidden_states_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed = hidden_states_fp32 * torch.rsqrt(variance + self.config.rms_norm_eps)
        normed = normed.to(dtype=input_dtype) * global_weights.final_norm.to(dtype=input_dtype)
        return F.linear(normed, global_weights.lm_head)
