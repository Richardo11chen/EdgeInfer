from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping

import torch


LAYER_WEIGHT_KEYS: tuple[str, ...] = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

GLOBAL_WEIGHT_KEYS: tuple[str, ...] = (
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
)


@dataclass(frozen=True)
class LayerWeights:
    layer_id: int
    tensors: Mapping[str, torch.Tensor]

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]

    def keys(self) -> Iterable[str]:
        return self.tensors.keys()


@dataclass(frozen=True)
class GlobalWeights:
    embed_tokens: torch.Tensor
    final_norm: torch.Tensor
    lm_head: torch.Tensor
