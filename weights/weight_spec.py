from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping

import torch


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
