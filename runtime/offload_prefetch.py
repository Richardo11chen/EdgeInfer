from __future__ import annotations

import torch

from runtime.weight_provider import WeightProvider
from weights.loader import WeightLoader
from weights.weight_spec import GlobalWeights, LayerWeights


class PrefetchOffloadWeightProvider(WeightProvider):
    def __init__(
        self,
        loader: WeightLoader,
        device: torch.device,
        dtype: torch.dtype,
        gpu_layer_budget: int | None,
    ):
        self.loader = loader
        self.device = device
        self.dtype = dtype
        self.gpu_layer_budget = gpu_layer_budget

    def get_global_weights(self) -> GlobalWeights:
        raise NotImplementedError

    def prefetch_layer(self, layer_id: int) -> None:
        raise NotImplementedError

    def synchronize_layer(self, layer_id: int) -> None:
        raise NotImplementedError

    def get_layer_weights(self, layer_id: int) -> LayerWeights:
        raise NotImplementedError

    def release_layer(self, layer_id: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None
