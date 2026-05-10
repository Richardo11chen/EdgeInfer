from __future__ import annotations

import torch

from runtime.weight_provider import WeightProvider
from weights.model_bundle import ModelBundle
from weights.weight_spec import GlobalWeights, LayerWeights


class NaiveOffloadWeightProvider(WeightProvider):
    def __init__(
        self,
        model_bundle: ModelBundle,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.model_bundle = model_bundle
        self.device = device
        self.dtype = dtype

    def get_global_weights(self) -> GlobalWeights:
        raise NotImplementedError

    def prefetch_layer(self, layer_id: int) -> None:
        return None

    def synchronize_layer(self, layer_id: int) -> None:
        raise NotImplementedError

    def get_layer_weights(self, layer_id: int) -> LayerWeights:
        raise NotImplementedError

    def release_layer(self, layer_id: int) -> None:
        return None

    def close(self) -> None:
        return None
