from __future__ import annotations

from typing import Literal, Protocol

import torch

from runtime.offload_naive import NaiveOffloadWeightProvider
from runtime.offload_prefetch import PrefetchOffloadWeightProvider
from weights.model_bundle import ModelBundle
from weights.weight_spec import GlobalWeights, LayerWeights


class WeightProvider(Protocol):
    def get_global_weights(self) -> GlobalWeights:
        ...

    def prefetch_layer(self, layer_id: int) -> None:
        ...

    def synchronize_layer(self, layer_id: int) -> None:
        ...

    def get_layer_weights(self, layer_id: int) -> LayerWeights:
        ...

    def release_layer(self, layer_id: int) -> None:
        ...

    def close(self) -> None:
        ...


class ResidentWeightProvider:
    def __init__(
        self,
        model_bundle: ModelBundle,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.device = device
        self.dtype = dtype

        self._global_weights = model_bundle.loader.load_global_weights()
        self._global_weights = GlobalWeights(
            embed_tokens=self._global_weights.embed_tokens.to(device=device, dtype=dtype),
            final_norm=self._global_weights.final_norm.to(device=device, dtype=dtype),
            lm_head=self._global_weights.lm_head.to(device=device, dtype=dtype),
        )

        self._layer_weights: dict[int, LayerWeights] = {}
        for layer_id in model_bundle.loader.iter_layer_ids():
            lw = model_bundle.loader.load_layer_weights(layer_id)
            gpu_tensors = {
                k: t.to(device=device, dtype=dtype) for k, t in lw.tensors.items()
            }
            self._layer_weights[layer_id] = LayerWeights(
                layer_id=layer_id, tensors=gpu_tensors
            )

    def get_global_weights(self) -> GlobalWeights:
        return self._global_weights

    def prefetch_layer(self, layer_id: int) -> None:
        return None

    def synchronize_layer(self, layer_id: int) -> None:
        return None

    def get_layer_weights(self, layer_id: int) -> LayerWeights:
        return self._layer_weights[layer_id]

    def release_layer(self, layer_id: int) -> None:
        return None

    def close(self) -> None:
        del self._layer_weights
        del self._global_weights


def create_weight_provider(
    model_bundle: ModelBundle,
    mode: Literal["resident", "naive", "prefetch"],
    device: torch.device,
    dtype: torch.dtype,
    gpu_layer_budget: int | None,
) -> WeightProvider:
    if mode == "resident":
        return ResidentWeightProvider(
            model_bundle=model_bundle,
            device=device,
            dtype=dtype,
        )
    elif mode == "naive":
        return NaiveOffloadWeightProvider(
            model_bundle=model_bundle,
            device=device,
            dtype=dtype,
        )
    elif mode == "prefetch":
        return PrefetchOffloadWeightProvider(
            model_bundle=model_bundle,
            device=device,
            dtype=dtype,
            gpu_layer_budget=gpu_layer_budget,
        )
    else:
        raise ValueError(f"Unsupported offload mode: {mode}")
