from __future__ import annotations

from typing import Protocol

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
