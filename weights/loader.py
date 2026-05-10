from __future__ import annotations

from collections.abc import Iterable

from weights.model_config import ModelConfig
from weights.weight_spec import GlobalWeights, LayerWeights


class WeightLoader:
    def __init__(self, model_dir: str, config: ModelConfig):
        self.model_dir = model_dir
        self.config = config

    def load_global_weights(self) -> GlobalWeights:
        raise NotImplementedError("Contract only: implement safetensors loading in runtime tasks")

    def load_layer_weights(self, layer_id: int) -> LayerWeights:
        raise NotImplementedError("Contract only: implement per-layer loading in runtime tasks")

    def iter_layer_ids(self) -> Iterable[int]:
        return range(self.config.num_hidden_layers)
