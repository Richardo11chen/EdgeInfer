from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors import safe_open

from weights.model_config import ModelConfig
from weights.name_mapping import WeightNameMapper
from weights.weight_spec import LAYER_WEIGHT_KEYS, GLOBAL_WEIGHT_KEYS, GlobalWeights, LayerWeights


class WeightLoader:
    def __init__(self, model_dir: str, config: ModelConfig):
        self.model_dir = model_dir
        self.config = config
        self._name_mapper = WeightNameMapper()

        safetensors_dir = Path(model_dir)
        safetensor_files = sorted(safetensors_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

        self._name_to_file: dict[str, str] = {}
        for safetensor_path in safetensor_files:
            with safe_open(str(safetensor_path), framework="pt") as f:
                for tensor_name in f.keys():
                    self._name_to_file[tensor_name] = str(safetensor_path)

    def _load_tensor(self, hf_name: str) -> torch.Tensor:
        file_path = self._name_to_file.get(hf_name)
        if file_path is None:
            raise KeyError(f"Tensor '{hf_name}' not found in safetensors index under {self.model_dir}")
        with safe_open(file_path, framework="pt") as f:
            return f.get_tensor(hf_name).contiguous()

    def load_global_weights(self) -> GlobalWeights:
        tensors: dict[str, torch.Tensor] = {}
        for hf_name in GLOBAL_WEIGHT_KEYS:
            internal_name = self._name_mapper.hf_global_name_to_internal(hf_name)
            tensors[internal_name] = self._load_tensor(hf_name)

        return GlobalWeights(
            embed_tokens=tensors["model.embed_tokens.weight"],
            final_norm=tensors["model.norm.weight"],
            lm_head=tensors["lm_head.weight"],
        )

    def load_layer_weights(self, layer_id: int) -> LayerWeights:
        tensors: dict[str, torch.Tensor] = {}
        for internal_key in LAYER_WEIGHT_KEYS:
            hf_name = self._name_mapper.internal_layer_name_to_hf(internal_key, layer_id)
            tensor = self._load_tensor(hf_name)
            tensors[internal_key] = tensor

        return LayerWeights(layer_id=layer_id, tensors=tensors)

    def iter_layer_ids(self) -> Iterable[int]:
        return range(self.config.num_hidden_layers)
