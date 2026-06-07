from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from weights.model_bundle import ModelBundle
from weights.weight_spec import GlobalWeights, LayerWeights

if TYPE_CHECKING:
    from eval.benchmark import BenchmarkHarness


class NaiveOffloadWeightProvider:
    def __init__(
        self,
        model_bundle: ModelBundle,
        device: torch.device,
        dtype: torch.dtype,
        benchmark: BenchmarkHarness | None = None,
    ):
        self.model_bundle = model_bundle
        self.device = device
        self.dtype = dtype
        self.benchmark = benchmark

        cpu_global = model_bundle.loader.load_global_weights()
        embed_tokens = cpu_global.embed_tokens.to(device=device, dtype=dtype)
        if cpu_global.lm_head.data_ptr() == cpu_global.embed_tokens.data_ptr():
            lm_head = embed_tokens
        else:
            lm_head = cpu_global.lm_head.to(device=device, dtype=dtype)
        self._global_weights = GlobalWeights(
            embed_tokens=embed_tokens,
            final_norm=cpu_global.final_norm.to(device=device, dtype=dtype),
            lm_head=lm_head,
        )

        self._cpu_layer_cache: dict[int, LayerWeights] = {}
        self._gpu_layer_cache: dict[int, LayerWeights] = {}
        pin_cpu_layers = self.device.type == "cuda"

        for layer_id in model_bundle.loader.iter_layer_ids():
            lw = model_bundle.loader.load_layer_weights(layer_id)
            cpu_tensors = {
                k: self._prepare_cpu_tensor(t, pin_memory=pin_cpu_layers)
                for k, t in lw.tensors.items()
            }
            self._cpu_layer_cache[layer_id] = LayerWeights(
                layer_id=layer_id,
                tensors=cpu_tensors,
            )

    def _prepare_cpu_tensor(
        self,
        tensor: torch.Tensor,
        *,
        pin_memory: bool,
    ) -> torch.Tensor:
        cpu_tensor = tensor.contiguous()
        if pin_memory:
            cpu_tensor = cpu_tensor.pin_memory()
        return cpu_tensor

    def get_global_weights(self) -> GlobalWeights:
        return self._global_weights

    def prefetch_layer(self, layer_id: int) -> None:
        return None

    def synchronize_layer(self, layer_id: int) -> None:
        if layer_id in self._gpu_layer_cache:
            return None

        if self.benchmark is not None:
            self.benchmark.on_layer_copy_start(layer_id)

        lw = self._cpu_layer_cache[layer_id]

        gpu_tensors = {
            k: t.to(device=self.device, dtype=self.dtype, non_blocking=False)
            for k, t in lw.tensors.items()
        }
        self._gpu_layer_cache[layer_id] = LayerWeights(
            layer_id=layer_id, tensors=gpu_tensors
        )

        if self.benchmark is not None:
            self.benchmark.on_layer_copy_end(layer_id)

    def get_layer_weights(self, layer_id: int) -> LayerWeights:
        return self._gpu_layer_cache[layer_id]

    def release_layer(self, layer_id: int) -> None:
        self._gpu_layer_cache.pop(layer_id, None)

    def close(self) -> None:
        self._gpu_layer_cache.clear()
        self._cpu_layer_cache.clear()
        del self._global_weights
