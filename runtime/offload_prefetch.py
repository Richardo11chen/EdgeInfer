from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from weights.model_bundle import ModelBundle
from weights.weight_spec import GlobalWeights, LayerWeights

if TYPE_CHECKING:
    from eval.benchmark import BenchmarkHarness


class PrefetchOffloadWeightProvider:
    def __init__(
        self,
        model_bundle: ModelBundle,
        device: torch.device,
        dtype: torch.dtype,
        gpu_layer_budget: int | None,
        benchmark: BenchmarkHarness | None = None,
    ):
        self.model_bundle = model_bundle
        self.device = device
        self.dtype = dtype
        self.gpu_layer_budget = gpu_layer_budget
        self.benchmark = benchmark

        self._copy_stream = torch.cuda.Stream(device=device)

        self._cpu_cache: dict[int, LayerWeights] = {}
        self._gpu_cache: dict[int, LayerWeights] = {}
        self._events: dict[int, torch.cuda.Event] = {}
        self._gpu_access_order: list[int] = []

        cpu_global = model_bundle.loader.load_global_weights()
        self._global_weights = GlobalWeights(
            embed_tokens=cpu_global.embed_tokens.to(device=device, dtype=dtype),
            final_norm=cpu_global.final_norm.to(device=device, dtype=dtype),
            lm_head=cpu_global.lm_head.to(device=device, dtype=dtype),
        )

    def get_global_weights(self) -> GlobalWeights:
        return self._global_weights

    def _evict_lru(self) -> None:
        """Evict the least recently used layer from GPU to stay within budget."""
        if not self._gpu_access_order:
            return
        oldest = self._gpu_access_order.pop(0)
        event = self._events.pop(oldest, None)
        if event is not None:
            event.synchronize()
        self._gpu_cache.pop(oldest, None)

    def prefetch_layer(self, layer_id: int) -> None:
        if layer_id not in self._cpu_cache:
            lw = self.model_bundle.loader.load_layer_weights(layer_id)
            pinned = {
                k: t.contiguous().pin_memory() for k, t in lw.tensors.items()
            }
            self._cpu_cache[layer_id] = LayerWeights(
                layer_id=layer_id, tensors=pinned
            )

        cpu_lw = self._cpu_cache[layer_id]

        # Enforce GPU layer budget: evict LRU before adding a new layer.
        if self.gpu_layer_budget is not None and layer_id not in self._gpu_cache:
            while len(self._gpu_cache) >= self.gpu_layer_budget:
                self._evict_lru()

        if layer_id in self._gpu_access_order:
            self._gpu_access_order.remove(layer_id)
        self._gpu_access_order.append(layer_id)

        if self.benchmark is not None:
            self.benchmark.on_layer_copy_start(layer_id)

        gpu_tensors = {
            k: torch.empty(t.shape, dtype=self.dtype, device=self.device)
            for k, t in cpu_lw.tensors.items()
        }

        event = torch.cuda.Event()
        with torch.cuda.stream(self._copy_stream):
            for k in cpu_lw.tensors:
                gpu_tensors[k].copy_(cpu_lw.tensors[k], non_blocking=True)
            event.record(self._copy_stream)

        self._gpu_cache[layer_id] = LayerWeights(
            layer_id=layer_id, tensors=gpu_tensors
        )
        self._events[layer_id] = event

    def synchronize_layer(self, layer_id: int) -> None:
        event = self._events.get(layer_id)
        if event is not None:
            event.synchronize()
        if self.benchmark is not None:
            self.benchmark.on_layer_copy_end(layer_id)

    def get_layer_weights(self, layer_id: int) -> LayerWeights:
        return self._gpu_cache[layer_id]

    def release_layer(self, layer_id: int) -> None:
        self._gpu_cache.pop(layer_id, None)
        self._events.pop(layer_id, None)
        if layer_id in self._gpu_access_order:
            self._gpu_access_order.remove(layer_id)
        if self._cpu_cache.get(layer_id) is not None:
            del self._cpu_cache[layer_id]

    def close(self) -> None:
        self._gpu_cache.clear()
        self._cpu_cache.clear()
        self._events.clear()
        self._gpu_access_order.clear()
        del self._global_weights
