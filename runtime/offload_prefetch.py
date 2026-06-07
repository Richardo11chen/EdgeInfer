from __future__ import annotations

from collections import OrderedDict
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
        self.benchmark = benchmark
        self._layer_ids = tuple(model_bundle.loader.iter_layer_ids())
        self._total_layers = len(self._layer_ids)
        self.gpu_layer_budget = self._resolve_budget(gpu_layer_budget)
        self._full_gpu_residency = self.gpu_layer_budget >= self._total_layers

        self._copy_stream = torch.cuda.Stream(device=device)
        self._cpu_cache: dict[int, LayerWeights] = {}
        self._gpu_cache: OrderedDict[int, LayerWeights] = OrderedDict()
        self._events: dict[int, torch.cuda.Event] = {}
        self._in_flight: set[int] = set()
        self._released: set[int] = set()
        self._requested: set[int] = set()

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

        for layer_id in self._layer_ids:
            lw = model_bundle.loader.load_layer_weights(layer_id)
            pinned = {
                k: t.contiguous().pin_memory()
                for k, t in lw.tensors.items()
            }
            self._cpu_cache[layer_id] = LayerWeights(layer_id=layer_id, tensors=pinned)

        if self._full_gpu_residency:
            for layer_id in self._layer_ids:
                self._ensure_gpu_layer(layer_id)
                self.synchronize_layer(layer_id)
                self._released.add(layer_id)

    def _resolve_budget(self, gpu_layer_budget: int | None) -> int:
        if gpu_layer_budget is None:
            return self._total_layers
        if gpu_layer_budget <= 0:
            raise ValueError(f"gpu_layer_budget must be >= 1, got {gpu_layer_budget}")
        return gpu_layer_budget

    def get_global_weights(self) -> GlobalWeights:
        return self._global_weights

    def _mark_recent(self, layer_id: int) -> None:
        if layer_id in self._gpu_cache:
            self._gpu_cache.move_to_end(layer_id)

    def _evict_one_layer(self) -> bool:
        for candidate in tuple(self._gpu_cache.keys()):
            if candidate in self._in_flight:
                continue
            if candidate not in self._released:
                continue
            self._events.pop(candidate, None)
            self._gpu_cache.pop(candidate, None)
            return True
        return False

    def _ensure_capacity_for_new_layer(self) -> bool:
        while len(self._gpu_cache) >= self.gpu_layer_budget:
            if not self._evict_one_layer():
                return False
        return True

    def _ensure_gpu_layer(self, layer_id: int) -> bool:
        if layer_id in self._gpu_cache:
            self._mark_recent(layer_id)
            return True

        if not self._ensure_capacity_for_new_layer():
            self._requested.add(layer_id)
            return False

        cpu_lw = self._cpu_cache[layer_id]
        if self.benchmark is not None:
            self.benchmark.on_layer_copy_start(layer_id)

        gpu_tensors = {
            k: torch.empty_like(t, dtype=self.dtype, device=self.device)
            for k, t in cpu_lw.tensors.items()
        }
        event = torch.cuda.Event()
        with torch.cuda.stream(self._copy_stream):
            for key, cpu_tensor in cpu_lw.tensors.items():
                gpu_tensors[key].copy_(cpu_tensor, non_blocking=True)
            event.record(self._copy_stream)

        self._gpu_cache[layer_id] = LayerWeights(layer_id=layer_id, tensors=gpu_tensors)
        self._events[layer_id] = event
        self._in_flight.add(layer_id)
        self._requested.discard(layer_id)
        self._released.discard(layer_id)
        self._mark_recent(layer_id)
        return True

    def prefetch_layer(self, layer_id: int) -> None:
        self._requested.add(layer_id)
        self._ensure_gpu_layer(layer_id)

    def synchronize_layer(self, layer_id: int) -> None:
        if layer_id not in self._gpu_cache:
            if not self._ensure_gpu_layer(layer_id):
                raise RuntimeError(
                    f"Unable to make GPU capacity available for layer {layer_id} with budget {self.gpu_layer_budget}"
                )

        self._mark_recent(layer_id)
        event = self._events.get(layer_id)
        if event is not None:
            torch.cuda.current_stream(self.device).wait_event(event)
            torch.cuda.synchronize(self.device)
            self._events.pop(layer_id, None)
            self._in_flight.discard(layer_id)
            if self.benchmark is not None:
                self.benchmark.on_layer_copy_end(layer_id)
        self._released.discard(layer_id)

    def get_layer_weights(self, layer_id: int) -> LayerWeights:
        self._mark_recent(layer_id)
        return self._gpu_cache[layer_id]

    def release_layer(self, layer_id: int) -> None:
        if layer_id not in self._gpu_cache:
            return None

        if self._full_gpu_residency:
            self._released.add(layer_id)
            self._mark_recent(layer_id)
            return None

        self._released.add(layer_id)
        self._mark_recent(layer_id)

        for requested_layer in tuple(self._requested):
            if requested_layer in self._gpu_cache:
                self._requested.discard(requested_layer)
                continue
            if not self._ensure_gpu_layer(requested_layer):
                break

    def close(self) -> None:
        self._gpu_cache.clear()
        self._cpu_cache.clear()
        self._events.clear()
        self._in_flight.clear()
        self._released.clear()
        self._requested.clear()
        del self._global_weights
