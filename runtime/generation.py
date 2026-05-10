from __future__ import annotations

import torch

from model.kv_cache import KVCache
from model.qwen3 import Qwen3Model
from runtime.weight_provider import WeightProvider
from weights.model_config import ModelConfig


class GenerationRuntime:
    def __init__(
        self,
        model: Qwen3Model,
        provider: WeightProvider,
        config: ModelConfig,
        benchmark: "BenchmarkHarness | None" = None,
    ):
        self.model = model
        self.provider = provider
        self.config = config
        self.benchmark = benchmark

    def prefill(self, input_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        raise NotImplementedError

    def decode_one(
        self,
        input_ids: torch.Tensor,
        position_id: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        raise NotImplementedError

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        raise NotImplementedError
