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
        global_weights = self.provider.get_global_weights()
        hidden_states = self.model.embed(input_ids, global_weights)

        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        for layer_id in range(self.config.num_hidden_layers):
            self.provider.prefetch_layer(layer_id)
            self.provider.synchronize_layer(layer_id)
            layer_weights = self.provider.get_layer_weights(layer_id)
            hidden_states = self.model.forward_layer(
                layer_id=layer_id,
                hidden_states=hidden_states,
                position_ids=position_ids,
                layer_weights=layer_weights,
                kv_cache=kv_cache,
            )
            self.provider.release_layer(layer_id)

        return self.model.final_logits(hidden_states, global_weights)

    def decode_one(
        self,
        input_ids: torch.Tensor,
        position_id: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        global_weights = self.provider.get_global_weights()
        hidden_states = self.model.embed(input_ids, global_weights)

        for layer_id in range(self.config.num_hidden_layers):
            self.provider.prefetch_layer(layer_id)
            self.provider.synchronize_layer(layer_id)
            layer_weights = self.provider.get_layer_weights(layer_id)
            hidden_states = self.model.forward_layer(
                layer_id=layer_id,
                hidden_states=hidden_states,
                position_ids=position_id,
                layer_weights=layer_weights,
                kv_cache=kv_cache,
            )
            self.provider.release_layer(layer_id)

        return self.model.final_logits(hidden_states, global_weights)

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if max_new_tokens == 0:
            return input_ids

        prompt_len = input_ids.shape[1]
        kv_cache = KVCache(
            config=self.config,
            batch_size=input_ids.shape[0],
            max_sequence_length=prompt_len + max_new_tokens,
            device=input_ids.device,
            dtype=self.provider.get_global_weights().embed_tokens.dtype,
        )

        generated = input_ids
        logits = self.prefill(generated, kv_cache)

        for step in range(max_new_tokens):
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if step == max_new_tokens - 1:
                break

            next_position_id = torch.full(
                (generated.shape[0], 1),
                prompt_len + step,
                device=generated.device,
                dtype=torch.long,
            )
            logits = self.decode_one(next_token, next_position_id, kv_cache)

        return generated
