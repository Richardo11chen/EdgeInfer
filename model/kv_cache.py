from __future__ import annotations

import torch

from weights.model_config import ModelConfig


class KVCache:
    def __init__(
        self,
        config: ModelConfig,
        batch_size: int,
        max_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.config = config
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.dtype = dtype

        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if max_sequence_length <= 0:
            raise ValueError(f"max_sequence_length must be > 0, got {max_sequence_length}")

        shape = (
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_sequence_length,
            config.head_dim,
        )
        self.key_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.value_cache = torch.zeros(shape, device=device, dtype=dtype)

    def append(
        self,
        layer_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int,
    ) -> None:
        self._validate_layer_id(layer_id)
        if start_pos < 0:
            raise ValueError(f"start_pos must be >= 0, got {start_pos}")

        expected = (
            self.batch_size,
            self.config.num_key_value_heads,
            key.shape[2],
            self.config.head_dim,
        )
        if key.ndim != 4:
            raise ValueError(f"key must have shape [batch, kv_heads, seq_len, head_dim], got {tuple(key.shape)}")
        if value.ndim != 4:
            raise ValueError(
                f"value must have shape [batch, kv_heads, seq_len, head_dim], got {tuple(value.shape)}"
            )
        if key.shape != value.shape:
            raise ValueError(f"key/value shape mismatch: {tuple(key.shape)} vs {tuple(value.shape)}")
        if key.shape[0] != expected[0] or key.shape[1] != expected[1] or key.shape[3] != expected[3]:
            raise ValueError(
                "key/value must have shape "
                f"[{self.batch_size}, {self.config.num_key_value_heads}, seq_len, {self.config.head_dim}], "
                f"got {tuple(key.shape)}"
            )

        seq_len = key.shape[2]
        end_pos = start_pos + seq_len
        if end_pos > self.max_sequence_length:
            raise ValueError(
                f"append range [{start_pos}, {end_pos}) exceeds max_sequence_length={self.max_sequence_length}"
            )

        self.key_cache[layer_id, :, :, start_pos:end_pos, :] = key
        self.value_cache[layer_id, :, :, start_pos:end_pos, :] = value

    def get(self, layer_id: int, end_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_layer_id(layer_id)
        if end_pos < 0 or end_pos > self.max_sequence_length:
            raise ValueError(
                f"end_pos must be in [0, {self.max_sequence_length}], got {end_pos}"
            )

        key = self.key_cache[layer_id, :, :, :end_pos, :]
        value = self.value_cache[layer_id, :, :, :end_pos, :]
        return key, value

    def reset(self) -> None:
        self.key_cache.zero_()
        self.value_cache.zero_()

    def _validate_layer_id(self, layer_id: int) -> None:
        if layer_id < 0 or layer_id >= self.config.num_hidden_layers:
            raise ValueError(
                f"layer_id must be in [0, {self.config.num_hidden_layers - 1}], got {layer_id}"
            )
