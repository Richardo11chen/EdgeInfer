from __future__ import annotations

import torch

from weights.model_config import ModelConfig


class RotaryEmbedding:
    def __init__(self, config: ModelConfig, device: torch.device):
        self.config = config
        self.device = device

    def get_cos_sin(
        self,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2:
            raise ValueError(
                f"position_ids must have shape [batch_size, seq_len], got {tuple(position_ids.shape)}"
            )

        head_dim = self.config.head_dim
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

        # Compute in float32 for numerical stability, then cast back.
        pos = position_ids.to(dtype=torch.float32)
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float32) / head_dim)
        )

        freqs = pos.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = torch.cos(emb).unsqueeze(1)
        sin = torch.sin(emb).unsqueeze(1)

        return cos, sin
