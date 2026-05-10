from __future__ import annotations

import torch


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    position_ids: torch.Tensor,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError
