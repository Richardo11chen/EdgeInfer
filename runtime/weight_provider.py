import torch
from typing import Dict

class WeightProvider:
    def __init__(self, device: torch.device):
        self.device = device

    def get_layer_weights(self, layer_id: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def release_layer(self, layer_id: int) -> None:
        pass

class MockWeightProvider(WeightProvider):
    """
    开发阶段造假数据，让模型主循环先跑通
    """
    def get_layer_weights(self, layer_id: int) -> Dict[str, torch.Tensor]:
        # 提供一个假的 Qwen3-1.7B MLP 权重
        return {
            "mlp.down_proj.weight": torch.randn((1536, 4096), dtype=torch.float16, device=self.device)
        }