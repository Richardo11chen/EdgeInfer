from dataclasses import dataclass

@dataclass
class Qwen3Config:
    # 预设参数，具体值后续可通过读取 config.json 覆盖
    vocab_size: int = 151936
    hidden_size: int = 3584       # 8B 尺寸
    num_hidden_layers: int = 28   # 8B 层数
    num_attention_heads: int = 28
    num_key_value_heads: int = 4

@dataclass
class ExperimentConfig:
    """由成员 B 控制的实验评测配置"""
    model_path: str = "/home/user/ondevice-models/Qwen3-8B"
    batch_size: int = 1
    prompt_length: int = 512
    generate_length: int = 128
    
    # 权重调度模式: "resident" (常驻), "naive" (朴素offload), "prefetch" (流水线预取)
    offload_mode: str = "prefetch"
    
    # gpu-layer-budget: GPU 上最多同时保留几层权重 (用来做 budget 扫描实验)
    # 例如 28 层的模型，budget=4，剩下的 24 层全部在 CPU 锁页内存中
    gpu_layer_budget: int = 4 