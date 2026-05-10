# EdgeInfer 接口设计文档

## 1. 设计目标

本设计文档定义 EdgeInfer 第一版的开发契约，目标是让模型计算与权重调度清晰解耦，使成员 A/B 可以并行开发并在统一接口下集成。

第一版强调以下原则：
- 模型结构配置、实验配置、权重加载、运行时调度相互分离。
- 以“可验证、可集成、可实验”为第一优先级，不追求工业级完整性。
- 接口语义、tensor shape、device/dtype、生命周期均显式约定，避免隐式行为。

## 2. 模块边界

建议目录结构：
- `weights/model_config.py`
- `weights/weight_spec.py`
- `weights/name_mapping.py`
- `weights/loader.py`
- `weights/model_bundle.py`
- `model/qwen3.py`
- `model/layers.py`
- `model/rope.py`
- `model/kv_cache.py`
- `runtime/generation.py`
- `runtime/weight_provider.py`
- `runtime/offload_naive.py`
- `runtime/offload_prefetch.py`
- `runtime/memory.py`
- `eval/experiment_config.py`
- `eval/benchmark.py`
- `eval/correctness.py`

模块职责：
- `weights/*`：负责模型制品解析、模型结构配置读取、权重命名映射、按层/全局权重加载。
- `model/*`：负责纯 Qwen3 计算，不承担权重来源管理与调度。
- `runtime/*`：负责 prefill/decode 编排与不同 offload 策略下的权重提供。
- `eval/*`：负责实验参数配置、正确性验证、性能评测。

## 3. ModelConfig

`ModelConfig` 放在 `weights/model_config.py`。

理由：
- `ModelConfig` 来源于模型目录中的 `config.json`。
- 它描述模型结构本身，不描述实验如何运行。
- 它与权重加载、权重映射属于同一层（模型制品层）。

接口草案：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    bos_token_id: int | None
    eos_token_id: int | list[int] | None
    pad_token_id: int | None
    torch_dtype: str | None
    tie_word_embeddings: bool

    @classmethod
    def from_model_dir(cls, model_dir: str) -> "ModelConfig":
        ...
```

固定约束：
- `ModelConfig` 不包含 `model_path`。
- `ModelConfig` 不包含 `batch_size`。
- `ModelConfig` 不包含 `prompt_length`、`generate_length`。
- `ModelConfig` 不包含 `offload_mode`。
- `ModelConfig` 不包含 benchmark 参数。

## 4. ExperimentConfig

`ExperimentConfig` 放在 `eval/experiment_config.py`。

接口草案：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    model_dir: str
    device: str
    dtype: str
    offload_mode: str
    gpu_layer_budget: int | None
    batch_size: int
    prompt_length: int
    generate_length: int
    max_sequence_length: int
    output_dir: str
    seed: int
```

约束说明：
- `ExperimentConfig` 描述“一次实验如何运行”。
- `offload_mode` 只允许：`resident`、`naive`、`prefetch`。
- `ExperimentConfig` 不能替代 `ModelConfig`，两者职责不可混合。

## 5. 权重命名规范

内部层权重 key 统一为去掉 HuggingFace 层号前缀后的名称：
- `input_layernorm.weight`
- `self_attn.q_proj.weight`
- `self_attn.k_proj.weight`
- `self_attn.v_proj.weight`
- `self_attn.o_proj.weight`
- `post_attention_layernorm.weight`
- `mlp.gate_proj.weight`
- `mlp.up_proj.weight`
- `mlp.down_proj.weight`

全局权重 key：
- `model.embed_tokens.weight`
- `model.norm.weight`
- `lm_head.weight`

语义约束：
- 层内权重属于 `LayerWeights`。
- 全局权重属于 `GlobalWeights`。
- `tie_word_embeddings=True` 时，接口层仍显式暴露 `lm_head`。

## 6. 权重数据结构

放在 `weights/weight_spec.py`。

接口草案：

```python
from dataclasses import dataclass
from collections.abc import Iterable, Mapping
import torch

@dataclass(frozen=True)
class LayerWeights:
    layer_id: int
    tensors: Mapping[str, torch.Tensor]

    def get(self, name: str) -> torch.Tensor:
        ...

    def keys(self) -> Iterable[str]:
        ...

@dataclass(frozen=True)
class GlobalWeights:
    embed_tokens: torch.Tensor
    final_norm: torch.Tensor
    lm_head: torch.Tensor
```

语义约束：
- `LayerWeights` 表示单个 Decoder Layer 的完整权重集合。
- 传入模型计算前，所有 tensor 必须已在目标 device。
- 模型层不负责 `.to(device)`。
- 模型层不负责释放权重。
- `GlobalWeights` 在一次 `generate` 生命周期内保持可用。

## 7. WeightLoader

放在 `weights/loader.py`。

接口草案：

```python
from collections.abc import Iterable

class WeightLoader:
    def __init__(self, model_dir: str, config: ModelConfig):
        ...

    def load_global_weights(self) -> GlobalWeights:
        ...

    def load_layer_weights(self, layer_id: int) -> LayerWeights:
        ...

    def iter_layer_ids(self) -> Iterable[int]:
        ...
```

固定约束：
- `WeightLoader` 可以解析 `safetensors`。
- `WeightLoader` 可以使用 `name_mapping`。
- `WeightLoader` 默认返回 CPU 权重。
- `WeightLoader` 不处理 GPU budget。
- `WeightLoader` 不处理 CUDA stream。
- `WeightLoader` 不处理 `resident`、`naive`、`prefetch` 策略。

## 8. WeightNameMapper

放在 `weights/name_mapping.py`。

接口草案：

```python
class WeightNameMapper:
    def hf_global_name_to_internal(self, hf_name: str) -> str:
        ...

    def hf_layer_name_to_internal(
        self,
        hf_name: str,
        layer_id: int,
    ) -> str:
        ...

    def internal_layer_name_to_hf(
        self,
        internal_name: str,
        layer_id: int,
    ) -> str:
        ...
```

模块约束：
- 该模块只处理名称转换。
- 不读取文件。
- 不处理 tensor。
- 不处理 device。

## 9. ModelBundle

放在 `weights/model_bundle.py`。

接口草案：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelBundle:
    model_dir: str
    config: ModelConfig
    loader: WeightLoader

def open_model_bundle(model_dir: str) -> ModelBundle:
    ...
```

约束说明：
- `open_model_bundle` 读取 `config.json`。
- `open_model_bundle` 构造 `WeightLoader`。
- `open_model_bundle` 不预加载所有权重。
- runtime 统一接收 `ModelBundle`，不分散传递 `model_dir/config/loader`。

## 10. WeightProvider

放在 `runtime/weight_provider.py`。

接口草案：

```python
from typing import Protocol

class WeightProvider(Protocol):
    def get_global_weights(self) -> GlobalWeights:
        ...

    def prefetch_layer(self, layer_id: int) -> None:
        ...

    def synchronize_layer(self, layer_id: int) -> None:
        ...

    def get_layer_weights(self, layer_id: int) -> LayerWeights:
        ...

    def release_layer(self, layer_id: int) -> None:
        ...

    def close(self) -> None:
        ...
```

方法语义：
- `get_global_weights`：可阻塞，返回全局权重。
- `prefetch_layer`：不应阻塞，提示 provider 准备某层。
- `synchronize_layer`：阻塞，等待某层权重可安全使用。
- `get_layer_weights`：返回已经准备好的层权重，不做重拷贝。
- `release_layer`：释放或回收该层 GPU 权重。
- `close`：释放 provider 持有资源。

固定约束：
- `GenerationRuntime` 不直接传 `torch.cuda.Stream`。
- CUDA stream/event 仅是 `PrefetchOffloadWeightProvider` 的内部细节。

## 11. 三种 WeightProvider 实现

`ResidentWeightProvider`：
- 初始化阶段将所有层权重加载到 GPU。
- `prefetch_layer` 为空操作。
- `synchronize_layer` 为空操作。
- `release_layer` 为空操作。

`NaiveOffloadWeightProvider`：
- 权重主要保留在 CPU。
- `prefetch_layer` 为空操作。
- `synchronize_layer` 同步将该层复制到 GPU。
- `get_layer_weights` 返回 GPU 层权重。
- `release_layer` 释放上一层 GPU 权重。

`PrefetchOffloadWeightProvider`：
- 权重主要保留在 CPU pinned memory。
- `prefetch_layer` 使用内部 CUDA stream 异步复制。
- `synchronize_layer` 等待对应 event。
- `get_layer_weights` 返回完成拷贝后的 GPU 层权重。
- `release_layer` 按 `gpu_layer_budget` 回收。

## 12. 模型计算接口

### Qwen3DecoderLayer

放在 `model/layers.py`。

接口草案：

```python
class Qwen3DecoderLayer:
    def __init__(self, config: ModelConfig):
        ...

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        layer_weights: LayerWeights,
        kv_cache: "KVCache",
        layer_id: int,
    ) -> torch.Tensor:
        ...
```

输入输出约定：
- `hidden_states` shape：`[batch_size, seq_len, hidden_size]`
- `hidden_states` device：`cuda`
- `hidden_states` dtype：`fp16` 或 `bf16`
- `position_ids` shape：`[batch_size, seq_len]`
- `position_ids` dtype：`int64`
- `layer_weights` 必须为完整单层权重
- 输出 shape 与 `hidden_states` 一致

固定约束：
- `Qwen3DecoderLayer` 不读取 `safetensors`。
- `Qwen3DecoderLayer` 不知道 `offload_mode`。
- `Qwen3DecoderLayer` 不负责权重搬运。

### Qwen3Model

放在 `model/qwen3.py`。

接口草案：

```python
class Qwen3Model:
    def __init__(self, config: ModelConfig):
        ...

    def embed(
        self,
        input_ids: torch.Tensor,
        global_weights: GlobalWeights,
    ) -> torch.Tensor:
        ...

    def forward_layer(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        layer_weights: LayerWeights,
        kv_cache: "KVCache",
    ) -> torch.Tensor:
        ...

    def final_logits(
        self,
        hidden_states: torch.Tensor,
        global_weights: GlobalWeights,
    ) -> torch.Tensor:
        ...
```

固定约束：
- `Qwen3Model` 不持有完整权重。
- `Qwen3Model` 不持有 `WeightProvider`。
- `Qwen3Model` 不读取模型目录。
- `Qwen3Model` 只组织模型结构与计算。

## 13. KVCache

放在 `model/kv_cache.py`。

第一版使用静态预分配 KV Cache。

接口草案：

```python
class KVCache:
    def __init__(
        self,
        config: ModelConfig,
        batch_size: int,
        max_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        ...

    def append(
        self,
        layer_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int,
    ) -> None:
        ...

    def get(
        self,
        layer_id: int,
        end_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def reset(self) -> None:
        ...
```

shape 约定：
- `key_cache`：`[num_layers, batch_size, num_key_value_heads, max_sequence_length, head_dim]`
- `value_cache`：`[num_layers, batch_size, num_key_value_heads, max_sequence_length, head_dim]`

固定约束：
- 第一版不做 paged KV cache。
- `KVCache` 不管理权重。
- `KVCache` 不参与 offload 策略。

## 14. GenerationRuntime

放在 `runtime/generation.py`。

接口草案：

```python
class GenerationRuntime:
    def __init__(
        self,
        model: Qwen3Model,
        provider: WeightProvider,
        config: ModelConfig,
        benchmark: "BenchmarkHarness | None" = None,
    ):
        ...

    def prefill(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        ...

    def decode_one(
        self,
        input_ids: torch.Tensor,
        position_id: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        ...

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        ...
```

调度职责（单步语义）：
1. `get_global_weights`
2. embedding
3. 按 `layer_id` 遍历 decoder layer
4. prefetch 下一层
5. synchronize 当前层
6. get 当前层权重
7. forward 当前层
8. release 旧层
9. final norm + `lm_head`
10. greedy token selection

固定不负责项：
- `safetensors` 解析；
- 权重名映射；
- CUDA stream 细节；
- attention 内部计算；
- benchmark CSV 格式细节。

## 15. BenchmarkHarness

放在 `eval/benchmark.py`。

接口草案：

```python
class BenchmarkHarness:
    def on_prefill_start(self) -> None:
        ...

    def on_prefill_end(self) -> None:
        ...

    def on_decode_start(self) -> None:
        ...

    def on_decode_token_end(self, token_index: int) -> None:
        ...

    def on_layer_copy_start(self, layer_id: int) -> None:
        ...

    def on_layer_copy_end(self, layer_id: int) -> None:
        ...

    def on_layer_compute_start(self, layer_id: int) -> None:
        ...

    def on_layer_compute_end(self, layer_id: int) -> None:
        ...

    def get_summary(self) -> dict[str, float]:
        ...

    def export_csv(self, output_path: str) -> None:
        ...
```

约束说明：
- Benchmark 不侵入 `Qwen3DecoderLayer`。
- Provider 记录 copy。
- Runtime 记录 compute 与生成阶段。
- 输出指标包括：TTFT、prefill latency、decode tokens/s、peak memory、per-layer H2D copy time、per-layer compute time。

## 16. Provider 工厂函数

接口草案：

```python
def create_weight_provider(
    model_bundle: ModelBundle,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    gpu_layer_budget: int | None,
) -> WeightProvider:
    ...
```

约束说明：
- `mode` 只允许 `resident`、`naive`、`prefetch`。
- 工厂函数负责选择具体 Provider。
- 不在 `GenerationRuntime` 中写 `if/else` 区分 offload 策略。

## 17. 最小集成流程

伪代码：

```python
model_bundle = open_model_bundle(model_dir)

provider = create_weight_provider(
    model_bundle=model_bundle,
    mode=experiment_config.offload_mode,
    device=experiment_config.device,
    dtype=experiment_config.dtype,
    gpu_layer_budget=experiment_config.gpu_layer_budget,
)

model = Qwen3Model(model_bundle.config)

runtime = GenerationRuntime(
    model=model,
    provider=provider,
    config=model_bundle.config,
    benchmark=benchmark,
)

output_ids = runtime.generate(input_ids, max_new_tokens)
```

## 18. 并行开发边界

成员 A 负责：
- `model/qwen3.py`
- `model/layers.py`
- `model/rope.py`
- `model/kv_cache.py`
- `eval/correctness.py`

成员 A 依赖：
- `ModelConfig`
- `LayerWeights`
- `GlobalWeights`
- `WeightProvider` 协议

成员 B 负责：
- `weights/loader.py`
- `weights/name_mapping.py`
- `runtime/weight_provider.py`
- `runtime/offload_naive.py`
- `runtime/offload_prefetch.py`
- `runtime/memory.py`
- `eval/benchmark.py`

成员 B 依赖：
- `LayerWeights` key 规范
- `ModelConfig`
- `WeightLoader`
- `BenchmarkHarness`

## 19. 第一版必须固定的契约

1. `ModelConfig` 放在 `weights/model_config.py`，且只描述模型结构。
2. `LayerWeights` key 与 HuggingFace 去层号后的名称一致。
3. `WeightProvider` 不暴露 CUDA stream。
4. `KVCache` 使用静态预分配 shape。
5. 模型层只做计算，不做权重搬运。
6. Runtime 只编排流程，不解析 `safetensors`。
7. Benchmark 通过 hook 记录，不侵入模型计算代码。
