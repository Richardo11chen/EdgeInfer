# Qwen3 Offload 推理框架接口协议

## 1. ModelConfig (数据类)
- **作用**：保存模型结构参数，摆脱对 HuggingFace `transformers` 库的运行时依赖。
- **字段约定**：
  - `vocab_size` (int): 词表大小
  - `hidden_size` (int): 隐藏层维度
  - `intermediate_size` (int): FFN 隐层维度
  - `num_hidden_layers` (int): Decoder 层数
  - `num_attention_heads` (int): Q 头数
  - `num_key_value_heads` (int): KV 头数 (GQA)
  - `rms_norm_eps` (float): Norm 层的 epsilon

## 2. LayerWeights (数据类)
- **作用**：表示单层 Decoder Layer 的权重集合。
- **字段约定**：一个包含该层所有 Tensor 的 Dict。键名统一规范为：
  `attn.q_proj`, `attn.k_proj`, `attn.v_proj`, `attn.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`, `input_layernorm`, `post_attention_layernorm`。

## 3. WeightProvider (由成员 B 实现，成员 A 调用)
- `get_layer_weights(layer_id: int) -> Dict[str, torch.Tensor]`：阻塞获取当前层权重（GPU端）。
- `prefetch_layer(layer_id: int, stream: torch.cuda.Stream) -> None`：异步预取下一层。
- `release_layer(layer_id: int) -> None`：释放当前层 GPU 显存。

## 4. KVCache (由成员 A 实现，成员 B 会监控其显存占用)
- `update(layer_id: int, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`：拼接并返回历史 KV Cache。

## 5. GenerationRuntime
- **作用**：组织 prefill 和 decode 循环的主控类。成员 B 会在这里注入 `WeightProvider`。

## 6. BenchmarkHarness (由成员 B 实现)
- `start_timer()`, `record_prefill()`, `record_decode_token()`：统一测速打点。
- `get_peak_memory() -> float`：统计 GPU 显存峰值。
- `export_csv(filepath: str)`：输出实验数据。