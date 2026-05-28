# 2026-05-28 EdgeInfer 验收记录

## 1. 仓库状态

- **分支**: master
- **最近提交**:
  ```
  84d372c feat(runtime): pipeline layer prefetch in generation runtime
  62772e6 chore(scripts): wire 8b offload acceptance scripts
  850586d feat(eval): add minimal generation acceptance runner
  ad1a897 test(eval): align correctness tests with tolerance API
  ba1c07d Merge pull request #2 from Richardo11chen/feat/qwen3-generation-runtime
  ```
- **工作区状态**: 干净，无未提交修改
- **未提交修改**: 无

## 2. A/B 分阶段验收表

| 阶段 | 成员 A 任务 | A 状态 | A 证据 | 成员 B 任务 | B 状态 | B 证据 | 阶段结论 |
|---|---|---|---|---|---|---|---|
| 5/9–5/11 环境+接口 | 环境搭建、模型下载、接口设计 | 通过 | pyproject.toml、docs/interface_design.md (714行)、两个模型完整下载 | 环境搭建、模型下载、接口设计 | 通过 | scripts/download_model.py、scripts/check_env.py、环境检查通过 (CUDA 13.0, RTX 4060 8GB) | **通过** |
| 5/12–5/16 单层+Provider | 实现 Qwen3 单层结构 | 通过 | model/layers.py、model/qwen3.py、model/kv_cache.py、model/rope.py; test_qwen3_decoder_layer.py 通过 | 实现 WeightProvider 原型 | 通过 | runtime/weight_provider.py (含 ResidentWeightProvider)、test_weight_providers.py 通过 | **通过** |
| 5/17–5/20 权重+offload | 完成 Qwen3-1.7B 权重加载 | 通过 | test_real_qwen3_1_7b_smoke.py 1 passed; GPU fp16 correctness: logits max_abs_error=0.0, tokens mismatch_count=0 | 完成朴素 offload | 通过 | runtime/offload_naive.py、NaiveOffloadWeightProvider 8B 真实运行通过 (peak 3120 MiB) | **通过** |
| 5/21–5/23 decode+prefetch | 完成 decode 与 token 对齐 | 通过 | eval/correctness.py; 1.7B GPU fp16 correctness: tokens mismatch_count=0; 8B naive 输出 "Hello," 与 HF 一致 | 完成 prefetch 原型 | 通过 | runtime/offload_prefetch.py (CUDA stream + pin memory + LRU); pipeline overlap 已验证: avg copy 40ms vs naive 395ms (10x) | **通过** |
| 5/24–5/28 8B 集成 | 集成模型实现与 offloading runtime | **通过** | Qwen3-8B naive offload 真实运行通过; Qwen3-8B prefetch offload 真实运行通过; BenchmarkHarness CSV 产出; pipeline overlap 确认 | 集成模型实现与 offloading runtime | **通过** | 同左 | **通过** |

## 3. 本次补齐内容

本次验收前完成的 4 个 commit：

1. **`ad1a897` test(eval): align correctness tests with tolerance API** — `compare_logits()` 添加默认 tolerance 值，修正测试中错误的参数名 `atol`→`abs_tol`、`rtol`→`rel_tol`。92 passed。

2. **`850586d` feat(eval): add minimal generation acceptance runner** — 新增 `eval/run_generation.py` 作为 resident/naive/prefetch 三模式的统一 CLI 入口；在 `GenerationRuntime` 中添加 benchmark hook 调用（prefill start/end、decode token timing）。

3. **`62772e6` chore(scripts): wire 8b offload acceptance scripts** — 填充 `scripts/run_8b_naive_offload.sh` 和 `scripts/run_8b_prefetch_offload.sh`，均调用 `eval.run_generation`。

4. **`84d372c` feat(runtime): pipeline layer prefetch in generation runtime** — 重写 layer 调度为 pipeline 模式：先 prefetch layer 0 + sync，随后每轮先 prefetch layer N+1、再 compute layer N、再 sync layer N+1。H2D copy 与 GPU compute 形成 overlap。新增 6 个 `_RecordingProvider` 调用顺序测试验证 pipeline 正确性。

## 4. 真实运行结果

| 项目 | Command | Result | Pass/Fail/Skip | 备注 |
|---|---|---|---|---|
| pytest 全量 | `uv run pytest -q` | 92 passed | **通过** | — |
| check_env | `uv run python scripts/check_env.py` | CUDA 可用, RTX 4060 8GB | **通过** | EDGEINFER_MODEL_PATH 未设置 |
| 1.7B smoke | `uv run pytest tests/test_real_qwen3_1_7b_smoke.py -q` | 1 passed | **通过** | — |
| 1.7B correctness | `bash scripts/run_1_7b_correctness.sh models/qwen3-1.7b "Hello" 4 cuda float16 1e-2 1e-2` | logits max_abs_error=0.0, tokens mismatch_count=0 | **通过** | GPU fp16 完全对齐 |
| 8B naive offload | `bash scripts/run_8b_naive_offload.sh models/qwen3-8b "Hello" 1` | generated_text="Hello,", peak 3120 MiB, prefill 14.5s | **通过** | 8B 模型在 8GB GPU 上成功运行 |
| 8B prefetch offload | `bash scripts/run_8b_prefetch_offload.sh models/qwen3-8b "Hello" 1` | generated_text="Hello,", peak 3584 MiB, prefill 3.8s | **通过** | Pipeline overlap: avg copy 40ms vs naive 395ms (10x) |

### 8B 运行详细指标

| Metric | Naive | Prefetch |
|---|---|---|
| prefill_latency_ms | 14470 | 3766 |
| ttft_ms | 14470 | 3766 |
| decode_tokens_per_sec | 137 | 138 |
| peak_memory_mib | 3120 | 3584 |
| total_copy_ms | 14205 | 1422 |
| avg_copy_ms | 395 | 40 |
| total_compute_ms | 233 | 282 |

### Pipeline Overlap 验证

Naive 模式每层 copy 时间 ~395ms（阻塞式 H2D），prefetch 模式每层 copy 时间 ~40ms（H2D 与上层的 GPU compute 重叠）。prefill 总延迟从 14.5s 降至 3.8s（3.8x 加速），证明 pipeline overlap 已形成。

## 5. 5/28 阶段结论

**是，已完成。** 截至 2026-05-28，项目完成了 5/24–5/28 阶段要求的 Qwen3-8B offload 初步运行。

运行证据：
- Qwen3-8B naive offload: RTX 4060 8GB 上成功运行，峰值显存 3120 MiB，生成 "Hello,"
- Qwen3-8B prefetch offload: 同上 GPU 成功运行，峰值显存 3584 MiB，prefill 3.8x 加速
- Benchmark CSV 已产出（含 per-layer copy/compute 时间）
- Pipeline overlap 通过 `_RecordingProvider` 测试和真实 8B per-layer timing 双重验证

## 6. 未纳入本次范围

- `scripts/run_budget_scan.sh` — 空文件，属于 5/29–6/2 阶段，不在本次补齐范围
- `eval/compare_vllm.py` — 空文件，属于 6/3–6/6 阶段，不在本次补齐范围
- 大规模 benchmark、vLLM/SGLang 对照 — 不在本次范围
