# EdgeInfer Final Report

## Prefetch 加速来源分析

为了补齐 prefetch 相对 naive offload 的加速来源证据，本项目提供了基于 `torch.profiler` 的 timeline trace：

- `outputs/required/profile/naive_trace.json`
- `outputs/required/profile/prefetch_trace.json`
- `outputs/required/profile/naive_profiler_summary.txt`
- `outputs/required/profile/prefetch_profiler_summary.txt`

如果系统安装了 Nsight Systems，还可以额外生成：

- `outputs/required/profile/nsys_naive.nsys-rep`
- `outputs/required/profile/nsys_prefetch.nsys-rep`

从实现语义上看，naive offload 在每层计算前才执行同步 H2D 拷贝，因此 layer weight copy 与当前层 compute 基本串行。对应的 profiler 标记为：

- `edgeinfer_h2d_layer_{i}`
- `edgeinfer_compute_layer_{i}`

Prefetch offload 则使用独立 CUDA stream，在 layer `i` 计算期间提前发起 layer `i+1` 的 H2D copy。由于权重首先常驻于 CPU pinned memory，`non_blocking=True` 的 H2D copy 可以真正异步提交；随后通过 CUDA event 和 `synchronize_layer()` 中的流间等待，保证 layer `i+1` 开始计算前权重已经就绪。

因此，在 timeline 中应当观察到：

- naive trace 中，`edgeinfer_h2d_layer_i` 与 `edgeinfer_compute_layer_i` 主要呈串行排列；
- prefetch trace 中，`edgeinfer_h2d_layer_{i+1}` 会在 `edgeinfer_compute_layer_i` 期间出现时间重叠；
- 这种重叠不会 100% 完全隐藏拷贝，但会减少可见的 copy stall。

这组 trace 的作用是为 final report 提供可视化证据，证明 prefetch 的收益来自 copy/compute overlap，而不仅仅是预算或缓存命中带来的偶然波动。
