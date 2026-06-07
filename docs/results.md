# Results Guide

## Profiling Commands

生成 `torch.profiler` timeline：

```bash
bash scripts/run_profile_timeline.sh
```

可选生成 Nsight Systems trace：

```bash
bash scripts/run_nsys_profile.sh
```

## Trace Outputs

- `outputs/required/profile/naive_trace.json`
- `outputs/required/profile/prefetch_trace.json`
- `outputs/required/profile/naive_profiler_summary.txt`
- `outputs/required/profile/prefetch_profiler_summary.txt`
- `outputs/required/profile/nsys_naive.nsys-rep`（如果系统安装了 `nsys`）
- `outputs/required/profile/nsys_prefetch.nsys-rep`（如果系统安装了 `nsys`）

## 查看方式

- Chrome 打开 `chrome://tracing` 并加载 `*_trace.json`
- 或使用 PyTorch profiler / TensorBoard 查看 trace
- Nsight Systems 打开对应的 `.nsys-rep`
