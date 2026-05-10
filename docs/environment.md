# Environment

## Python

本项目使用 Python 3.11。

```bash
uv python pin 3.11
uv sync
```

## GPU and CUDA

项目使用 PyTorch CUDA 13.0 版本（cu130），支持 RTX 30/40/50 系列 GPU。

Requirements:

- NVIDIA GPU: RTX 30 series or newer
- NVIDIA driver: 580+ recommended
- System CUDA Toolkit is not required unless building custom CUDA extensions

Check:

```bash
nvidia-smi
uv run python scripts/check_env.py
```

## Main dependencies

- torch
- transformers
- tokenizers
- safetensors
- huggingface-hub
- numpy
- einops
- pandas
- matplotlib
- nvidia-ml-py
- psutil

## Model files

模型权重不允许上传到 GitHub 仓库。

设置模型路径的环境变量：

```bash
export EDGEINFER_MODEL_PATH=/path/to/Qwen3-8B
```

推荐的目录结构：

```text
models/
├── Qwen3-1.7B/
└── Qwen3-8B/
```

## Install

```bash
uv sync
uv run python scripts/check_env.py
```
