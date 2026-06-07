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

下载模型权重方法

```bash
uv run python scripts/download_model.py qwen3-1.7b
uv run python scripts/download_model.py qwen3-8b
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

## Quickstart

公开入口统一为根目录下的 `./run`：

```bash
cp edgeinfer.env.example edgeinfer.env
vim edgeinfer.env

./run smoke
./run prof
./run report
./run final
```

## vLLM Runner

vLLM 对照实验现在通过 `uv` 的 `vllm` dependency group 管理，不再使用单独的 `requirements-vllm.txt`。

安装包含 vLLM 的环境：

```bash
uv sync --group vllm
```

运行 vLLM 对照脚本时，脚本会自动使用：

```bash
uv run --group vllm ...
```
