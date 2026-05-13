from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file as safe_save

from weights.model_bundle import ModelBundle, open_model_bundle
from weights.model_config import ModelConfig


# Small config for fast tests — shapes must be self-consistent.
# hidden_size must be divisible by num_attention_heads.
# head_dim = hidden_size // num_attention_heads.
TEST_CONFIG = ModelConfig(
    model_type="qwen2",
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    max_position_embeddings=128,
    bos_token_id=0,
    eos_token_id=2,
    pad_token_id=None,
    torch_dtype="float32",
    tie_word_embeddings=False,
)


def _make_layer_hf_tensors(layer_id: int, config: ModelConfig) -> dict[str, torch.Tensor]:
    """Build a dict of HF-weight-name → tensor for one decoder layer."""
    d = config.hidden_size
    d_i = config.intermediate_size
    tensors = {
        f"model.layers.{layer_id}.input_layernorm.weight": torch.randn(d),
        f"model.layers.{layer_id}.self_attn.q_proj.weight": torch.randn(d, d),
        f"model.layers.{layer_id}.self_attn.k_proj.weight": torch.randn(d, d),
        f"model.layers.{layer_id}.self_attn.v_proj.weight": torch.randn(d, d),
        f"model.layers.{layer_id}.self_attn.o_proj.weight": torch.randn(d, d),
        f"model.layers.{layer_id}.post_attention_layernorm.weight": torch.randn(d),
        f"model.layers.{layer_id}.mlp.gate_proj.weight": torch.randn(d_i, d),
        f"model.layers.{layer_id}.mlp.up_proj.weight": torch.randn(d_i, d),
        f"model.layers.{layer_id}.mlp.down_proj.weight": torch.randn(d, d_i),
    }
    return tensors


def _make_global_hf_tensors(config: ModelConfig) -> dict[str, torch.Tensor]:
    """Build a dict of HF-weight-name → tensor for global weights."""
    d = config.hidden_size
    v = config.vocab_size
    return {
        "model.embed_tokens.weight": torch.randn(v, d),
        "model.norm.weight": torch.randn(d),
        "lm_head.weight": torch.randn(v, d),
    }


def _write_safetensors(
    directory: Path, config: ModelConfig, num_shards: int = 1
) -> None:
    """Write synthetic safetensors files into *directory*."""
    tensors: dict[str, torch.Tensor] = {}
    tensors.update(_make_global_hf_tensors(config))
    for layer_id in range(config.num_hidden_layers):
        tensors.update(_make_layer_hf_tensors(layer_id, config))

    if num_shards > 1:
        all_keys = list(tensors.keys())
        chunk_size = (len(all_keys) + num_shards - 1) // num_shards
        for shard_idx in range(num_shards):
            start = shard_idx * chunk_size
            end = start + chunk_size
            shard_tensors = {k: tensors[k] for k in all_keys[start:end]}
            filename = f"model-{shard_idx + 1:05d}-of-{num_shards:05d}.safetensors"
            safe_save(shard_tensors, str(directory / filename))
    else:
        safe_save(tensors, str(directory / "model.safetensors"))


def _write_config(directory: Path, config: ModelConfig) -> None:
    cfg = {
        "model_type": config.model_type,
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "max_position_embeddings": config.max_position_embeddings,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "torch_dtype": config.torch_dtype,
        "tie_word_embeddings": config.tie_word_embeddings,
    }
    (directory / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


@pytest.fixture(scope="session")
def model_config() -> ModelConfig:
    return TEST_CONFIG


@pytest.fixture()
def temp_model_dir(
    tmp_path: Path, model_config: ModelConfig
) -> Generator[Path, None, None]:
    """Create a temporary model directory with synthetic weights."""
    _write_config(tmp_path, model_config)
    _write_safetensors(tmp_path, model_config, num_shards=1)
    yield tmp_path


@pytest.fixture()
def temp_model_dir_sharded(
    tmp_path: Path, model_config: ModelConfig
) -> Generator[Path, None, None]:
    """Create a temporary model directory with sharded synthetic weights."""
    _write_config(tmp_path, model_config)
    _write_safetensors(tmp_path, model_config, num_shards=2)
    yield tmp_path


@pytest.fixture()
def model_bundle(temp_model_dir: Path) -> ModelBundle:
    return open_model_bundle(str(temp_model_dir))
