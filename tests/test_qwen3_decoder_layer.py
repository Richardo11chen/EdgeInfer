from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.kv_cache import KVCache
from model.layers import Qwen3DecoderLayer
from model.qwen3 import Qwen3Model
from model.rope import RotaryEmbedding
from weights.model_config import ModelConfig
from weights.weight_spec import GlobalWeights, LayerWeights


def make_config() -> ModelConfig:
    return ModelConfig(
        model_type="qwen3",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_position_embeddings=1024,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        torch_dtype="float32",
        tie_word_embeddings=False,
    )


def make_layer_weights(config: ModelConfig, layer_id: int = 0) -> LayerWeights:
    hs = config.hidden_size
    isz = config.intermediate_size
    return LayerWeights(
        layer_id=layer_id,
        tensors={
            "input_layernorm.weight": torch.ones(hs, dtype=torch.float32),
            "self_attn.q_proj.weight": torch.randn(hs, hs, dtype=torch.float32) * 0.02,
            "self_attn.k_proj.weight": torch.randn(
                config.num_key_value_heads * config.head_dim, hs, dtype=torch.float32
            )
            * 0.02,
            "self_attn.v_proj.weight": torch.randn(
                config.num_key_value_heads * config.head_dim, hs, dtype=torch.float32
            )
            * 0.02,
            "self_attn.o_proj.weight": torch.randn(hs, hs, dtype=torch.float32) * 0.02,
            "post_attention_layernorm.weight": torch.ones(hs, dtype=torch.float32),
            "mlp.gate_proj.weight": torch.randn(isz, hs, dtype=torch.float32) * 0.02,
            "mlp.up_proj.weight": torch.randn(isz, hs, dtype=torch.float32) * 0.02,
            "mlp.down_proj.weight": torch.randn(hs, isz, dtype=torch.float32) * 0.02,
        },
    )


def make_global_weights(config: ModelConfig) -> GlobalWeights:
    return GlobalWeights(
        embed_tokens=torch.randn(config.vocab_size, config.hidden_size, dtype=torch.float32) * 0.02,
        final_norm=torch.ones(config.hidden_size, dtype=torch.float32),
        lm_head=torch.randn(config.vocab_size, config.hidden_size, dtype=torch.float32) * 0.02,
    )


def test_rotary_embedding_get_cos_sin_broadcast_shape() -> None:
    config = make_config()
    rope = RotaryEmbedding(config, device=torch.device("cpu"))
    position_ids = torch.arange(8, dtype=torch.long).unsqueeze(0).repeat(2, 1)

    cos, sin = rope.get_cos_sin(position_ids)

    assert cos.shape == (2, 1, 8, config.head_dim)
    assert sin.shape == (2, 1, 8, config.head_dim)


def test_kv_cache_append_get_shape() -> None:
    config = make_config()
    cache = KVCache(
        config=config,
        batch_size=2,
        max_sequence_length=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    key = torch.randn(2, config.num_key_value_heads, 8, config.head_dim, dtype=torch.float32)
    value = torch.randn(2, config.num_key_value_heads, 8, config.head_dim, dtype=torch.float32)
    cache.append(layer_id=0, key=key, value=value, start_pos=0)

    out_k, out_v = cache.get(layer_id=0, end_pos=8)
    assert out_k.shape == key.shape
    assert out_v.shape == value.shape


def test_decoder_layer_forward_shapes_gqa_seq1_seq8_and_no_nan() -> None:
    config = make_config()
    layer = Qwen3DecoderLayer(config)
    weights = make_layer_weights(config, layer_id=0)

    cache = KVCache(
        config=config,
        batch_size=2,
        max_sequence_length=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    hidden_1 = torch.randn(2, 1, config.hidden_size, dtype=torch.float32)
    pos_1 = torch.zeros((2, 1), dtype=torch.long)
    out_1 = layer.forward(hidden_1, pos_1, weights, cache, layer_id=0)
    assert out_1.shape == hidden_1.shape
    assert not torch.isnan(out_1).any()

    hidden_8 = torch.randn(2, 8, config.hidden_size, dtype=torch.float32)
    pos_8 = torch.arange(8, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    out_8 = layer.forward(hidden_8, pos_8, weights, cache, layer_id=1)
    assert out_8.shape == hidden_8.shape
    assert not torch.isnan(out_8).any()


def test_qwen3_model_embed_and_final_logits_shape() -> None:
    config = make_config()
    model = Qwen3Model(config)
    global_weights = make_global_weights(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
    hidden_states = model.embed(input_ids, global_weights)
    assert hidden_states.shape == (2, 8, config.hidden_size)

    logits = model.final_logits(hidden_states, global_weights)
    assert logits.shape == (2, 8, config.vocab_size)
    assert not torch.isnan(logits).any()
