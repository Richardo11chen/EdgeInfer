from __future__ import annotations

from pathlib import Path

import pytest

from weights.model_bundle import open_model_bundle


MODEL_DIR = Path("models/qwen3-1.7b")


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="local models/qwen3-1.7b not found")
def test_real_qwen3_1_7b_weight_loading_smoke() -> None:
    bundle = open_model_bundle(str(MODEL_DIR))
    config = bundle.config
    loader = bundle.loader

    global_weights = loader.load_global_weights()
    assert global_weights.embed_tokens.shape == (config.vocab_size, config.hidden_size)
    assert global_weights.final_norm.shape == (config.hidden_size,)
    assert global_weights.lm_head.shape == (config.vocab_size, config.hidden_size)

    layer0 = loader.load_layer_weights(0)
    last_layer_id = config.num_hidden_layers - 1
    layer_last = loader.load_layer_weights(last_layer_id)

    expected_q = (config.num_attention_heads * config.head_dim, config.hidden_size)
    expected_kv = (config.num_key_value_heads * config.head_dim, config.hidden_size)
    expected_o = (config.hidden_size, config.num_attention_heads * config.head_dim)

    for layer in (layer0, layer_last):
        assert layer.get("input_layernorm.weight").shape == (config.hidden_size,)
        assert layer.get("self_attn.q_proj.weight").shape == expected_q
        assert layer.get("self_attn.k_proj.weight").shape == expected_kv
        assert layer.get("self_attn.q_norm.weight").shape == (config.head_dim,)
        assert layer.get("self_attn.k_norm.weight").shape == (config.head_dim,)
        assert layer.get("self_attn.v_proj.weight").shape == expected_kv
        assert layer.get("self_attn.o_proj.weight").shape == expected_o
        assert layer.get("post_attention_layernorm.weight").shape == (config.hidden_size,)
        assert layer.get("mlp.gate_proj.weight").shape == (config.intermediate_size, config.hidden_size)
        assert layer.get("mlp.up_proj.weight").shape == (config.intermediate_size, config.hidden_size)
        assert layer.get("mlp.down_proj.weight").shape == (config.hidden_size, config.intermediate_size)

    assert list(loader.iter_layer_ids()) == list(range(config.num_hidden_layers))

    if config.tie_word_embeddings:
        assert global_weights.lm_head.data_ptr() == global_weights.embed_tokens.data_ptr()
