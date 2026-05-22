from __future__ import annotations

import torch

from model.kv_cache import KVCache
from model.qwen3 import Qwen3Model
from runtime.generation import GenerationRuntime
from runtime.weight_provider import ResidentWeightProvider
from weights.model_bundle import ModelBundle
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


class _FakeLoader:
    def __init__(self, config: ModelConfig):
        self.config = config

    def load_global_weights(self) -> GlobalWeights:
        return GlobalWeights(
            embed_tokens=torch.randn(self.config.vocab_size, self.config.hidden_size, dtype=torch.float32) * 0.02,
            final_norm=torch.ones(self.config.hidden_size, dtype=torch.float32),
            lm_head=torch.randn(self.config.vocab_size, self.config.hidden_size, dtype=torch.float32) * 0.02,
        )

    def load_layer_weights(self, layer_id: int) -> LayerWeights:
        hs = self.config.hidden_size
        isz = self.config.intermediate_size
        return LayerWeights(
            layer_id=layer_id,
            tensors={
                "input_layernorm.weight": torch.ones(hs, dtype=torch.float32),
                "self_attn.q_proj.weight": torch.randn(hs, hs, dtype=torch.float32) * 0.02,
                "self_attn.k_proj.weight": torch.randn(
                    self.config.num_key_value_heads * self.config.head_dim, hs, dtype=torch.float32
                )
                * 0.02,
                "self_attn.v_proj.weight": torch.randn(
                    self.config.num_key_value_heads * self.config.head_dim, hs, dtype=torch.float32
                )
                * 0.02,
                "self_attn.o_proj.weight": torch.randn(hs, hs, dtype=torch.float32) * 0.02,
                "post_attention_layernorm.weight": torch.ones(hs, dtype=torch.float32),
                "mlp.gate_proj.weight": torch.randn(isz, hs, dtype=torch.float32) * 0.02,
                "mlp.up_proj.weight": torch.randn(isz, hs, dtype=torch.float32) * 0.02,
                "mlp.down_proj.weight": torch.randn(hs, isz, dtype=torch.float32) * 0.02,
            },
        )

    def iter_layer_ids(self):
        return range(self.config.num_hidden_layers)


def make_runtime(config: ModelConfig) -> GenerationRuntime:
    model = Qwen3Model(config)
    bundle = ModelBundle(model_dir="", config=config, loader=_FakeLoader(config))
    provider = ResidentWeightProvider(bundle, torch.device("cpu"), torch.float32)
    return GenerationRuntime(model=model, provider=provider, config=config)


def test_prefill_logits_shape() -> None:
    config = make_config()
    runtime = make_runtime(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
    cache = KVCache(config, batch_size=2, max_sequence_length=16, device=torch.device("cpu"), dtype=torch.float32)

    logits = runtime.prefill(input_ids, cache)

    assert logits.shape == (2, 8, config.vocab_size)


def test_decode_one_logits_shape() -> None:
    config = make_config()
    runtime = make_runtime(config)

    prompt = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
    cache = KVCache(config, batch_size=2, max_sequence_length=16, device=torch.device("cpu"), dtype=torch.float32)
    runtime.prefill(prompt, cache)

    next_ids = torch.randint(0, config.vocab_size, (2, 1), dtype=torch.long)
    pos = torch.full((2, 1), 8, dtype=torch.long)
    logits = runtime.decode_one(next_ids, pos, cache)

    assert logits.shape == (2, 1, config.vocab_size)


def test_generate_runs_for_specified_steps() -> None:
    config = make_config()
    runtime = make_runtime(config)

    prompt = torch.randint(0, config.vocab_size, (2, 6), dtype=torch.long)
    out = runtime.generate(prompt, max_new_tokens=5)

    assert out.shape == (2, 11)


def test_kv_cache_length_grows_with_decode() -> None:
    config = make_config()
    runtime = make_runtime(config)

    prompt_len = 4
    prompt = torch.randint(0, config.vocab_size, (1, prompt_len), dtype=torch.long)
    cache = KVCache(config, batch_size=1, max_sequence_length=12, device=torch.device("cpu"), dtype=torch.float32)

    logits = runtime.prefill(prompt, cache)
    for layer_id in range(config.num_hidden_layers):
        key, _ = cache.get(layer_id, end_pos=prompt_len)
        assert key.shape[2] == prompt_len

    for step in range(3):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        pos = torch.full((1, 1), prompt_len + step, dtype=torch.long)
        logits = runtime.decode_one(next_token, pos, cache)
        expected_len = prompt_len + step + 1
        for layer_id in range(config.num_hidden_layers):
            key, _ = cache.get(layer_id, end_pos=expected_len)
            assert key.shape[2] == expected_len
