from __future__ import annotations

import pytest
import torch

from weights.loader import WeightLoader
from weights.weight_spec import GLOBAL_WEIGHT_KEYS, LAYER_WEIGHT_KEYS, GlobalWeights, LayerWeights


class TestWeightLoader:
    def test_open_single_file(self, temp_model_dir, model_config):
        loader = WeightLoader(str(temp_model_dir), model_config)
        assert len(loader._name_to_file) > 0

    def test_open_sharded_files(self, temp_model_dir_sharded, model_config):
        loader = WeightLoader(str(temp_model_dir_sharded), model_config)
        assert len(loader._name_to_file) > 0

    def test_load_global_weights_shape(self, model_bundle, model_config):
        gw = model_bundle.loader.load_global_weights()
        assert isinstance(gw, GlobalWeights)
        assert gw.embed_tokens.shape == (model_config.vocab_size, model_config.hidden_size)
        assert gw.final_norm.shape == (model_config.hidden_size,)
        assert gw.lm_head.shape == (model_config.vocab_size, model_config.hidden_size)

    def test_load_global_weights_on_cpu(self, model_bundle):
        gw = model_bundle.loader.load_global_weights()
        assert gw.embed_tokens.device.type == "cpu"
        assert gw.final_norm.device.type == "cpu"
        assert gw.lm_head.device.type == "cpu"

    def test_load_layer_weights_shape(self, model_bundle, model_config):
        lw = model_bundle.loader.load_layer_weights(0)
        assert isinstance(lw, LayerWeights)
        assert lw.layer_id == 0
        d = model_config.hidden_size
        d_i = model_config.intermediate_size

        assert lw.tensors["input_layernorm.weight"].shape == (d,)
        assert lw.tensors["self_attn.q_proj.weight"].shape == (d, d)
        assert lw.tensors["self_attn.k_proj.weight"].shape == (d, d)
        assert lw.tensors["self_attn.v_proj.weight"].shape == (d, d)
        assert lw.tensors["self_attn.o_proj.weight"].shape == (d, d)
        assert lw.tensors["post_attention_layernorm.weight"].shape == (d,)
        assert lw.tensors["mlp.gate_proj.weight"].shape == (d_i, d)
        assert lw.tensors["mlp.up_proj.weight"].shape == (d_i, d)
        assert lw.tensors["mlp.down_proj.weight"].shape == (d, d_i)

    def test_load_layer_weights_on_cpu(self, model_bundle):
        lw = model_bundle.loader.load_layer_weights(0)
        for t in lw.tensors.values():
            assert t.device.type == "cpu"

    def test_load_all_layers(self, model_bundle, model_config):
        for layer_id in model_bundle.loader.iter_layer_ids():
            lw = model_bundle.loader.load_layer_weights(layer_id)
            assert lw.layer_id == layer_id
            assert set(lw.tensors.keys()) == set(LAYER_WEIGHT_KEYS)

    def test_num_hidden_layers(self, model_bundle, model_config):
        layer_ids = list(model_bundle.loader.iter_layer_ids())
        assert layer_ids == list(range(model_config.num_hidden_layers))

    def test_sharded_vs_single_identical(self, temp_model_dir, temp_model_dir_sharded, model_config):
        loader_single = WeightLoader(str(temp_model_dir), model_config)
        loader_sharded = WeightLoader(str(temp_model_dir_sharded), model_config)

        gw_single = loader_single.load_global_weights()
        gw_sharded = loader_sharded.load_global_weights()
        assert torch.equal(gw_single.embed_tokens, gw_sharded.embed_tokens)
        assert torch.equal(gw_single.final_norm, gw_sharded.final_norm)
        assert torch.equal(gw_single.lm_head, gw_sharded.lm_head)

    def test_missing_tensor_raises(self, model_bundle):
        with pytest.raises(KeyError):
            model_bundle.loader._load_tensor("nonexistent.tensor")

    def test_iter_layer_ids(self, model_bundle, model_config):
        ids = list(model_bundle.loader.iter_layer_ids())
        assert ids == [0, 1, 2, 3]

    def test_loader_contains_all_expected_keys(self, model_bundle):
        loader = model_bundle.loader
        all_expected = list(GLOBAL_WEIGHT_KEYS)
        for i in range(loader.config.num_hidden_layers):
            for key in LAYER_WEIGHT_KEYS:
                all_expected.append(f"model.layers.{i}.{key}")
        for name in all_expected:
            assert name in loader._name_to_file, f"Missing key: {name}"
