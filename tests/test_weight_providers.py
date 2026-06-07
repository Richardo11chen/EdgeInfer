from __future__ import annotations

import pytest
import torch

from runtime.weight_provider import (
    NaiveOffloadWeightProvider,
    PrefetchOffloadWeightProvider,
    ResidentWeightProvider,
    create_weight_provider,
)
from weights.weight_spec import GlobalWeights, LayerWeights


CUDA_AVAILABLE = torch.cuda.is_available()


class TestFactory:
    def test_resident(self, model_bundle):
        provider = create_weight_provider(model_bundle, "resident", torch.device("cpu"), torch.float32, None)
        assert isinstance(provider, ResidentWeightProvider)

    def test_naive(self, model_bundle):
        provider = create_weight_provider(model_bundle, "naive", torch.device("cpu"), torch.float32, None)
        assert isinstance(provider, NaiveOffloadWeightProvider)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_prefetch(self, model_bundle):
        provider = create_weight_provider(model_bundle, "prefetch", torch.device("cuda"), torch.float32, None)
        assert isinstance(provider, PrefetchOffloadWeightProvider)

    def test_invalid_mode(self, model_bundle):
        with pytest.raises(ValueError, match="Unsupported offload mode"):
            create_weight_provider(model_bundle, "bad_mode", torch.device("cpu"), torch.float32, None)


class TestResidentProvider:
    def test_global_weights(self, model_bundle, model_config):
        provider = ResidentWeightProvider(model_bundle, torch.device("cpu"), torch.float32)
        gw = provider.get_global_weights()
        assert isinstance(gw, GlobalWeights)
        assert gw.embed_tokens.shape == (model_config.vocab_size, model_config.hidden_size)

    def test_all_layers_loaded(self, model_bundle, model_config):
        provider = ResidentWeightProvider(model_bundle, torch.device("cpu"), torch.float32)
        for i in range(model_config.num_hidden_layers):
            lw = provider.get_layer_weights(i)
            assert isinstance(lw, LayerWeights)
            assert lw.layer_id == i

    def test_prefetch_is_noop(self, model_bundle):
        provider = ResidentWeightProvider(model_bundle, torch.device("cpu"), torch.float32)
        provider.prefetch_layer(0)  # should not raise

    def test_synchronize_is_noop(self, model_bundle):
        provider = ResidentWeightProvider(model_bundle, torch.device("cpu"), torch.float32)
        provider.synchronize_layer(0)

    def test_release_is_noop(self, model_bundle):
        provider = ResidentWeightProvider(model_bundle, torch.device("cpu"), torch.float32)
        provider.release_layer(0)

    def test_close(self, model_bundle):
        provider = ResidentWeightProvider(model_bundle, torch.device("cpu"), torch.float32)
        provider.close()

    def test_layer_device(self, model_bundle):
        provider = ResidentWeightProvider(model_bundle, torch.device("cpu"), torch.float32)
        lw = provider.get_layer_weights(0)
        for t in lw.tensors.values():
            assert t.device.type == "cpu"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestNaiveProviderGPU:
    def test_global_weights(self, model_bundle, model_config):
        provider = NaiveOffloadWeightProvider(model_bundle, torch.device("cuda"), torch.float32)
        gw = provider.get_global_weights()
        assert gw.embed_tokens.device.type == "cuda"
        assert gw.embed_tokens.shape == (model_config.vocab_size, model_config.hidden_size)

    def test_synchronize_layer(self, model_bundle, model_config):
        provider = NaiveOffloadWeightProvider(model_bundle, torch.device("cuda"), torch.float32)
        provider.synchronize_layer(1)
        lw = provider.get_layer_weights(1)
        assert lw.layer_id == 1
        for t in lw.tensors.values():
            assert t.device.type == "cuda"

    def test_release_layer(self, model_bundle):
        provider = NaiveOffloadWeightProvider(model_bundle, torch.device("cuda"), torch.float32)
        provider.synchronize_layer(0)
        lw_before = provider.get_layer_weights(0)
        assert lw_before is not None
        provider.release_layer(0)
        with pytest.raises(KeyError):
            provider.get_layer_weights(0)

    def test_prefetch_is_noop(self, model_bundle):
        provider = NaiveOffloadWeightProvider(model_bundle, torch.device("cuda"), torch.float32)
        provider.prefetch_layer(0)

    def test_release_non_existent(self, model_bundle):
        provider = NaiveOffloadWeightProvider(model_bundle, torch.device("cuda"), torch.float32)
        provider.release_layer(99)

    def test_close(self, model_bundle):
        provider = NaiveOffloadWeightProvider(model_bundle, torch.device("cuda"), torch.float32)
        provider.close()

    def test_cpu_cache_stays_resident_after_release(self, model_bundle):
        provider = NaiveOffloadWeightProvider(model_bundle, torch.device("cuda"), torch.float32)
        assert len(provider._cpu_layer_cache) == model_bundle.config.num_hidden_layers
        provider.synchronize_layer(0)
        provider.release_layer(0)
        assert 0 in provider._cpu_layer_cache


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestPrefetchProviderGPU:
    def test_global_weights(self, model_bundle, model_config):
        provider = PrefetchOffloadWeightProvider(model_bundle, torch.device("cuda"), torch.float32, gpu_layer_budget=2)
        gw = provider.get_global_weights()
        assert gw.embed_tokens.device.type == "cuda"
        assert gw.embed_tokens.shape == (model_config.vocab_size, model_config.hidden_size)

    def test_prefetch_and_synchronize(self, model_bundle):
        provider = PrefetchOffloadWeightProvider(model_bundle, torch.device("cuda"), torch.float32, gpu_layer_budget=2)
        provider.prefetch_layer(0)
        provider.synchronize_layer(0)
        lw = provider.get_layer_weights(0)
        assert lw.layer_id == 0
        for t in lw.tensors.values():
            assert t.device.type == "cuda"

    def test_prefetch_multiple_layers(self, model_bundle):
        provider = PrefetchOffloadWeightProvider(model_bundle, torch.device("cuda"), torch.float32, gpu_layer_budget=4)
        for i in range(4):
            provider.prefetch_layer(i)
        for i in range(4):
            provider.synchronize_layer(i)
            lw = provider.get_layer_weights(i)
            assert lw.layer_id == i
            assert all(t.device.type == "cuda" for t in lw.tensors.values())

    def test_release_layer(self, model_bundle):
        provider = PrefetchOffloadWeightProvider(model_bundle, torch.device("cuda"), torch.float32, gpu_layer_budget=2)
        provider.prefetch_layer(0)
        provider.synchronize_layer(0)
        provider.release_layer(0)
        assert 0 in provider._cpu_cache

    def test_close(self, model_bundle):
        provider = PrefetchOffloadWeightProvider(model_bundle, torch.device("cuda"), torch.float32, gpu_layer_budget=2)
        provider.close()

    def test_gpu_layer_budget_eviction(self, model_bundle):
        provider = PrefetchOffloadWeightProvider(model_bundle, torch.device("cuda"), torch.float32, gpu_layer_budget=2)
        provider.prefetch_layer(0)
        provider.synchronize_layer(0)
        provider.prefetch_layer(1)
        provider.synchronize_layer(1)
        provider.release_layer(0)
        provider.prefetch_layer(2)
        provider.synchronize_layer(2)

        assert provider.get_layer_weights(1).layer_id == 1
        assert provider.get_layer_weights(2).layer_id == 2
        with pytest.raises(KeyError):
            provider.get_layer_weights(0)

    def test_gpu_layer_budget_none(self, model_bundle):
        provider = PrefetchOffloadWeightProvider(model_bundle, torch.device("cuda"), torch.float32, gpu_layer_budget=None)
        for i in range(4):
            lw = provider.get_layer_weights(i)
            assert lw.layer_id == i

    def test_budget_one_never_exceeds_gpu_cache(self, model_bundle):
        provider = PrefetchOffloadWeightProvider(model_bundle, torch.device("cuda"), torch.float32, gpu_layer_budget=1)
        provider.prefetch_layer(0)
        provider.synchronize_layer(0)
        assert len(provider._gpu_cache) == 1

        provider.prefetch_layer(1)
        assert len(provider._gpu_cache) == 1
        provider.release_layer(0)
        provider.synchronize_layer(1)
        assert len(provider._gpu_cache) == 1
        assert provider.get_layer_weights(1).layer_id == 1


class TestProviderCPU:
    """Provider tests on CPU — covers construction and basic access."""

    def test_resident_layer_dtype(self, model_bundle):
        provider = ResidentWeightProvider(model_bundle, torch.device("cpu"), torch.float64)
        lw = provider.get_layer_weights(0)
        for t in lw.tensors.values():
            assert t.dtype == torch.float64

    def test_naive_layer_dtype(self, model_bundle):
        provider = NaiveOffloadWeightProvider(model_bundle, torch.device("cpu"), torch.float64)
        provider.synchronize_layer(0)
        lw = provider.get_layer_weights(0)
        for t in lw.tensors.values():
            assert t.dtype == torch.float64

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_prefetch_layer_dtype(self, model_bundle):
        provider = PrefetchOffloadWeightProvider(model_bundle, torch.device("cuda"), torch.float64, gpu_layer_budget=2)
        provider.prefetch_layer(0)
        provider.synchronize_layer(0)
        lw = provider.get_layer_weights(0)
        for t in lw.tensors.values():
            assert t.dtype == torch.float64
