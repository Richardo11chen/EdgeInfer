from __future__ import annotations


class WeightNameMapper:
    def hf_global_name_to_internal(self, hf_name: str) -> str:
        mapping = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "model.norm.weight": "model.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }
        if hf_name not in mapping:
            raise KeyError(f"Unsupported global weight name: {hf_name}")
        return mapping[hf_name]

    def hf_layer_name_to_internal(self, hf_name: str, layer_id: int) -> str:
        prefix = f"model.layers.{layer_id}."
        if not hf_name.startswith(prefix):
            raise KeyError(f"Layer name does not match layer_id={layer_id}: {hf_name}")
        return hf_name[len(prefix) :]

    def internal_layer_name_to_hf(self, internal_name: str, layer_id: int) -> str:
        return f"model.layers.{layer_id}.{internal_name}"
