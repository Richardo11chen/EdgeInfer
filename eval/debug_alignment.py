from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.kv_cache import KVCache
from model.qwen3 import Qwen3Model
from runtime.weight_provider import create_weight_provider
from weights.model_bundle import open_model_bundle


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[name]


def stage_stats(name: str, a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, bool, bool, bool]:
    diff = (a - b).abs()
    max_abs = float(diff.max().item())
    rel = diff / b.abs().clamp_min(1e-12)
    max_rel = float(rel.max().item())
    argmax_same = bool(torch.equal(torch.argmax(a[:, -1, :], dim=-1), torch.argmax(b[:, -1, :], dim=-1))) if a.ndim == 3 else True
    has_nan_inf = bool(
        torch.isnan(a).any().item()
        or torch.isinf(a).any().item()
        or torch.isnan(b).any().item()
        or torch.isinf(b).any().item()
    )
    bad = (max_abs > 1e-2 and max_rel > 1e-2) or has_nan_inf
    print(
        f"[{name}] max_abs_error={max_abs:.6f} max_rel_error={max_rel:.6f} "
        f"argmax_same={argmax_same} has_nan_inf={has_nan_inf} bad={bad}"
    )
    return max_abs, max_rel, argmax_same, has_nan_inf, bad


def run(model_path: str, prompt: str, device: str, dtype: str) -> None:
    torch_device = torch.device(device)
    torch_dtype = parse_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=True).to(torch_device)
    hf.eval()

    bundle = open_model_bundle(model_path)
    model = Qwen3Model(bundle.config)
    provider = create_weight_provider(bundle, "resident", torch_device, torch_dtype, None)

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(torch_device)
    seq_len = input_ids.shape[1]
    pos = torch.arange(seq_len, device=torch_device, dtype=torch.long).unsqueeze(0).expand(input_ids.shape[0], -1)

    kv = KVCache(
        config=bundle.config,
        batch_size=input_ids.shape[0],
        max_sequence_length=seq_len,
        device=torch_device,
        dtype=torch_dtype,
    )

    with torch.no_grad():
        gw = provider.get_global_weights()

        hf_out = hf.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hf_states = hf_out.hidden_states
        hf_embed = hf_states[0]
        hf_layer_states = hf_states[1 : 1 + bundle.config.num_hidden_layers]
        my_hidden = model.embed(input_ids, gw)
        stage_stats("embedding", my_hidden, hf_embed)

        first_bad_layer = None
        for lid in range(bundle.config.num_hidden_layers):
            lw = provider.get_layer_weights(lid)
            my_hidden = model.forward_layer(lid, my_hidden, pos, lw, kv)
            hf_layer = hf_layer_states[lid]
            _, _, _, _, bad = stage_stats(f"layer_{lid}", my_hidden, hf_layer)
            if first_bad_layer is None and bad:
                first_bad_layer = lid

        hf_final_norm = hf_out.last_hidden_state
        my_final_norm = my_hidden.float()
        my_final_norm = my_final_norm * torch.rsqrt(my_final_norm.pow(2).mean(dim=-1, keepdim=True) + bundle.config.rms_norm_eps)
        my_final_norm = (my_final_norm * gw.final_norm.float()).to(dtype=my_hidden.dtype)
        stage_stats("final_norm", my_final_norm, hf_final_norm)

        hf_logits = hf.lm_head(hf_final_norm)
        my_logits = model.final_logits(my_hidden, gw)
        stage_stats("final_logits", my_logits, hf_logits)

    print(f"first_bad_layer={first_bad_layer}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()
    run(args.model_path, args.prompt, args.device, args.dtype)


if __name__ == "__main__":
    main()
