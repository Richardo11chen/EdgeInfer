from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from model.kv_cache import KVCache
from model.rope import RotaryEmbedding
from weights.model_config import ModelConfig
from weights.weight_spec import LayerWeights


class Qwen3DecoderLayer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.rope = RotaryEmbedding(config=config, device=torch.device("cpu"))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        layer_weights: LayerWeights,
        kv_cache: KVCache,
        layer_id: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, layer_weights.get("input_layernorm.weight"))
        hidden_states = self._self_attn(hidden_states, position_ids, layer_weights, kv_cache, layer_id)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, layer_weights.get("post_attention_layernorm.weight"))
        hidden_states = self._mlp(hidden_states, layer_weights)
        hidden_states = residual + hidden_states

        return hidden_states

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.config.rms_norm_eps)
        return x_norm * weight

    def _self_attn(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        layer_weights: LayerWeights,
        kv_cache: KVCache,
        layer_id: int,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        q = F.linear(hidden_states, layer_weights.get("self_attn.q_proj.weight"))
        k = F.linear(hidden_states, layer_weights.get("self_attn.k_proj.weight"))
        v = F.linear(hidden_states, layer_weights.get("self_attn.v_proj.weight"))

        q = q.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)

        cos, sin = self.rope.get_cos_sin(position_ids)
        cos = cos.to(dtype=q.dtype, device=q.device)
        sin = sin.to(dtype=q.dtype, device=q.device)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        start_pos = int(position_ids[0, 0].item())
        kv_cache.append(layer_id=layer_id, key=k, value=v, start_pos=start_pos)
        end_pos = start_pos + seq_len
        all_k, all_v = kv_cache.get(layer_id=layer_id, end_pos=end_pos)

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_heads}) must be divisible by num_key_value_heads ({num_kv_heads})"
            )
        repeat_factor = num_heads // num_kv_heads
        all_k = all_k.repeat_interleave(repeat_factor, dim=1)
        all_v = all_v.repeat_interleave(repeat_factor, dim=1)

        attn_scores = torch.matmul(q, all_k.transpose(-2, -1)) / math.sqrt(head_dim)

        query_pos = position_ids.to(device=attn_scores.device, dtype=torch.long).unsqueeze(-1)
        key_pos = torch.arange(end_pos, device=attn_scores.device, dtype=torch.long).view(1, 1, 1, -1)
        causal_mask = key_pos <= query_pos.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))
        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(dtype=attn_scores.dtype)
        attn_output = torch.matmul(attn_probs, all_v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)

        output = F.linear(attn_output, layer_weights.get("self_attn.o_proj.weight"))
        return output

    def _mlp(self, hidden_states: torch.Tensor, layer_weights: LayerWeights) -> torch.Tensor:
        gate = F.linear(hidden_states, layer_weights.get("mlp.gate_proj.weight"))
        up = F.linear(hidden_states, layer_weights.get("mlp.up_proj.weight"))
        activated = F.silu(gate) * up
        down = F.linear(activated, layer_weights.get("mlp.down_proj.weight"))
        return down

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x * cos + rotated * sin
