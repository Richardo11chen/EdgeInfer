from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    bos_token_id: int | None
    eos_token_id: int | list[int] | None
    pad_token_id: int | None
    torch_dtype: str | None
    tie_word_embeddings: bool

    @classmethod
    def from_model_dir(cls, model_dir: str) -> "ModelConfig":
        config_path = Path(model_dir) / "config.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))

        hidden_size = int(data["hidden_size"])
        num_attention_heads = int(data["num_attention_heads"])
        head_dim = int(data.get("head_dim", hidden_size // num_attention_heads))

        return cls(
            model_type=str(data["model_type"]),
            vocab_size=int(data["vocab_size"]),
            hidden_size=hidden_size,
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=int(data["num_key_value_heads"]),
            head_dim=head_dim,
            rms_norm_eps=float(data["rms_norm_eps"]),
            rope_theta=float(data.get("rope_theta", 10000.0)),
            max_position_embeddings=int(data["max_position_embeddings"]),
            bos_token_id=data.get("bos_token_id"),
            eos_token_id=data.get("eos_token_id"),
            pad_token_id=data.get("pad_token_id"),
            torch_dtype=data.get("torch_dtype"),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
        )
