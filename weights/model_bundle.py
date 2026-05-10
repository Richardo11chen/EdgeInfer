from __future__ import annotations

from dataclasses import dataclass

from weights.loader import WeightLoader
from weights.model_config import ModelConfig


@dataclass(frozen=True)
class ModelBundle:
    model_dir: str
    config: ModelConfig
    loader: WeightLoader


def open_model_bundle(model_dir: str) -> ModelBundle:
    config = ModelConfig.from_model_dir(model_dir)
    loader = WeightLoader(model_dir=model_dir, config=config)
    return ModelBundle(model_dir=model_dir, config=config, loader=loader)
