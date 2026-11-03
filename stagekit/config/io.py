from typing import Type, Any
from omegaconf import OmegaConf
import stagekit.config.paths as s

from stagekit.config.classes import ModelConfig



def merge_to_dataclass(cls: Type[Any], override) -> Any:
    """Merge an OmegaConf/dict override into a dataclass default and return a plain object.

    This is the single implementation of the structured-merge-to_object
    pattern used throughout the config layer.
    """
    base = OmegaConf.structured(cls())
    merged = OmegaConf.merge(base, override)
    return OmegaConf.to_object(merged)


def load_named_config(name: str, subdir: str, cls: Type[Any]) -> Any:
    """
    Load YAML config from config/<subdir>/<name>.yaml and convert to dataclass 'cls'.
    """

    cfg_path = s.CONFIG_DIR / subdir / f"{name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config '{name}' not found at: {cfg_path}")

    yaml_cfg = OmegaConf.load(cfg_path)
    obj = merge_to_dataclass(cls, yaml_cfg)

    if not isinstance(obj, cls):
        raise TypeError(f"Loaded config is not of type {cls.__name__}")

    return obj


def load_model_config(name: str) -> ModelConfig:
    """Load a model config YAML into the flat ModelConfig.

    Unknown fields cause an error (strict dataclass merge). Thin wrapper
    over ``load_named_config`` — it existed as a line-for-line copy once
    (audit C5), which is exactly how the two loaders drift.
    """
    return load_named_config(name, "model", ModelConfig)

