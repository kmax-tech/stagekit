from typing import Dict, Type

from stagekit.splits.base import SplitStrategy
from stagekit.utils.registry import LazyRegistry

STRATEGY_MODULE_MAP: Dict[str, str] = {
    "no_split": "stagekit.splits.strategies",
}

SPLIT_STRATEGY_REGISTRY = LazyRegistry("split strategy", STRATEGY_MODULE_MAP)


def register_split_strategy(name: str):
    """Decorator that registers a split strategy class with type validation."""
    def decorator(cls: Type[SplitStrategy]):
        if not issubclass(cls, SplitStrategy):
            raise TypeError(
                f"Split strategy '{cls.__name__}' must inherit from SplitStrategy"
            )
        SPLIT_STRATEGY_REGISTRY.register(name)(cls)
        return cls
    return decorator


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------

def create_split_strategy(name: str, **kwargs) -> SplitStrategy:
    """Instantiate a registered split strategy."""
    strategy_cls = SPLIT_STRATEGY_REGISTRY.get(name)
    return strategy_cls(**kwargs)
