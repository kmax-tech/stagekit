from typing import Dict, Type

from stagekit.parsers.base import BaseResponseParser
from stagekit.utils.registry import LazyRegistry

#: The framework has no parsers of its own; a topic adds its entries
#: when loaded (stagekit/run/topics.py).
PARSER_MODULE_MAP: Dict[str, str] = {}

PARSER_REGISTRY = LazyRegistry("parser", PARSER_MODULE_MAP)


def register_parser(name: str):
    """Decorator that registers a parser class with type validation."""
    def decorator(cls: Type[BaseResponseParser]):
        if not issubclass(cls, BaseResponseParser):
            raise TypeError(
                f"Parser '{cls.__name__}' must inherit from BaseResponseParser"
            )
        PARSER_REGISTRY.register(name)(cls)
        return cls
    return decorator


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------

def get_parser(name: str) -> BaseResponseParser:
    """Instantiate and return a parser by name."""
    parser_cls = PARSER_REGISTRY.get(name)
    return parser_cls()
