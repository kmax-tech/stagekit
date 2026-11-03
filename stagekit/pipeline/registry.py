"""
Pipeline stage registry with lazy module loading.

Two-part registration mechanism:

    1. The module map — declares which module contains each stage.
       This is the lazy-import lookup table: the registry uses it to
       know *where* to import from without loading all stage modules
       eagerly at startup. The framework has no stages of its own
       (STAGE_MODULE_MAP is empty); a topic adds its entries when loaded
       (stagekit/run/topics.py).

    2. @register_stage("name")  — decorator applied to the PipelineStage
       subclass.  When the module is imported (lazily, via the map), this
       decorator registers the class in the runtime registry.

These are complementary, not duplicated.  Without the map, the registry
wouldn't know where to find unloaded stages.  Without the decorator, the
class wouldn't be registered even after import.
"""
from typing import Callable

from stagekit.utils.registry import LazyRegistry


STAGE_MODULE_MAP: dict[str, str] = {}

STAGE_REGISTRY = LazyRegistry("stage", STAGE_MODULE_MAP)
register_stage = STAGE_REGISTRY.register


def get_stage_fn(name: str) -> Callable:
    return STAGE_REGISTRY.get(name)
