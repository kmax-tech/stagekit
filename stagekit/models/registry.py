from typing import Dict, Type, TypeVar

from stagekit.models.base import BaseModel
from stagekit.utils.registry import LazyRegistry

MODEL_MODULE_MAP: Dict[str, str] = {
    "llm": "stagekit.models.llm_model",
}

MODEL_REGISTRY = LazyRegistry("model", MODEL_MODULE_MAP)

_M = TypeVar("_M", bound=BaseModel)


def register_model(name: str):
    """Decorator that registers a model class with type validation.

    Preserves the decorated class's type (TypeVar), so subclasses keep
    their own attributes for type checkers.
    """
    def decorator(cls: Type[_M]) -> Type[_M]:
        if not issubclass(cls, BaseModel):
            raise TypeError(
                f"Model '{cls.__name__}' must inherit from BaseModel"
            )
        MODEL_REGISTRY.register(name)(cls)
        return cls
    return decorator


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------

def get_model_class(name: str) -> Type[BaseModel]:
    return MODEL_REGISTRY.get(name)


def load_model(model_cfg, llm_task_cfg=None, meta_cfg=None):
    model_cls = get_model_class(model_cfg.model_type)

    model = model_cls.from_config(
        model_cfg=model_cfg,
        llm_task_cfg=llm_task_cfg,
        meta_cfg=meta_cfg,
    )

    model_dir = model_cfg.model_dir
    if model_dir:
        model.load(model_dir)

    return model
