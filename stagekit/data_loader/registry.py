from typing import Dict, Type
from warnings import warn

from stagekit.config.classes import DataConfig
from stagekit.config.paths import project_path
from stagekit.splits.manager import SplitManager
from stagekit.data_loader.base import BaseDataLoader, Dataset
from stagekit.utils.registry import LazyRegistry

#: The framework has no data loaders of its own; a topic adds its entries
#: when loaded (stagekit/run/topics.py).
LOADER_MODULE_MAP: Dict[str, str] = {}

DATA_LOADER_REGISTRY = LazyRegistry("data loader", LOADER_MODULE_MAP)


def register_dataset(name: str):
    """Decorator that registers a data loader class with type validation."""
    def decorator(cls: Type[BaseDataLoader]):
        if not issubclass(cls, BaseDataLoader):
            raise TypeError(
                f"Data loader '{cls.__name__}' must inherit from BaseDataLoader"
            )
        DATA_LOADER_REGISTRY.register(name)(cls)
        return cls
    return decorator


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------

def load_dataset(data_cfg: DataConfig) -> Dataset:
    """Load a dataset using the configured data loader and apply splits."""
    loader_cls = DATA_LOADER_REGISTRY.get(data_cfg.data_loader)
    loader = loader_cls()

    dataset: Dataset = loader.load(data_cfg)

    if not data_cfg.split_id:
        raise ValueError(
            "data.split_id must be set. "
            "Use 'no_split' for full-dataset annotation runs."
        )

    # --------------------------------------------------
    # Special-case: no_split (full dataset for annotation, not evaluable)
    # --------------------------------------------------
    if data_cfg.split_id == "no_split":
        dataset.train_idx = list(range(len(dataset)))
        dataset.dev_idx = []
        dataset.test_idx = []

        return dataset

    # --------------------------------------------------
    # Normal path: persisted split
    # --------------------------------------------------
    split_mgr = SplitManager(project_path(data_cfg.split_dir))
    split = split_mgr.load(data_cfg.split_id, dataset=dataset)

    dataset.train_idx = split.get("train_idx", [])
    dataset.dev_idx = split.get("dev_idx", [])
    dataset.test_idx = split.get("test_idx", [])
    dataset.val_idx = split.get("val_idx", [])

    if (
        dataset.train_idx
        and dataset.test_idx
        and set(dataset.train_idx) == set(dataset.test_idx)
    ):
        warn(
            f"Train and test splits are identical (split_id={data_cfg.split_id}). "
            "This is usually intended only for debugging."
        )

    return dataset
