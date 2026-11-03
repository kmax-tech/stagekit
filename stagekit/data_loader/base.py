from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from stagekit.config.classes import DataConfig
from typing import List, ClassVar, Optional, Tuple

@dataclass
class Dataset:
    """A loaded dataset: one record per item, in file order, each with a string id.

    Ids are not checked for uniqueness here (``check_data_loader`` in
    stagekit/pipeline/conformance.py does, on a sample).

    The framework reads only ``id``, and for stratified splits the
    ``labels`` keys the loader names (``stratify_keys``); every other
    field of a record — text, components, … — is the loader's, and so the
    topic's. The pipeline gets the records as they are (``to_records``).
    """

    # One dict per item, each with a string "id".
    records: List[dict]

    # Dataset splits — integer indices into ``records``.
    # Persisted splits use string IDs (*_ids in SplitResult);
    # SplitManager.load() translates IDs → indices at load time.
    train_idx: List[int] = field(default_factory=list)
    dev_idx: List[int] = field(default_factory=list)
    test_idx: List[int] = field(default_factory=list)
    val_idx: List[int] = field(default_factory=list)

    # Canonical dataset identifier for caching (set by loader)
    cache_id: Optional[str] = None

    # Keys of a record's ``labels`` tried in order for a stratification
    # label (set by loader from its ``stratify_keys``); empty: none.
    stratify_keys: Tuple[str, ...] = ()

    def __post_init__(self):
        missing = [i for i, r in enumerate(self.records) if not isinstance(r.get("id"), str)]
        if missing:
            raise ValueError(f"records without a string 'id' at positions {missing[:5]}")

    @property
    def ids(self) -> List[str]:
        return [r["id"] for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def to_records(self, indices: List[int] | None = None) -> List[dict]:
        """Records for pipeline consumption (shallow copies; all, or *indices*)."""
        idxs = indices if indices is not None else range(len(self.records))
        return [dict(self.records[i]) for i in idxs]

    def get_stratify_labels(self) -> list | None:
        """One stratification label per record, or None if there are none.

        None when the loader names no ``stratify_keys`` or no record has
        ``labels``. Otherwise each record's value under the first of the
        keys its labels carry; -1 for a record without labels or without
        any of the keys (the split layer decides what to do with them).
        """
        labels = [r.get("labels") for r in self.records]
        if not self.stratify_keys or all(lbl is None for lbl in labels):
            return None
        return [
            next((lbl[k] for k in self.stratify_keys if k in lbl), -1) if lbl else -1
            for lbl in labels
        ]


class BaseDataLoader(ABC):
    dataset_name: ClassVar[Optional[str]] = None
    # The labels keys a stratified split may use, in priority order
    # (Dataset.stratify_keys); empty: the loader's data is not stratified.
    stratify_keys: ClassVar[Tuple[str, ...]] = ()

    @abstractmethod
    def load(self, cfg: DataConfig) -> Dataset:
        pass
