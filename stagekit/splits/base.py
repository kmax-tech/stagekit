from typing import ClassVar, Optional, Protocol, runtime_checkable


@runtime_checkable
class Stratifiable(Protocol):
    """Dataset that can provide labels for stratified splitting."""

    def get_stratify_labels(self) -> list | None: ...


class SplitStrategy:
    strategy_name: ClassVar[Optional[str]] = None

    """Interface for split strategies."""
    def apply(self, dataset):
        raise NotImplementedError
