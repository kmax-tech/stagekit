from stagekit.splits.registry import register_split_strategy
from stagekit.splits.base import SplitStrategy, Stratifiable
from stagekit.splits.config import SplitResult


def _resolve_stratify(dataset) -> list | None:
    """Get stratification labels via Stratifiable protocol, or None.

    Returns None when the dataset does not support stratification or when
    all labels belong to a single class (stratified split is impossible).
    """
    if not isinstance(dataset, Stratifiable):
        return None
    labels = dataset.get_stratify_labels()
    if labels is None:
        return None
    return labels if len(set(labels)) > 1 else None


@register_split_strategy("no_split")
class NoSplitStrategy(SplitStrategy):
    """All IDs go to train; empty test set. Use for full-dataset annotation."""

    def __init__(self, **kwargs):
        pass

    def apply(self, dataset) -> SplitResult:
        return SplitResult(
            train_ids=list(dataset.ids),
            test_ids=[],
            evaluable=False,
        )
