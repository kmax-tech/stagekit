from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, Optional, TypedDict

from stagekit.utils.serialization import load_json


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    test_size: float
    dev_size: Optional[float] = None
    seed: int = 42


# ------------------------------------------------------------------
# Split result contract
# ------------------------------------------------------------------

EVALUABLE_DEFAULT = True


class SplitResult(TypedDict, total=False):
    """Typed contract for persisted split files.

    Every split producer (strategies, k-fold) must
    return a dict matching this shape.  ``train_ids`` is always present;
    other keys depend on strategy.

    Non-evaluable splits (e.g. no_split) set ``evaluable: False``
    explicitly.  When absent, the split is assumed evaluable.

    Note on IDs vs indices:
        Persisted splits use **string record IDs** (``*_ids``).
        The in-memory ``Dataset`` uses **integer indices**
        (``train_idx``, ``dev_idx``, ``test_idx``).
        ``SplitManager.load()`` translates between the two via
        an ``id → index`` map built from the loaded dataset.
    """

    # Always present
    train_ids: list[str]
    # Hold-out evaluation set (standard train/test splits)
    test_ids: list[str]
    # Per-fold validation set (k-fold only — NOT the same as dev_ids)
    val_ids: list[str]
    # Development / hyperparameter-tuning set (three-way splits only)
    dev_ids: list[str]
    # Metadata
    evaluable: bool
    strategy: str
    seed: int
    test_size: float
    dev_size: float


def validate_split_result(data: dict) -> list[str]:
    """Validate a split result dict at runtime.

    Returns a list of error strings (empty means valid).
    """
    errors = []

    if "train_ids" not in data:
        errors.append("missing required key 'train_ids'")
    elif not isinstance(data["train_ids"], list):
        errors.append("'train_ids' must be a list")

    for key in ("test_ids", "val_ids", "dev_ids"):
        if key in data and not isinstance(data[key], list):
            errors.append(f"'{key}' must be a list")

    if "evaluable" in data and not isinstance(data["evaluable"], bool):
        errors.append("'evaluable' must be a bool")

    # An evaluable split must have test_ids or val_ids
    is_evaluable = data.get("evaluable", EVALUABLE_DEFAULT)
    if is_evaluable:
        has_eval_ids = bool(data.get("test_ids") or data.get("val_ids"))
        if not has_eval_ids:
            errors.append(
                "evaluable split must contain non-empty 'test_ids' or 'val_ids'"
            )

    return errors


# ------------------------------------------------------------------
# K-fold assignment contract
# ------------------------------------------------------------------


class KFoldEntry(TypedDict):
    """A single fold within a KFoldAssignment."""
    fold: int
    train_ids: list[str]
    test_ids: list[str]


class KFoldAssignment(TypedDict):
    """Typed contract for persisted k-fold assignment files.

    Written by ``kfold_split`` stage into ``{split_dir}/kfold/``.
    Consumed by ``resolve_folds()`` in classify stages.

    Kept separate from ``SplitResult`` because the semantics differ:
    SplitResult is one split (train/test/dev), while KFoldAssignment
    contains N folds each with their own train/test partition.
    """
    n_folds: int
    seed: int
    n_examples: int
    folds: list[KFoldEntry]
    dataset_fingerprint: NotRequired[str]


def validate_kfold_assignment(data: dict) -> list[str]:
    """Validate a k-fold assignment dict at runtime.

    Returns a list of error strings (empty means valid).
    """
    errors = []

    for key in ("n_folds", "seed", "n_examples"):
        if key not in data:
            errors.append(f"missing required key '{key}'")
        elif not isinstance(data[key], int):
            errors.append(f"'{key}' must be an int")

    if "folds" not in data:
        errors.append("missing required key 'folds'")
    elif not isinstance(data["folds"], list):
        errors.append("'folds' must be a list")
    else:
        for i, fold in enumerate(data["folds"]):
            if not isinstance(fold, dict):
                errors.append(f"folds[{i}] must be a dict")
                continue
            if "train_ids" not in fold:
                errors.append(f"folds[{i}] missing 'train_ids'")
            if "test_ids" not in fold:
                errors.append(f"folds[{i}] missing 'test_ids'")

        if not errors and data.get("n_folds") != len(data["folds"]):
            errors.append(
                f"n_folds ({data.get('n_folds')}) != len(folds) ({len(data['folds'])})"
            )

    return errors


def load_kfold_split(path: str | Path) -> KFoldAssignment:
    """Load and validate a k-fold assignment file.

    Normative way to load kfold splits from disk.  Analogous to
    ``load_split()`` for dataset-level SplitResult files.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the loaded data fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"K-fold assignment file not found: {path}")

    data: dict = load_json(path)

    errors = validate_kfold_assignment(data)
    if errors:
        raise ValueError(
            f"Invalid k-fold assignment file '{path}': {'; '.join(errors)}"
        )

    return data  # type: ignore[return-value]


def load_split(path: str | Path) -> SplitResult:
    """Load and validate a split file.

    This is the single normative way to load a split from disk.
    All consumers (SplitManager, stages) should
    use this instead of raw ``load_json()`` to ensure consistent
    validation.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the loaded data fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    data: dict = load_json(path)

    errors = validate_split_result(data)
    if errors:
        raise ValueError(
            f"Invalid split file '{path}': {'; '.join(errors)}"
        )

    return data  # type: ignore[return-value]
