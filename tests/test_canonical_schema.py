"""Tests for the canonical record schema contract."""

import pytest

from stagekit.stages.canonical.schema import (
    validate_canonical_record,
    validate_canonical_batch,
    get_binary_label,
    ALL_FIELDS,
)


# ------------------------------------------------------------------
# Valid records
# ------------------------------------------------------------------


def test_minimal_valid_record():
    errors = validate_canonical_record({"id": "x", "text": "hello"})
    assert errors == []


def test_full_valid_record():
    record = {
        "id": "test:1",
        "text": "Some argument text",
        "labels": {"fallacy_status": "fallacy", "fallacy_type": "authority"},
        "context": {"title": "Thread title"},
        "_meta": {"source": "test_dataset", "canonical_version": "1"},
        "components": [{"text": "premise", "type": "premise"}],
        "enrichments": {"edus": []},
    }
    errors = validate_canonical_record(record)
    assert errors == []


# ------------------------------------------------------------------
# Missing required fields
# ------------------------------------------------------------------


def test_missing_id():
    errors = validate_canonical_record({"text": "hello"})
    assert any("'id'" in e for e in errors)


def test_missing_text():
    errors = validate_canonical_record({"id": "x"})
    assert any("'text'" in e for e in errors)


def test_missing_both():
    errors = validate_canonical_record({})
    assert len(errors) == 2


# ------------------------------------------------------------------
# Type enforcement
# ------------------------------------------------------------------


def test_labels_must_be_dict():
    errors = validate_canonical_record({"id": "x", "text": "t", "labels": [1, 2]})
    assert any("'labels' must be a dict" in e for e in errors)


def test_context_must_be_dict():
    errors = validate_canonical_record({"id": "x", "text": "t", "context": "string"})
    assert any("'context' must be a dict" in e for e in errors)


def test_meta_must_be_dict():
    errors = validate_canonical_record({"id": "x", "text": "t", "_meta": 42})
    assert any("'_meta' must be a dict" in e for e in errors)


def test_enrichments_must_be_dict():
    errors = validate_canonical_record({"id": "x", "text": "t", "enrichments": "bad"})
    assert any("'enrichments' must be a dict" in e for e in errors)


def test_components_must_be_list():
    errors = validate_canonical_record({"id": "x", "text": "t", "components": "bad"})
    assert any("'components' must be a list" in e for e in errors)


# ------------------------------------------------------------------
# Strict mode
# ------------------------------------------------------------------


def test_strict_rejects_unknown_fields():
    errors = validate_canonical_record(
        {"id": "x", "text": "t", "extra_field": 1}, strict=True
    )
    assert any("unknown fields" in e for e in errors)


def test_strict_allows_all_known_fields():
    record = {field: "v" for field in ALL_FIELDS}
    record["labels"] = {}
    record["context"] = {}
    record["_meta"] = {}
    record["enrichments"] = {}
    record["components"] = []
    errors = validate_canonical_record(record, strict=True)
    assert errors == []


# ------------------------------------------------------------------
# Batch validation
# ------------------------------------------------------------------


def test_batch_valid():
    records = [
        {"id": "1", "text": "a"},
        {"id": "2", "text": "b", "labels": {"fallacy_status": "non-fallacy"}},
    ]
    validate_canonical_batch(records)  # should not raise


def test_batch_raises_on_invalid_record():
    records = [
        {"id": "1", "text": "ok"},
        {"text": "missing id"},
    ]
    with pytest.raises(ValueError, match="Invalid canonical records"):
        validate_canonical_batch(records)


# ------------------------------------------------------------------
# get_binary_label
# ------------------------------------------------------------------


def test_binary_label_fallacy_status():
    assert get_binary_label({"fallacy_status": "fallacy"}) == 1
    assert get_binary_label({"fallacy_status": "non-fallacy"}) == 0


def test_binary_label_fallacy_exists():
    assert get_binary_label({"fallacy_exists": 1}) == 1
    assert get_binary_label({"fallacy_exists": 0}) == 0


def test_binary_label_invalid_fallacy_status():
    with pytest.raises(ValueError, match="fallacy_status must be one of"):
        get_binary_label({"fallacy_status": "maybe"})


def test_binary_label_invalid_fallacy_exists():
    with pytest.raises(ValueError, match="fallacy_exists must be 0 or 1"):
        get_binary_label({"fallacy_exists": 2})


def test_binary_label_unknown_key():
    with pytest.raises(KeyError, match="No known binary label key"):
        get_binary_label({"category": "authority"})


def test_binary_label_prefers_fallacy_status():
    """fallacy_status takes priority when both keys present."""
    assert get_binary_label({"fallacy_status": "non-fallacy", "fallacy_exists": 1}) == 0


# ------------------------------------------------------------------
# Dataset.get_stratify_labels() — with the argument loader's keys
# ------------------------------------------------------------------

KEYS = ("fallacy_status", "fallacy_exists", "label")


def test_stratify_label_takes_the_first_key_present():
    from stagekit.data_loader.base import Dataset

    ds = Dataset(stratify_keys=KEYS, records=[
        {"id": "a", "labels": {"fallacy_exists": 1}},
        {"id": "b", "labels": {"fallacy_exists": 0, "fallacy_status": "non-fallacy"}},
        {"id": "c", "labels": {"unknown": "x"}},
        {"id": "d"},
    ])
    assert ds.get_stratify_labels() == [1, "non-fallacy", -1, -1]


def test_no_stratify_keys_means_no_stratification():
    from stagekit.data_loader.base import Dataset

    ds = Dataset(records=[{"id": "a", "labels": {"fallacy_status": "fallacy"}}])
    assert ds.get_stratify_labels() is None


def test_dataset_get_stratify_labels():
    from stagekit.data_loader.base import Dataset

    ds = Dataset(stratify_keys=KEYS, records=[
        {"id": "a", "labels": {"fallacy_status": "fallacy"}},
        {"id": "b", "labels": {"fallacy_status": "non-fallacy"}},
        {"id": "c", "labels": {"fallacy_status": "fallacy"}},
    ])
    result = ds.get_stratify_labels()
    assert result == ["fallacy", "non-fallacy", "fallacy"]


def test_dataset_get_stratify_labels_none():
    from stagekit.data_loader.base import Dataset

    ds = Dataset(stratify_keys=KEYS, records=[{"id": "a", "text": "x"}])
    assert ds.get_stratify_labels() is None


def test_dataset_get_stratify_labels_single_class():
    from stagekit.data_loader.base import Dataset

    ds = Dataset(stratify_keys=KEYS, records=[
        {"id": "a", "labels": {"fallacy_status": "fallacy"}},
        {"id": "b", "labels": {"fallacy_status": "fallacy"}},
    ])
    # Dataset returns raw labels; single-class filtering is split-layer policy
    assert ds.get_stratify_labels() == ["fallacy", "fallacy"]


# ------------------------------------------------------------------
# _resolve_stratify (split-layer single-class policy)
# ------------------------------------------------------------------


def test_resolve_stratify_single_class_returns_none():
    from stagekit.data_loader.base import Dataset
    from stagekit.splits.strategies import _resolve_stratify

    ds = Dataset(stratify_keys=KEYS, records=[
        {"id": "a", "labels": {"fallacy_status": "fallacy"}},
        {"id": "b", "labels": {"fallacy_status": "fallacy"}},
    ])
    assert _resolve_stratify(ds) is None


def test_resolve_stratify_non_stratifiable_returns_none():
    from stagekit.splits.strategies import _resolve_stratify

    class PlainDataset:
        ids = ["a"]

    assert _resolve_stratify(PlainDataset()) is None
