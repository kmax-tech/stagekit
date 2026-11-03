"""The general Dataset: records addressed by id, stratification from their labels."""

import pytest

from stagekit.data_loader.base import Dataset


def test_a_record_without_a_string_id_is_refused():
    with pytest.raises(ValueError, match=r"without a string 'id' at positions \[1\]"):
        Dataset(records=[{"id": "a"}, {"id": 7}])


def test_records_without_labels_stratify_as_minus_one():
    ds = Dataset(
        records=[{"id": "a", "labels": {"fallacy_status": "fallacy"}}, {"id": "b"}],
        stratify_keys=("fallacy_status",),
    )
    assert ds.get_stratify_labels() == ["fallacy", -1]


def test_to_records_selects_by_index():
    ds = Dataset(records=[{"id": "a"}, {"id": "b"}, {"id": "c"}])
    assert [r["id"] for r in ds.to_records([2, 0])] == ["c", "a"]
    assert len(ds) == 3
