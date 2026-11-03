"""Tests for PipelineDataStore (audit gap D2 — previously untested)."""

import pytest

from stagekit.pipeline.data_store import PipelineDataStore


class TestAddOutput:
    def test_stores_and_retrieves(self):
        store = PipelineDataStore()
        store.add_output("s1", "labels", [1, 0, 1])
        assert store.get("s1", "labels") == [1, 0, 1]

    def test_duplicate_field_fails_loud(self):
        store = PipelineDataStore()
        store.add_output("s1", "labels", [1])
        with pytest.raises(ValueError, match="already exists"):
            store.add_output("s1", "labels", [0])

    def test_expected_len_mismatch_fails_loud(self):
        """The index-alignment constraint the class docstring demands."""
        store = PipelineDataStore()
        with pytest.raises(ValueError, match="3 values"):
            store.add_output("s1", "labels", [1, 0, 1], expected_len=5)

    def test_expected_len_match_passes(self):
        store = PipelineDataStore()
        store.add_output("s1", "labels", [1, 0], expected_len=2)
        assert store.get("s1", "labels") == [1, 0]

    def test_marks_dirty(self):
        store = PipelineDataStore()
        assert store._dirty is False
        store.add_output("s1", "labels", [1])
        assert store._dirty is True


class TestClearStage:
    def test_clears_and_allows_republish(self):
        store = PipelineDataStore()
        store.add_output("s1", "labels", [1])
        store.clear_stage("s1")
        assert store.get("s1", "labels") is None
        store.add_output("s1", "labels", [0])  # must not raise
        assert store.get("s1", "labels") == [0]

    def test_clearing_unknown_stage_is_a_noop(self):
        store = PipelineDataStore()
        store.clear_stage("ghost")
        assert store.is_empty()


class TestEmptiness:
    def test_fresh_store_is_empty(self):
        assert PipelineDataStore().is_empty()

    def test_cleared_stage_counts_as_empty(self):
        store = PipelineDataStore()
        store.add_output("s1", "f", [1])
        store.clear_stage("s1")
        assert store.is_empty()
