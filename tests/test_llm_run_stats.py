"""The numbers a run reports about itself: LLMRunStats and the batch summary.

``LLMRunStats`` is written from up to 50 worker threads and read by
``RunManager.save_llm_stage_stats``, which drops the result into
``llm_stats/<stage>.json``. That file is the only record of how much of a
run was cached, retried or lost — so a counter that drifts, a dict that
keeps inventing keys, or a ``reset()`` that forgets a field all show up as
a plausible-looking number nobody can check afterwards.

``LLMCallResult.summarize`` is the same story one level up: it becomes the
``StageResult`` metrics of every LLM stage.
"""

from threading import Barrier, Thread

import pytest

from stagekit.models.base import LLMCallResult, LLMRunStats


def _result(**overrides) -> LLMCallResult:
    """A successful live-call result, with *overrides* applied."""
    base = dict(value={"a": 1}, source="llm", success=True, attempts=1)
    base.update(overrides)
    return LLMCallResult(**base)


class TestConcurrentRecording:
    def test_counters_and_summary_survive_concurrent_writers(self):
        """The lock is not decorative: the default config runs 50 workers into
        one stats object while the summary may be taken at any moment. A read
        that iterated the dicts unlocked would blow up on a growing one."""
        stats = LLMRunStats()
        workers, per_worker = 4, 100
        start = Barrier(workers + 1)

        def _record(worker: int) -> None:
            start.wait(timeout=10)
            for i in range(per_worker):
                stats.record_llm_call()
                stats.record_parse_success(f"parser-{worker}-{i}")
                stats.record_failed_item("vllm", f"{worker}-{i}")

        threads = [Thread(target=_record, args=(w,)) for w in range(workers)]
        for t in threads:
            t.start()
        start.wait(timeout=10)
        for _ in range(100):
            stats.summary()
        for t in threads:
            t.join(timeout=10)

        summary = stats.summary()
        assert summary["total_calls"] == workers * per_worker
        assert summary["llm_calls"] == workers * per_worker
        assert summary["parse_success"] == workers * per_worker
        assert len(summary["per_parser"]) == workers * per_worker
        assert len(summary["failed_items"]["vllm"]) == workers * per_worker


class TestSummaryShape:
    def test_summary_hands_out_plain_dicts(self):
        """A defaultdict written to the stats file would answer every question
        with a zero — a mistyped error key in an analysis script would read as
        'this never happened'."""
        stats = LLMRunStats()
        stats.record_error("fake:timeout")
        stats.record_parse_failure("binary")

        summary = stats.summary()

        with pytest.raises(KeyError):
            summary["errors"]["fake:auth"]
        with pytest.raises(KeyError):
            summary["per_parser"]["other_parser"]

    def test_items_are_grouped_by_backend(self):
        """Which endpoint lost the items is the first question after a bad
        run, so the grouping is the point of these two fields."""
        stats = LLMRunStats()
        stats.record_failed_item("vllm", "a")
        stats.record_failed_item("vllm", "b")
        stats.record_failed_item("ollama", "c")
        stats.record_retry_exhausted("vllm", "a")

        summary = stats.summary()

        assert summary["failed_items"] == {"vllm": ["a", "b"], "ollama": ["c"]}
        assert summary["retry_exhausted_items"] == {"vllm": ["a"]}

    def test_cache_hits_and_llm_calls_both_count_as_calls(self):
        """``total_calls`` is the denominator of every rate in the file."""
        stats = LLMRunStats()
        stats.record_cache_hit()
        stats.record_llm_call()
        stats.record_llm_call()

        summary = stats.summary()

        assert (summary["cache_hits"], summary["llm_calls"]) == (1, 2)
        assert summary["total_calls"] == 3

    def test_diagnostics_stay_out_of_the_summary(self):
        """They hold every raw answer of the run and go to their own file;
        folding them in would put the run inside its own statistics."""
        stats = LLMRunStats()
        stats.record_parse_diagnostic("binary", object())

        assert "parse_diagnostics" not in stats.summary()


class TestReset:
    def test_reset_clears_every_field(self):
        """One model instance serves several stages. A field ``reset()``
        forgets carries the previous stage's numbers into the next stage's
        file, where nothing marks them as stale."""
        stats = LLMRunStats()
        stats.record_cache_hit()
        stats.record_llm_call()
        stats.record_retry()
        stats.record_parse_success("binary")
        stats.record_parse_failure("binary")
        stats.record_parse_diagnostic("binary", object())
        stats.record_error("fake:timeout")
        stats.record_failed_item("fake", "i1")
        stats.record_retry_exhausted("fake", "i1")

        stats.reset()

        assert stats.summary() == LLMRunStats().summary()
        assert dict(stats.parse_diagnostics) == {}

    def test_recording_still_works_after_a_reset(self):
        """The cleared containers must stay the same kind of container."""
        stats = LLMRunStats()
        stats.record_parse_success("binary")
        stats.reset()

        stats.record_parse_success("binary")

        assert stats.summary()["per_parser"] == {"binary": {"success": 1}}


class TestBatchSummary:
    def test_counts_successes_failures_and_cache_hits(self):
        results = [
            _result(),
            _result(source="cache"),
            _result(success=False, value=None, error_type="timeout"),
            _result(success=False, value=None, error_type="timeout"),
            _result(success=False, value=None, error_type="parse_failure"),
        ]

        assert LLMCallResult.summarize(results) == {
            "total": 5,
            "success": 2,
            "failed": 3,
            "cache_hits": 1,
            "errors": {"timeout": 2, "parse_failure": 1},
        }

    def test_success_and_failure_always_add_up_to_the_total(self):
        """``failed`` is derived, not counted — nothing can fall between."""
        summary = LLMCallResult.summarize([_result(), _result(success=False)])

        assert summary["success"] + summary["failed"] == summary["total"]

    def test_a_recovered_error_does_not_enter_the_histogram(self):
        """The histogram answers 'why did items fail', not 'what went wrong
        on the way' — an item that succeeded on a retry is not a failure."""
        summary = LLMCallResult.summarize([_result(error_type="timeout")])

        assert summary["errors"] == {}

    def test_a_failure_without_a_type_is_counted_but_not_classified(self):
        summary = LLMCallResult.summarize([_result(success=False, error_type=None)])

        assert summary["failed"] == 1
        assert summary["errors"] == {}

    def test_an_empty_batch_summarizes_to_zeros(self):
        assert LLMCallResult.summarize([]) == {
            "total": 0,
            "success": 0,
            "failed": 0,
            "cache_hits": 0,
            "errors": {},
        }
