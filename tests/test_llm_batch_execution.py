"""The batch entry points: how wide they run, in which order they answer.

``run_llm_tasks_with`` is the only place in the model where the caller's
list of tasks becomes a list of results, and three of its decisions are
invisible from the outside: how many workers it actually opens, whether a
result keeps its task's position when the workers finish out of order, and
whether a failing worker aborts the batch or leaves a hole in the list.

All three are the kind of thing that keeps working in a small pilot and
goes wrong on a real run — a backend cap ignored on a 50-worker config, a
misaligned result list that silently pairs answers with the wrong items.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock

import pytest

import stagekit.models.base as base_module
from stagekit.models.backends.backends_base import Answer
from stagekit.prompts.base import PromptTemplate
from tests.llm_harness import FakeBackend, FakeParser, HarnessModel, meta_cfg, model_cfg

_TEMPLATE = PromptTemplate(
    name="t",
    template_str="Q: $question",
    required_fields=["question"],
)


def _tasks(n: int) -> list[dict]:
    """*n* well-formed tasks, numbered so the answer identifies its task."""
    return [{"question": str(i), "item_id": f"i{i}"} for i in range(n)]


def _model(*, workers: int, backend: FakeBackend, parser=None) -> HarnessModel:
    """A model wired for batch runs: no cache, no retries, fast failures."""
    return HarnessModel(
        model_cfg(),
        meta_cfg(max_workers=workers, max_retries=0),
        backend=backend,
        parser=parser if parser is not None else FakeParser(),
        template=_TEMPLATE,
    )


@pytest.fixture
def executor_widths(monkeypatch) -> list[int]:
    """Record the worker count of every thread pool the batch API opens.

    Returns:
        One entry per pool, in construction order; empty when the serial
        path was taken.
    """
    widths: list[int] = []

    class _RecordingExecutor(ThreadPoolExecutor):
        def __init__(self, max_workers=None, **kwargs):
            widths.append(max_workers)
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(base_module, "ThreadPoolExecutor", _RecordingExecutor)
    return widths


# ------------------------------------------------------------------
# How wide the batch runs
# ------------------------------------------------------------------
class TestWorkerCount:
    """``max_workers`` is a request; the backend's cap is the limit."""

    def _run(self, executor_widths, *, workers: int, cap):
        model = _model(workers=workers, backend=FakeBackend(max_safe_workers=cap))
        model.run_llm_tasks_with(
            template=_TEMPLATE,
            parser=FakeParser(),
            tasks=_tasks(4),
            desc="d",
        )
        return executor_widths

    def test_backend_cap_wins_over_the_configured_workers(self, executor_widths):
        """A local server that tolerates 3 concurrent requests must not be hit
        with the config's 50 just because the config says so."""
        assert self._run(executor_widths, workers=50, cap=3) == [3]

    def test_configured_workers_win_when_they_are_lower(self, executor_widths):
        """The cap raises nothing — it is a ceiling, not a target."""
        assert self._run(executor_widths, workers=2, cap=8) == [2]

    def test_a_backend_without_a_cap_gets_the_configured_workers(
        self,
        executor_widths,
    ):
        assert self._run(executor_widths, workers=4, cap=None) == [4]

    def test_one_worker_runs_without_a_pool(self, executor_widths):
        """Nothing to parallelise; a pool would only add a thread hop and
        wrap every exception in a future."""
        assert self._run(executor_widths, workers=1, cap=None) == []

    def test_a_backend_cap_of_one_also_runs_serially(self, executor_widths):
        """The cap has to be applied before the serial decision — a CLI
        backend configured with 50 workers must still run one at a time."""
        assert self._run(executor_widths, workers=50, cap=1) == []


# ------------------------------------------------------------------
# Result order
# ------------------------------------------------------------------
class _ReverseCompletionBackend(FakeBackend):
    """Forces the workers to finish in reverse submission order.

    Every call meets the others at a barrier, then waits for its successor's
    event before answering, so task ``n-1`` completes first and task ``0``
    last. Deterministic where staggered sleeps would be a guess.

    Args:
        count: Number of tasks — all of them must run concurrently, so the
            batch needs at least this many workers.
    """

    def __init__(self, count: int, **kwargs) -> None:
        super().__init__(max_safe_workers=count, **kwargs)
        self._barrier = Barrier(count)
        self._events = [Event() for _ in range(count)]
        self._order_lock = Lock()
        self.finished: list[int] = []

    def generate(self, prompt: str) -> Answer:
        """Answer with the task number, after every later task has answered.

        Raises:
            BrokenBarrierError: If not all tasks are in flight within 10 s —
                which means the batch did not run as wide as the test asked.
        """
        index = int(prompt.rsplit(" ", 1)[1])
        self._barrier.wait(timeout=10)
        if index + 1 < len(self._events):
            self._events[index + 1].wait(timeout=10)
        with self._order_lock:
            self.finished.append(index)
        self._events[index].set()
        return Answer(raw=str(index))


class TestResultOrder:
    def test_results_follow_task_order_not_completion_order(self):
        """The list is indexed by position, not by arrival. If it ever went
        back to appending, every answer would be paired with the wrong item
        and nothing would crash."""
        count = 6
        backend = _ReverseCompletionBackend(count)
        model = _model(workers=count, backend=backend)

        results = model.run_llm_tasks_with(
            template=_TEMPLATE,
            parser=FakeParser(),
            tasks=_tasks(count),
            desc="d",
        )

        # The premise: the workers really did finish backwards.
        assert backend.finished == list(reversed(range(count)))
        assert [r.value["text"] for r in results] == [str(i) for i in range(count)]

    @pytest.mark.parametrize("workers", [1, 4])
    def test_every_task_is_run_exactly_once(self, workers: int):
        """Neither path may drop or duplicate a task."""
        backend = FakeBackend(max_safe_workers=None)
        model = _model(workers=workers, backend=backend)

        results = model.run_llm_tasks_with(
            template=_TEMPLATE,
            parser=FakeParser(),
            tasks=_tasks(7),
            desc="d",
        )

        assert sorted(backend.prompts) == sorted(f"Q: {i}" for i in range(7))
        assert len(results) == 7
        assert all(r.success for r in results)

    @pytest.mark.parametrize("workers", [1, 4])
    def test_an_empty_batch_returns_an_empty_list(self, workers: int):
        model = _model(workers=workers, backend=FakeBackend(max_safe_workers=None))

        assert (
            model.run_llm_tasks_with(
                template=_TEMPLATE,
                parser=FakeParser(),
                tasks=[],
                desc="d",
            )
            == []
        )


# ------------------------------------------------------------------
# Failures inside a worker
# ------------------------------------------------------------------
class TestFailFast:
    @pytest.mark.parametrize("workers", [1, 4])
    def test_a_malformed_task_aborts_the_whole_batch(self, workers: int):
        """Deliberate: the result list is pre-sized, so a swallowed worker
        exception would leave a ``None`` that downstream stages cannot tell
        apart from a real result."""
        tasks = _tasks(4)
        del tasks[2]["question"]
        model = _model(workers=workers, backend=FakeBackend(max_safe_workers=None))

        with pytest.raises(ValueError, match="Missing required fields"):
            model.run_llm_tasks_with(
                template=_TEMPLATE,
                parser=FakeParser(),
                tasks=tasks,
                desc="d",
            )

    @pytest.mark.parametrize("workers", [1, 4])
    def test_a_backend_error_does_not_abort_the_batch(self, workers: int):
        """Contrast case: what the retry loop already handles stays inside it
        and comes back as a failed result, not as an exception."""
        backend = FakeBackend(error=ValueError("endpoint down"), max_safe_workers=None)
        model = _model(workers=workers, backend=backend)

        results = model.run_llm_tasks_with(
            template=_TEMPLATE,
            parser=FakeParser(),
            tasks=_tasks(3),
            desc="d",
        )

        assert len(results) == 3
        assert not any(r.success for r in results)
        assert {r.error_type for r in results} == {"ValueError"}


# ------------------------------------------------------------------
# Which parser and template the batch uses
# ------------------------------------------------------------------
class TestStatelessBatch:
    def test_the_explicit_parser_is_used_and_the_models_own_is_untouched(self):
        """The documented promise of the stateless API — a workflow stage
        running three prompts through one model must not have its second
        prompt's parser leak into the third."""
        own = FakeParser(cache_id="own")
        explicit = FakeParser(cache_id="explicit")
        model = _model(
            workers=2, backend=FakeBackend(max_safe_workers=None), parser=own
        )
        model.template = None

        model.run_llm_tasks_with(
            template=_TEMPLATE,
            parser=explicit,
            tasks=_tasks(2),
            desc="d",
        )

        assert explicit.calls == 2
        assert own.calls == 0
        assert model.parser is own
        assert model.template is None
        assert set(model.llm_stats.per_parser) == {"explicit"}

    def test_run_llm_tasks_uses_the_models_own_parser_and_template(self):
        own = FakeParser(cache_id="own")
        model = _model(
            workers=2, backend=FakeBackend(max_safe_workers=None), parser=own
        )

        results = model.run_llm_tasks(tasks=_tasks(2), desc="d")

        assert own.calls == 2
        assert [r.value["text"] for r in results] == ["fine", "fine"]


class TestBatchGuards:
    """The stateful batch API cannot fall back to anything — it must say so."""

    def test_without_a_parser(self):
        model = _model(workers=2, backend=FakeBackend(), parser=FakeParser())
        model.parser = None

        with pytest.raises(RuntimeError, match="Parser not initialized"):
            model.run_llm_tasks(tasks=_tasks(1), desc="d")

    def test_without_a_template(self):
        model = _model(workers=2, backend=FakeBackend())
        model.template = None

        with pytest.raises(RuntimeError, match="Prompt template not initialized"):
            model.run_llm_tasks(tasks=_tasks(1), desc="d")

    def test_without_meta_cfg(self):
        model = HarnessModel(model_cfg(), None, template=_TEMPLATE)

        with pytest.raises(RuntimeError, match="meta_cfg not initialized"):
            model.run_llm_tasks_with(
                template=_TEMPLATE,
                parser=FakeParser(),
                tasks=_tasks(1),
                desc="d",
            )
