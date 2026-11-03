"""The labkit backend: calls go through the bound labkit cache, a replay miss stops the run.

labkit's cache needs a labkit run; these tests bind a stand-in with the
same ``complete`` signature. The real cache runs in fallacy-detection's
area experiments/02-pilot-bigbench-zs-raw (step 4b of the stagekit plan).
"""

from types import SimpleNamespace

import pytest
from labkit.run import llm, llm_cache

from stagekit.models.backends.factory import build_backend
from stagekit.models.backends.labkit_backend import LabkitBackend, answers_through
from stagekit.models.llm_cache import CacheCorruptionError, CacheMiss
from tests.llm_harness import HarnessModel, meta_cfg, model_cfg


def _response(text):
    return SimpleNamespace(text=text, thinking="trace")


class FakeCache:
    """labkit's ``Cache.complete`` as the backend calls it, answering or raising."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def complete(self, alias, prompt, *, temperature=None):
        self.calls.append((alias, prompt, temperature))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(response=_response(f"answer to {prompt}"), origin=None)


def _backend():
    return build_backend(model_cfg(backend="labkit", model_name="llama3.2", temperature=0.3))


def test_a_call_goes_through_the_bound_cache():
    cache = FakeCache()
    backend = _backend()
    with answers_through(cache):
        answer = backend.generate("p")
    assert isinstance(backend, LabkitBackend)
    assert (answer.raw, answer.thinking) == ("answer to p", "trace")
    assert cache.calls == [("llama3.2", "p", 0.3)]


def test_without_a_binding_the_model_is_asked_uncached(monkeypatch):
    calls = []
    monkeypatch.setattr(
        llm, "complete_response",
        lambda alias, prompt, *, temperature=None: calls.append((alias, prompt, temperature)) or _response("x"),
    )
    assert _backend().generate("p").raw == "x"
    assert calls == [("llama3.2", "p", 0.3)]


def test_the_binding_ends_with_the_block_and_is_one_at_a_time(monkeypatch):
    monkeypatch.setattr(llm, "complete_response", lambda *a, **k: _response("uncached"))
    with answers_through(FakeCache()):
        with pytest.raises(RuntimeError, match="bound already"):
            with answers_through(FakeCache()):
                pass
    assert _backend().generate("p").raw == "uncached"


def test_a_replay_miss_stops_the_call_instead_of_being_retried():
    """labkit's CacheMiss becomes stagekit's, which the retry loop re-raises."""
    cache = FakeCache(error=llm_cache.CacheMiss("no stored answer"))
    model = HarnessModel(model_cfg(backend="labkit"), meta_cfg(max_retries=3), backend=_backend())
    with answers_through(cache), pytest.raises(CacheMiss, match="labkit replay"):
        model._execute_llm_call(
            prompt="p", parser_fields={}, max_retries=3, prompt_version="v1", item_id="i1",
        )
    assert len(cache.calls) == 1


def test_a_corrupt_entry_stops_too():
    cache = FakeCache(error=llm_cache.CacheCorruptionError("not an entry"))
    with answers_through(cache), pytest.raises(CacheCorruptionError, match="labkit cache"):
        _backend().generate("p")
