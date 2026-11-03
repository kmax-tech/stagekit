# stagekit/models/backends/labkit_backend.py
"""Answers through labkit: the model is a `models.toml` alias, the call
``labkit.run.llm.complete_response``, the cache labkit's cache of raw
answers (``labkit.run.llm_cache.Cache``).

A labkit cache belongs to one labkit run (it writes the run's
``llm-calls.jsonl`` and fixes its mode — normal, replay, repeat — when the
run opens it). The caller opens it and binds it around the pipeline with
:func:`answers_through`; every call of this backend then goes through it.
Without a binding, a call goes to the model uncached.

Parsing stays in stagekit (``LLMBaseModel``): the cache holds raw answers,
so a corrected parser applies to them at once. stagekit's own cache of
parsed results is the wrong second store here — run with
``meta.enable_cache=false``. A retry after a parse failure reads the same
stored answer again (labkit keys no attempt), so ``meta.max_retries=0``
says what happens; labkit retries transport errors itself.

labkit is an optional dependency (``pip install stagekit[labkit]``),
imported when a call is made.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

from stagekit.models.backends.backends_base import Answer, LLMBackend
from stagekit.models.backends.factory import register_backend
from stagekit.models.llm_cache import CacheCorruptionError, CacheMiss

if TYPE_CHECKING:
    from labkit.run.llm_cache import Cache

#: the labkit cache bound by `answers_through`, one per process
_bound: dict[str, Cache] = {}
_bound_lock = threading.Lock()


@contextmanager
def answers_through(cache: Cache) -> Generator[None, None, None]:
    """Route every call of the labkit backend through CACHE while the block runs.

    Raises:
        RuntimeError: When another cache is bound already — two caches in
            one process would mix two runs' records.
    """
    with _bound_lock:
        if "cache" in _bound:
            raise RuntimeError("a labkit cache is bound already; one run at a time")
        _bound["cache"] = cache
    try:
        yield
    finally:
        with _bound_lock:
            del _bound["cache"]


@register_backend("labkit")
class LabkitBackend(LLMBackend):
    """``model_name`` is the `models.toml` alias; temperature is sent as given.

    One request at a time, as stagekit's ollama backend with one server: a
    served model answering parallel requests can answer differently
    (server-side batching), and the pinned runs were made that way.
    """

    name = "labkit"
    max_safe_workers = 1

    def generate(self, prompt: str) -> Answer:
        """One completion through the bound cache, or uncached without one.

        Raises:
            CacheMiss: A replay found no stored answer (labkit's CacheMiss,
                re-raised as stagekit's so the run stops instead of
                retrying).
            CacheCorruptionError: A stored entry is not one.
        """
        from labkit.run import llm, llm_cache

        cache = _bound.get("cache")
        try:
            if cache is None:
                response = llm.complete_response(self.model_name, prompt, temperature=self.temperature)
            else:
                response = cache.complete(self.model_name, prompt, temperature=self.temperature).response
        except llm_cache.CacheMiss as exc:
            raise CacheMiss(f"labkit replay: {exc}") from exc
        except llm_cache.CacheCorruptionError as exc:
            raise CacheCorruptionError(f"labkit cache: {exc}") from exc
        return Answer(raw=response.text, thinking=response.thinking)
