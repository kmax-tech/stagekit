"""Tests for LLMCache: singleton factory, read/write API, and race safety."""

import hashlib
import json
import threading
from pathlib import Path

import pytest


from stagekit.models.llm_cache import CacheCorruptionError, LLMCache


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

# Default kwargs shared across tests to reduce boilerplate.
SLOT_DEFAULTS = dict(
    dataset_id="ds",
    model_id="model",
    parser_id="parser",
    prompt_version="v1",
)


def _make_cache(tmp_path: Path) -> LLMCache:
    """Return an LLMCache via the normative .get() factory.

    Each test gets a unique tmp_path, so .get() always creates a fresh
    instance — no cross-test pollution despite the singleton registry.
    """
    return LLMCache.get(tmp_path / "cache")


def _make_key(cache: LLMCache, prompt: str = "hello") -> str:
    return cache.make_key(
        backend="openai", prompt=prompt, model="gpt-4o",
        temperature=0.0, parser_id="parser", prompt_version="v1",
    )


# ------------------------------------------------------------------
# Singleton factory
# ------------------------------------------------------------------


class TestSingletonFactory:
    """LLMCache.get() returns the same instance for the same resolved path."""

    def setup_method(self):
        # Clear class-level singleton registry between tests
        LLMCache._instances.clear()

    def test_same_path_returns_same_instance(self, tmp_path):
        a = LLMCache.get(tmp_path / "cache")
        b = LLMCache.get(tmp_path / "cache")
        assert a is b

    def test_different_paths_return_different_instances(self, tmp_path):
        a = LLMCache.get(tmp_path / "cache_a")
        b = LLMCache.get(tmp_path / "cache_b")
        assert a is not b

    def test_resolves_symlinks(self, tmp_path):
        real = tmp_path / "real_cache"
        real.mkdir()
        link = tmp_path / "link_cache"
        link.symlink_to(real)

        a = LLMCache.get(real)
        b = LLMCache.get(link)
        assert a is b

    def test_direct_constructor_warns(self, tmp_path, caplog):
        """Direct LLMCache() emits a warning nudging toward .get()."""
        import logging
        with caplog.at_level(logging.WARNING, logger="stagekit.models.llm_cache"):
            LLMCache(tmp_path / "cache")
        assert "LLMCache.get(cache_dir)" in caplog.text

    def test_shared_locks_across_callers(self, tmp_path):
        """Two .get() callers share the same lock objects."""
        a = LLMCache.get(tmp_path / "cache")
        b = LLMCache.get(tmp_path / "cache")

        lock_a = a._lock_for("some_shard.json")
        lock_b = b._lock_for("some_shard.json")
        assert lock_a is lock_b


# ------------------------------------------------------------------
# Basic read/write
# ------------------------------------------------------------------


class TestBasicReadWrite:
    """get_or_set, get_one, get_all round-trip correctly."""

    def test_write_then_read(self, tmp_path):
        cache = _make_cache(tmp_path)
        key = _make_key(cache)

        result, was_cached = cache.get_or_set(
            key=key, slot=0, prompt="hello", value={"label": "fallacy"},
            **SLOT_DEFAULTS,
        )

        assert result == {"label": "fallacy"}
        assert was_cached is False

        # Read back
        got = cache.get_one(key=key, slot=0, **SLOT_DEFAULTS)
        assert got == {"label": "fallacy"}

    def test_get_one_missing_returns_none(self, tmp_path):
        cache = _make_cache(tmp_path)
        key = _make_key(cache)
        assert cache.get_one(key=key, slot=0, **SLOT_DEFAULTS) is None

    def test_get_all_returns_all_slots(self, tmp_path):
        cache = _make_cache(tmp_path)
        key = _make_key(cache)

        for i in range(3):
            cache.get_or_set(
                key=key, slot=i, prompt="hello", value=f"answer_{i}",
                **SLOT_DEFAULTS,
            )

        all_answers = cache.get_all(key=key, **SLOT_DEFAULTS)
        assert all_answers == {"0": "answer_0", "1": "answer_1", "2": "answer_2"}

    def test_get_all_missing_returns_empty(self, tmp_path):
        cache = _make_cache(tmp_path)
        key = _make_key(cache)
        assert cache.get_all(key=key, **SLOT_DEFAULTS) == {}

    def test_different_keys_are_independent(self, tmp_path):
        cache = _make_cache(tmp_path)
        key_a = _make_key(cache, prompt="prompt_a")
        key_b = _make_key(cache, prompt="prompt_b")

        cache.get_or_set(key=key_a, slot=0, prompt="prompt_a", value="a", **SLOT_DEFAULTS)
        cache.get_or_set(key=key_b, slot=0, prompt="prompt_b", value="b", **SLOT_DEFAULTS)

        assert cache.get_one(key=key_a, slot=0, **SLOT_DEFAULTS) == "a"
        assert cache.get_one(key=key_b, slot=0, **SLOT_DEFAULTS) == "b"

    def test_raw_and_meta_persisted(self, tmp_path):
        cache = _make_cache(tmp_path)
        key = _make_key(cache)

        cache.get_or_set(
            key=key, slot=0, prompt="hello", value="answer",
            raw="raw LLM output", meta={"model_version": "2026-05"},
            **SLOT_DEFAULTS,
        )

        entry = cache.get_entry(key=key, **SLOT_DEFAULTS)
        assert entry is not None
        assert entry["raw"]["0"] == "raw LLM output"
        assert entry["meta"]["model_version"] == "2026-05"


# ------------------------------------------------------------------
# Prompt storage
# ------------------------------------------------------------------


class TestPromptStorage:
    """Prompt text is stored in _prompts.json and resolved by get_entry."""

    def test_prompt_stored_and_resolved(self, tmp_path):
        cache = _make_cache(tmp_path)
        key = _make_key(cache)

        cache.get_or_set(
            key=key, slot=0, prompt="Tell me about fallacies",
            value="answer", **SLOT_DEFAULTS,
        )

        entry = cache.get_entry(key=key, **SLOT_DEFAULTS)
        assert entry["prompt"] == "Tell me about fallacies"

    def test_prompt_deduplicated(self, tmp_path):
        """Same prompt across different keys is stored once."""
        cache = _make_cache(tmp_path)
        prompt = "shared prompt text"

        for i in range(3):
            key = _make_key(cache, prompt=f"unique_key_input_{i}")
            cache.get_or_set(
                key=key, slot=0, prompt=prompt, value=f"answer_{i}",
                **SLOT_DEFAULTS,
            )

        prompts_path = cache._prompts_path(**SLOT_DEFAULTS)
        prompts = json.loads(prompts_path.read_text())
        # Only one prompt hash entry despite 3 writes
        prompt_texts = list(prompts.values())
        assert prompt_texts.count(prompt) == 1


# ------------------------------------------------------------------
# get_or_set atomicity
# ------------------------------------------------------------------


class TestGetOrSetAtomicity:
    """get_or_set does not overwrite existing slots."""

    def test_second_write_returns_first_value(self, tmp_path):
        cache = _make_cache(tmp_path)
        key = _make_key(cache)

        v1, was_cached_1 = cache.get_or_set(
            key=key, slot=0, prompt="hello", value="first",
            **SLOT_DEFAULTS,
        )
        v2, was_cached_2 = cache.get_or_set(
            key=key, slot=0, prompt="hello", value="second",
            **SLOT_DEFAULTS,
        )

        assert v1 == "first"
        assert was_cached_1 is False
        assert v2 == "first"  # returns existing, not "second"
        assert was_cached_2 is True

    def test_existing_slot_not_overwritten_on_disk(self, tmp_path):
        cache = _make_cache(tmp_path)
        key = _make_key(cache)

        cache.get_or_set(key=key, slot=0, prompt="hello", value="first", **SLOT_DEFAULTS)
        cache.get_or_set(key=key, slot=0, prompt="hello", value="second", **SLOT_DEFAULTS)

        # Bypass singleton to get a fresh instance that reads from disk,
        # proving persistence is correct independent of in-memory state.
        cache2 = LLMCache(tmp_path / "cache", _factory=LLMCache._FACTORY_SENTINEL)
        assert cache2.get_one(key=key, slot=0, **SLOT_DEFAULTS) == "first"

    def test_different_slots_independent(self, tmp_path):
        """get_or_set on slot 0 does not block writing to slot 1."""
        cache = _make_cache(tmp_path)
        key = _make_key(cache)

        cache.get_or_set(key=key, slot=0, prompt="hello", value="slot_0", **SLOT_DEFAULTS)
        v1, was_cached = cache.get_or_set(
            key=key, slot=1, prompt="hello", value="slot_1", **SLOT_DEFAULTS,
        )

        assert v1 == "slot_1"
        assert was_cached is False
        assert cache.get_one(key=key, slot=0, **SLOT_DEFAULTS) == "slot_0"
        assert cache.get_one(key=key, slot=1, **SLOT_DEFAULTS) == "slot_1"


# ------------------------------------------------------------------
# Concurrent race simulation
# ------------------------------------------------------------------


class TestConcurrentRace:
    """Multiple threads racing on the same slot: first writer wins."""

    def test_concurrent_writers_first_wins(self, tmp_path):
        """N threads write to the same slot — all must agree on one value."""
        cache = _make_cache(tmp_path)
        key = _make_key(cache)
        n_threads = 20

        barrier = threading.Barrier(n_threads)
        results = [None] * n_threads

        def worker(thread_id):
            barrier.wait()  # all threads start simultaneously
            value, was_cached = cache.get_or_set(
                key=key, slot=0, prompt="hello",
                value=f"thread_{thread_id}",
                **SLOT_DEFAULTS,
            )
            results[thread_id] = value

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads must have received the same value
        unique_values = set(results)
        assert len(unique_values) == 1, f"Expected 1 unique value, got {unique_values}"

        # The value in cache must match what all threads received
        winner = unique_values.pop()
        on_disk = cache.get_one(key=key, slot=0, **SLOT_DEFAULTS)
        assert on_disk == winner

    def test_concurrent_different_slots_all_written(self, tmp_path):
        """Threads writing to different slots on the same key all succeed."""
        cache = _make_cache(tmp_path)
        key = _make_key(cache)
        n_threads = 10

        barrier = threading.Barrier(n_threads)

        def worker(slot):
            barrier.wait()
            cache.get_or_set(
                key=key, slot=slot, prompt="hello",
                value=f"slot_{slot}",
                **SLOT_DEFAULTS,
            )

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_answers = cache.get_all(key=key, **SLOT_DEFAULTS)
        assert len(all_answers) == n_threads
        for i in range(n_threads):
            assert all_answers[str(i)] == f"slot_{i}"

    def test_concurrent_different_keys_no_interference(self, tmp_path):
        """Threads writing to different keys don't interfere."""
        cache = _make_cache(tmp_path)
        n_threads = 10

        keys = [_make_key(cache, prompt=f"prompt_{i}") for i in range(n_threads)]
        barrier = threading.Barrier(n_threads)

        def worker(idx):
            barrier.wait()
            cache.get_or_set(
                key=keys[idx], slot=0, prompt=f"prompt_{idx}",
                value=f"value_{idx}",
                **SLOT_DEFAULTS,
            )

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(n_threads):
            assert cache.get_one(key=keys[i], slot=0, **SLOT_DEFAULTS) == f"value_{i}"

    def test_race_was_cached_flag_consistent(self, tmp_path):
        """Exactly one thread gets was_cached=False, all others get True."""
        cache = _make_cache(tmp_path)
        key = _make_key(cache)
        n_threads = 20

        barrier = threading.Barrier(n_threads)
        flags = [None] * n_threads

        def worker(thread_id):
            barrier.wait()
            _, was_cached = cache.get_or_set(
                key=key, slot=0, prompt="hello",
                value=f"thread_{thread_id}",
                **SLOT_DEFAULTS,
            )
            flags[thread_id] = was_cached

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        writers = [i for i, f in enumerate(flags) if f is False]
        readers = [i for i, f in enumerate(flags) if f is True]

        assert len(writers) == 1, f"Expected exactly 1 writer, got {len(writers)}: {writers}"
        assert len(readers) == n_threads - 1


# ------------------------------------------------------------------
# Shard resilience
# ------------------------------------------------------------------


class TestShardResilience:
    """Missing files are fine; corrupt files stop the run (fail loud).

    The old contract ("recover gracefully") was the bug: reading a
    corrupt shard as empty let the next write replace the file and
    silently destroy every entry it held. See TestCacheCorruption for
    the full corruption contract.
    """

    def test_missing_shard_returns_none(self, tmp_path):
        cache = _make_cache(tmp_path)
        key = _make_key(cache)

        assert cache.get_one(key=key, slot=0, **SLOT_DEFAULTS) is None

    def test_corrupt_shard_raises_instead_of_recovering(self, tmp_path):
        cache = _make_cache(tmp_path)
        key = _make_key(cache)

        # Write a valid entry, then corrupt the shard
        cache.get_or_set(key=key, slot=0, prompt="hello", value="ok", **SLOT_DEFAULTS)
        shard_path = cache._path(key=key, **SLOT_DEFAULTS)
        shard_path.write_text("not valid json{{{")

        with pytest.raises(CacheCorruptionError):
            cache.get_one(key=key, slot=0, **SLOT_DEFAULTS)


# ------------------------------------------------------------------
# Cache key computation
# ------------------------------------------------------------------


class TestCacheKey:
    """make_key is deterministic and sensitive to all inputs."""

    def test_deterministic(self, tmp_path):
        cache = _make_cache(tmp_path)
        k1 = _make_key(cache, prompt="test")
        k2 = _make_key(cache, prompt="test")
        assert k1 == k2

    def test_different_prompts_different_keys(self, tmp_path):
        cache = _make_cache(tmp_path)
        k1 = _make_key(cache, prompt="alpha")
        k2 = _make_key(cache, prompt="beta")
        assert k1 != k2

    def test_sensitive_to_parser_id(self, tmp_path):
        cache = _make_cache(tmp_path)
        k1 = cache.make_key(
            backend="openai", prompt="x", model="m",
            temperature=0.0, parser_id="parser_a", prompt_version="v1",
        )
        k2 = cache.make_key(
            backend="openai", prompt="x", model="m",
            temperature=0.0, parser_id="parser_b", prompt_version="v1",
        )
        assert k1 != k2

    def test_sensitive_to_temperature(self, tmp_path):
        cache = _make_cache(tmp_path)
        k1 = cache.make_key(
            backend="openai", prompt="x", model="m",
            temperature=0.0, parser_id="p", prompt_version="v1",
        )
        k2 = cache.make_key(
            backend="openai", prompt="x", model="m",
            temperature=0.7, parser_id="p", prompt_version="v1",
        )
        assert k1 != k2


# ------------------------------------------------------------------
# Cross-process locking (flock on sidecar .lock files)
# ------------------------------------------------------------------

_WORKER_SCRIPT = """
import sys
from pathlib import Path
from stagekit.models.llm_cache import LLMCache

cache_dir, start = Path(sys.argv[1]), int(sys.argv[2])
cache = LLMCache.get(cache_dir)
for i in range(start, start + 40):
    key = "aa" + format(i, "062x")  # same 2-hex prefix -> same shard file
    cache.get_or_set(
        key=key, slot=1, prompt=f"prompt {i}",
        dataset_id="ds", model_id="model", parser_id="parser",
        prompt_version="v1", value={"answer": i},
    )
"""


class TestCrossProcessLocking:
    """Two processes hammering the SAME shard must not lose updates.

    Before the flock, both processes read-modify-wrote the shard JSON
    concurrently and the last writer silently dropped the other's
    entries (lost update). With the sidecar flock, all 80 entries from
    both processes must survive.
    """

    def test_concurrent_processes_lose_no_updates(self, tmp_path):
        import subprocess
        import sys

        project_root = Path(__file__).resolve().parent.parent
        cache_dir = tmp_path / "cache"

        procs = [
            subprocess.Popen(
                [sys.executable, "-c", _WORKER_SCRIPT, str(cache_dir), str(start)],
                cwd=project_root,
            )
            for start in (0, 100_000)
        ]
        for proc in procs:
            assert proc.wait(timeout=120) == 0

        shard_path = cache_dir / "ds" / "model" / "parser" / "v1" / "aa.json"
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        assert len(shard["entries"]) == 80  # 40 per process, none lost

        # prompt store shares the same locking discipline
        prompts_path = shard_path.parent / "_prompts.json"
        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        assert len(prompts) == 80

        # sidecar lock files exist but are not cache data
        assert (shard_path.parent / "aa.json.lock").exists()


# ------------------------------------------------------------------
# Key payload: backend-specific settings must not re-key stored runs
# ------------------------------------------------------------------
class TestBackendParamsInKey:

    def test_omitted_when_empty_keeps_existing_keys_valid(self, tmp_path):
        """The payload must stay byte-identical for the four old backends.

        Every entry written before backend_params existed was keyed on this
        exact six-field payload; if the field were always present, the whole
        stored cache would miss at once.
        """
        cache = _make_cache(tmp_path)
        expected = hashlib.sha256(json.dumps({
            "backend": "openai", "prompt": "hello", "model": "gpt-4o",
            "temperature": 0.0, "parser": "parser", "prompt_version": "v1",
        }, sort_keys=True).encode("utf-8")).hexdigest()

        assert _make_key(cache) == expected
        assert cache.make_key(
            backend="openai", prompt="hello", model="gpt-4o", temperature=0.0,
            parser_id="parser", prompt_version="v1", backend_params={},
        ) == expected

    def test_present_params_separate_the_key(self, tmp_path):
        """Two settings that change the answer must not share an entry."""
        cache = _make_cache(tmp_path)
        base = dict(backend="vllm", prompt="hello", model="m",
                    temperature=0.0, parser_id="parser", prompt_version="v1")

        assert (cache.make_key(**base, backend_params={"max_tokens": 512})
                != cache.make_key(**base, backend_params={"max_tokens": 4096}))
        assert (cache.make_key(**base, backend_params={"reasoning": True})
                != cache.make_key(**base))


class TestTemperatureKeyedOnlyWhenConsumed:
    """Cache key and run identity must agree on temperature (audit B1).

    repro_hash._model_canon drops temperature for backends that never
    send it (CLI backends); the cache key must do the same, or two runs
    with the SAME repro_hash get disjoint keys and pay every call twice.
    """

    def _key_for(self, cache, backend, temperature):
        """Build the key exactly as LLMBaseModel._build_cache_key does."""
        from stagekit.models.base import LLMBaseModel
        from stagekit.config.classes import ModelConfig

        model = object.__new__(LLMBaseModel)
        model.cache = cache
        model.model_cfg = ModelConfig(
            model_type="llm", backend=backend, model_name="m",
            temperature=temperature,
        )
        key, _ = model._build_cache_key(
            backend_name=backend, prompt="hello", parser_id="parser",
            prompt_version="v1", dataset_id="ds",
        )
        return key

    def test_cli_backend_ignores_temperature(self, tmp_path):
        cache = _make_cache(tmp_path)
        assert (self._key_for(cache, "claude_cli", 1.0)
                == self._key_for(cache, "claude_cli", 0.5))

    def test_consuming_backend_keys_on_temperature(self, tmp_path):
        cache = _make_cache(tmp_path)
        assert (self._key_for(cache, "openai", 1.0)
                != self._key_for(cache, "openai", 0.5))

    def test_consuming_backend_key_is_byte_identical_to_legacy(self, tmp_path):
        """Stored keys of the consuming backends must keep resolving."""
        cache = _make_cache(tmp_path)
        legacy = hashlib.sha256(json.dumps({
            "backend": "openai", "prompt": "hello", "model": "m",
            "temperature": 0.5, "parser": "parser", "prompt_version": "v1",
        }, sort_keys=True).encode("utf-8")).hexdigest()

        assert self._key_for(cache, "openai", 0.5) == legacy


# ------------------------------------------------------------------
# Reasoning trace storage
# ------------------------------------------------------------------
class TestThinkingSlot:

    def test_trace_stored_beside_raw_as_string(self, tmp_path):
        cache = _make_cache(tmp_path)
        key = _make_key(cache)

        cache.get_or_set(key=key, prompt="hello", slot=0, value={"a": 1},
                         raw="the answer", thinking="the trace", **SLOT_DEFAULTS)

        entry = cache.get_entry(key=key, **SLOT_DEFAULTS)
        assert entry["thinking"]["0"] == "the trace"
        assert entry["raw"]["0"] == "the answer"  # still a plain string

    def test_writes_into_an_entry_that_predates_the_field(self, tmp_path):
        """Regression: sub-dicts are seeded for NEW keys only.

        ``entries.setdefault(key, {...})`` does not touch an entry that is
        already there, so an entry written by an older version — or merged
        in by hand — has no ``thinking`` (nor ``raw``/``meta``) dict and a
        direct subscript would raise KeyError mid-run.
        """
        cache = _make_cache(tmp_path)
        key = _make_key(cache)
        path = cache._path(key=key, **SLOT_DEFAULTS)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"entries": {key: {
            "prompt_hash": "x", "answers": {"0": {"old": True}},
        }}}), encoding="utf-8")

        value, was_cached = cache.get_or_set(
            key=key, prompt="hello", slot=1, value={"new": True},
            raw="answer", thinking="trace", **SLOT_DEFAULTS,
        )

        assert was_cached is False
        entry = cache.get_entry(key=key, **SLOT_DEFAULTS)
        assert entry["answers"]["0"] == {"old": True}  # untouched
        assert entry["thinking"]["1"] == "trace"


# ------------------------------------------------------------------
# Corruption: never destroy a shard that still exists on disk
# ------------------------------------------------------------------
class TestCacheCorruption:
    """A corrupt cache file must stop the run, not be replaced.

    Reading a corrupt shard as empty would let the next write REPLACE
    the file — permanent silent loss of every entry it held.
    """

    def _corrupt_shard(self, cache: LLMCache, key: str) -> Path:
        path = cache._path(key=key, **SLOT_DEFAULTS)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json", encoding="utf-8")
        return path

    def test_get_or_set_raises_and_leaves_file_untouched(self, tmp_path):
        cache = _make_cache(tmp_path)
        key = _make_key(cache)
        path = self._corrupt_shard(cache, key)

        with pytest.raises(CacheCorruptionError, match="not valid JSON"):
            cache.get_or_set(
                key=key, prompt="hello", slot=0, value={"a": 1},
                **SLOT_DEFAULTS,
            )
        assert path.read_text(encoding="utf-8") == "{not valid json"

    def test_get_one_raises(self, tmp_path):
        cache = _make_cache(tmp_path)
        key = _make_key(cache)
        self._corrupt_shard(cache, key)

        with pytest.raises(CacheCorruptionError):
            cache.get_one(key=key, slot=0, **SLOT_DEFAULTS)

    def test_shard_without_entries_dict_raises(self, tmp_path):
        cache = _make_cache(tmp_path)
        key = _make_key(cache)
        path = cache._path(key=key, **SLOT_DEFAULTS)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"entries": "wrong-type"}), encoding="utf-8")

        with pytest.raises(CacheCorruptionError, match="no 'entries' dict"):
            cache.get_one(key=key, slot=0, **SLOT_DEFAULTS)

    def test_get_entry_stays_tolerant(self, tmp_path):
        """The read-only audit path warns and returns None instead."""
        cache = _make_cache(tmp_path)
        key = _make_key(cache)
        self._corrupt_shard(cache, key)

        assert cache.get_entry(key=key, **SLOT_DEFAULTS) is None

    def test_corrupt_prompt_store_raises_instead_of_reset(self, tmp_path):
        cache = _make_cache(tmp_path)
        key = _make_key(cache)
        prompts_path = cache._prompts_path(**SLOT_DEFAULTS)
        prompts_path.parent.mkdir(parents=True, exist_ok=True)
        prompts_path.write_text("{broken", encoding="utf-8")

        with pytest.raises(CacheCorruptionError, match="Prompt store"):
            cache.get_or_set(
                key=key, prompt="hello", slot=0, value={"a": 1},
                **SLOT_DEFAULTS,
            )
        assert prompts_path.read_text(encoding="utf-8") == "{broken"
