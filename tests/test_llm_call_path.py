"""The call path from backend answer to cache entry and diagnostics.

Covers the seam the unit tests around LLMCache and the backends leave open:
those check `make_key`/`get_or_set` with hand-passed values, and the backends
check their own return shape. What neither checks is that the model actually
wires one into the other — which is where the reasoning trace would go
missing, and where two instruments would end up sharing cache entries.

The second half of the file is about what happens when that wiring is used
wrongly: a lost write race, a cache without a dataset namespace, a model
whose parser or template was never resolved.
"""

import pytest

from stagekit.models.backends.backends_base import Answer
from stagekit.models.base import _LLMCallContext
from stagekit.prompts.base import PromptTemplate
from tests.llm_harness import (
    SLOT_DEFAULTS,
    FakeBackend,
    FakeParser,
    HarnessModel,
    meta_cfg,
    model_cfg,
)

_TEMPLATE = PromptTemplate(
    name="t",
    template_str="Q: $question",
    required_fields=["question"],
)


def _ctx(**overrides) -> _LLMCallContext:
    """A call context pointing at the default cache slot, with *overrides*."""
    base = dict(
        prompt="p",
        parser_fields={},
        max_retries=0,
        prompt_version="v1",
        item_id="i1",
        template_str=None,
        run_metadata=None,
        parser_id="binary",
        backend_name="fake",
        requested_nbr=1,
        cache_key="k" * 8,
        cache_dataset_id="ds",
    )
    base.update(overrides)
    return _LLMCallContext(**base)


# ------------------------------------------------------------------
# backend_params: the cache key follows the capability declaration
# ------------------------------------------------------------------
class TestBackendParamsWiring:
    def _params(self, backend_name, **cfg_kwargs):
        model = HarnessModel(model_cfg(**cfg_kwargs))
        return model._backend_params(backend_name)

    def test_legacy_backends_contribute_nothing(self):
        """Empty for all four old backends — that is what keeps stored keys valid."""
        for backend in ("openai", "ollama", "claude_cli", "codex_cli"):
            assert self._params(backend, max_tokens=4096, reasoning=True) == {}, backend

    def test_vllm_contributes_the_fields_it_consumes(self):
        assert self._params("vllm", max_tokens=4096, reasoning=True) == {
            "max_tokens": 4096,
            "reasoning": True,
        }

    def test_temperature_stays_out(self):
        """It already has its own key in the payload; twice would be a second truth."""
        assert "temperature" not in self._params("vllm", max_tokens=512)

    def test_same_model_on_two_backends_gets_two_keys(self, tmp_path):
        """The instrument rule, at the level where it actually bites.

        vLLM serves fp16, ollama serves Q4 — same model name, different
        answers. They must not read each other's cache entries.
        """
        keys = []
        for backend in ("ollama", "vllm"):
            model = HarnessModel(
                model_cfg(backend=backend, max_tokens=4096, reasoning=True),
                meta_cfg(enable_cache=True),
                cache_dir=tmp_path / backend,
            )
            key, _ = model._build_cache_key(
                backend,
                "the prompt",
                "binary",
                "v1",
                "ds",
            )
            keys.append(key)
        assert keys[0] != keys[1]


# ------------------------------------------------------------------
# The trace survives from generate() to the stored entry
# ------------------------------------------------------------------
class TestTraceReachesTheCache:
    def _run(self, tmp_path, save_raw=True):
        model = HarnessModel(
            model_cfg(),
            meta_cfg(enable_cache=True, save_raw=save_raw),
            backend=FakeBackend(Answer(raw="the answer", thinking="the trace")),
            cache_dir=tmp_path,
        )
        result = model._execute_llm_call(
            prompt="p",
            parser_fields={},
            max_retries=0,
            prompt_version="v1",
            dataset_id="ds",
            item_id="i1",
        )
        key, _ = model._build_cache_key("fake", "p", "binary", "v1", "ds")
        entry = model.cache.get_entry(key=key, model_id="m", **SLOT_DEFAULTS)
        return result, entry, str(model.meta_cfg.prompt_nbr)

    def test_answer_and_trace_are_stored_apart(self, tmp_path):
        result, entry, slot = self._run(tmp_path)

        assert result.success is True
        assert entry["raw"][slot] == "the answer"
        assert entry["thinking"][slot] == "the trace"

    def test_save_raw_false_stores_neither(self, tmp_path):
        """The trace rides on the same switch as the raw answer, by design."""
        _, entry, _slot = self._run(tmp_path, save_raw=False)

        assert entry.get("raw", {}) == {}
        assert entry.get("thinking", {}) == {}

    def test_cache_hit_carries_neither(self, tmp_path):
        """Documented asymmetry — a hit returns the parsed value only."""
        backend = FakeBackend(Answer(raw="the answer", thinking="the trace"))
        model = HarnessModel(
            model_cfg(),
            meta_cfg(enable_cache=True),
            backend=backend,
            cache_dir=tmp_path,
        )
        call = dict(
            prompt="p",
            parser_fields={},
            max_retries=0,
            prompt_version="v1",
            dataset_id="ds",
            item_id="i1",
        )
        model._execute_llm_call(**call)

        second = model._execute_llm_call(**call)

        assert second.source == "cache"
        assert second.raw_response is None
        # The whole point of the cache: the second call cost no generation.
        assert backend.calls == 1

    def test_a_failed_parse_is_never_cached(self, tmp_path):
        """Otherwise one malformed answer would be served for the rest of the
        project — the cache serves parsed values, so a cached failure is
        permanent."""
        model = HarnessModel(
            model_cfg(),
            meta_cfg(enable_cache=True),
            backend=FakeBackend(Answer(raw="junk")),
            parser=FakeParser(script=["none", "none"]),
            cache_dir=tmp_path,
        )

        result = model._execute_llm_call(
            prompt="p",
            parser_fields={},
            max_retries=1,
            prompt_version="v1",
            dataset_id="ds",
            item_id="i1",
        )

        key, _ = model._build_cache_key("fake", "p", "binary", "v1", "ds")
        assert result.success is False
        assert model.cache.get_entry(key=key, model_id="m", **SLOT_DEFAULTS) is None


# ------------------------------------------------------------------
# The write race: another thread got there first
# ------------------------------------------------------------------
class _RacingBackend(FakeBackend):
    """Stands in for the thread that wins the cache race.

    Writes a foreign value into the slot the caller is about to fill, then
    answers — so by the time ``_write_cache`` runs, ``get_or_set`` reports
    the slot as already taken. Deterministic where two real threads would
    not be.
    """

    def __init__(self, model: HarnessModel, foreign_value, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.foreign_value = foreign_value

    def generate(self, prompt: str) -> Answer:
        """Answer, but let the other thread write first."""
        answer = super().generate(prompt)
        key, _ = self.model._build_cache_key("fake", prompt, "binary", "v1", "ds")
        self.model.cache.get_or_set(
            key=key,
            value=self.foreign_value,
            prompt=prompt,
            slot=self.model.meta_cfg.prompt_nbr,
            model_id="m",
            **SLOT_DEFAULTS,
        )
        return answer


class TestLostWriteRace:
    def _run(self, tmp_path):
        model = HarnessModel(
            model_cfg(),
            meta_cfg(enable_cache=True),
            cache_dir=tmp_path,
        )
        model.backend = _RacingBackend(
            model,
            {"text": "winner"},
            answer=Answer(raw="loser"),
        )
        result = model._execute_llm_call(
            prompt="p",
            parser_fields={},
            max_retries=0,
            prompt_version="v1",
            dataset_id="ds",
            item_id="i1",
        )
        return model, result

    def test_the_loser_adopts_the_winners_value(self, tmp_path):
        """Two threads on one key must not hand two different answers
        downstream — whoever wrote first defines the value for both, so the
        artifact and the cache never disagree."""
        _model, result = self._run(tmp_path)

        assert result.success is True
        assert result.value == {"text": "winner"}
        assert result.attempts == 1

    def test_the_lost_race_is_reported_as_a_cache_hit(self, tmp_path):
        """The value did come out of the cache, and saying so is what keeps
        `source` an honest statement about where the value originated."""
        _model, result = self._run(tmp_path)

        assert result.source == "cache"

    def test_the_generation_is_still_counted(self, tmp_path):
        """The call was paid for even though the value was discarded; the
        stats must show both, or a lost race looks free."""
        model, _result = self._run(tmp_path)

        assert model.llm_stats.llm_calls == 1
        assert model.llm_stats.cache_hits == 1
        assert model.llm_stats.parse_success == 1

    def test_the_winners_entry_survives(self, tmp_path):
        """The loser must not overwrite the slot it just lost."""
        model, _result = self._run(tmp_path)

        key, _ = model._build_cache_key("fake", "p", "binary", "v1", "ds")
        entry = model.cache.get_entry(key=key, model_id="m", **SLOT_DEFAULTS)
        assert entry["answers"]["1"] == {"text": "winner"}


# ------------------------------------------------------------------
# Cache-key preconditions
# ------------------------------------------------------------------
class TestCacheKeyPreconditions:
    def test_a_cached_run_needs_a_dataset_namespace(self, tmp_path):
        """Entries are filed under the dataset id; guessing one would mix two
        datasets' answers into a single shard."""
        model = HarnessModel(
            model_cfg(),
            meta_cfg(enable_cache=True),
            cache_dir=tmp_path,
        )

        with pytest.raises(ValueError, match="dataset_id must be provided"):
            model._execute_llm_call(
                prompt="p",
                parser_fields={},
                max_retries=0,
                prompt_version="v1",
            )

    def test_without_a_cache_there_is_no_key(self):
        """And no dataset_id demand either — an uncached run is legitimate."""
        model = HarnessModel()

        assert model._build_cache_key("fake", "p", "binary", "v1", None) == (None, None)


class TestWriteCacheInvariants:
    """``_write_cache`` is only reached behind ``if self.cache and cache_key``.

    The guards turn a caller that forgets that into a crash instead of a
    silent no-op, which is the difference between "the cache is broken" and
    "the run was mysteriously slow".
    """

    def test_without_a_cache(self):
        model = HarnessModel()

        with pytest.raises(RuntimeError, match="self.cache is None"):
            model._write_cache(_ctx(), parsed={"a": 1}, raw="r")

    def test_without_a_key(self, tmp_path):
        model = HarnessModel(
            model_cfg(),
            meta_cfg(enable_cache=True),
            cache_dir=tmp_path,
        )

        with pytest.raises(RuntimeError, match="cache_key is None"):
            model._write_cache(_ctx(cache_key=None), parsed={"a": 1}, raw="r")

    def test_without_a_dataset_id(self, tmp_path):
        model = HarnessModel(
            model_cfg(),
            meta_cfg(enable_cache=True),
            cache_dir=tmp_path,
        )

        with pytest.raises(RuntimeError, match="cache_dataset_id is None"):
            model._write_cache(_ctx(cache_dataset_id=None), parsed={"a": 1}, raw="r")


# ------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------
class TestDiagnostics:
    def test_trace_recorded_but_never_parsed(self):
        model = HarnessModel(
            model_cfg(),
            meta_cfg(store_parse_diagnostics=True),
            backend=FakeBackend(Answer(raw="the answer", thinking="the trace")),
        )
        result = model._execute_llm_call(
            prompt="p",
            parser_fields={},
            max_retries=0,
            prompt_version="v1",
            item_id="i1",
        )

        # Only the answer went to the parser.
        assert result.value == {"text": "the answer"}
        diags = [
            d for entries in model.llm_stats.parse_diagnostics.values() for d in entries
        ]
        assert len(diags) == 1
        assert diags[0].raw_response == "the answer"
        assert diags[0].thinking == "the trace"

    def test_a_raising_parser_is_a_parse_failure_with_its_message(self):
        """The exception must reach the diagnostic; swallowing it leaves a
        failed item whose only trace is a count."""
        model = HarnessModel(
            model_cfg(),
            meta_cfg(store_parse_diagnostics=True),
            backend=FakeBackend(Answer(raw="junk")),
            parser=FakeParser(script=[ValueError("no label found")]),
        )

        result = model._execute_llm_call(
            prompt="p",
            parser_fields={},
            max_retries=0,
            prompt_version="v1",
            item_id="i1",
        )

        assert result.error_type == "parse_failure"
        assert result.error == "no label found"
        diag = model.llm_stats.parse_diagnostics["binary"][0]
        assert diag.error == "no label found"
        assert diag.parsed_output is None

    def test_diagnostics_stay_off_when_not_asked_for(self):
        """They hold every raw answer of a run; recording them unasked turns a
        large batch's stats file into a copy of the run."""
        model = HarnessModel(
            model_cfg(),
            meta_cfg(store_parse_diagnostics=False),
            backend=FakeBackend(Answer(raw="the answer")),
        )

        model._execute_llm_call(
            prompt="p",
            parser_fields={},
            max_retries=0,
            prompt_version="v1",
            item_id="i1",
        )

        assert dict(model.llm_stats.parse_diagnostics) == {}


# ------------------------------------------------------------------
# A model that was never fully initialized
# ------------------------------------------------------------------
class TestUninitializedModelFailsLoud:
    """Every entry point checks the two things it cannot do without.

    Without these the failure surfaces much later, as an AttributeError on
    ``None`` deep inside a worker thread of a batch that is already running.
    """

    def test_execute_without_a_parser(self):
        model = HarnessModel(parser=None)

        with pytest.raises(RuntimeError, match="Parser not initialized"):
            model._execute_llm_call(
                prompt="p",
                parser_fields={},
                max_retries=0,
                prompt_version="v1",
            )

    def test_execute_takes_the_explicit_parser_over_the_missing_one(self):
        """The stateless API must work on a model that has no parser of its
        own — that is the whole point of workflow stages."""
        model = HarnessModel(parser=None)

        result = model._execute_llm_call(
            prompt="p",
            parser_fields={},
            max_retries=0,
            prompt_version="v1",
            parser=FakeParser(),
        )

        assert result.value == {"text": "fine"}

    def test_execute_without_meta_cfg(self):
        model = HarnessModel(model_cfg(), None)

        with pytest.raises(RuntimeError, match="meta_cfg not initialized"):
            model._execute_llm_call(
                prompt="p",
                parser_fields={},
                max_retries=0,
                prompt_version="v1",
            )

    def test_run_llm_call_without_a_parser(self):
        model = HarnessModel(parser=None, template=_TEMPLATE)

        with pytest.raises(RuntimeError, match="Parser not initialized"):
            model.run_llm_call(question="q")

    def test_run_llm_call_without_a_template(self):
        model = HarnessModel()

        with pytest.raises(RuntimeError, match="Prompt template not initialized"):
            model.run_llm_call(question="q")

    def test_stateless_call_without_meta_cfg(self):
        model = HarnessModel(model_cfg(), None)

        with pytest.raises(RuntimeError, match="meta_cfg not initialized"):
            model._run_llm_call_with(
                template=_TEMPLATE,
                parser=FakeParser(),
                fields={"question": "q"},
            )

    def test_run_llm_call_renders_the_template_with_the_fields(self):
        """The normal case for the stateful API: fields reach the prompt, and
        the prompt reaches the backend verbatim."""
        backend = FakeBackend()
        model = HarnessModel(backend=backend, template=_TEMPLATE)

        result = model.run_llm_call(question="why")

        assert backend.prompts == ["Q: why"]
        assert result.value == {"text": "fine"}


# ------------------------------------------------------------------
# Per-task sampling slots: repeated draws of one prompt stay distinct
# ------------------------------------------------------------------
class TestSampleSlots:
    """Multi-vote sampling at T>0 keys each draw into its own cache slot.

    Without a per-task slot, every draw of the same prompt shares one
    cache entry: draw 1 writes, draws 2..k read it back, and the vote
    distribution silently collapses to a single sample.
    """

    def _model(self, tmp_path, script=None, **meta):
        return HarnessModel(
            model_cfg(),
            meta_cfg(enable_cache=True, **meta),
            backend=FakeBackend(script=script),
            cache_dir=tmp_path,
        )

    def _call(self, model, sample_slot=None):
        return model._execute_llm_call(
            prompt="p", parser_fields={}, max_retries=0,
            prompt_version="v1", dataset_id="ds", item_id="i1",
            sample_slot=sample_slot,
        )

    def test_each_slot_is_its_own_draw(self, tmp_path):
        model = self._model(tmp_path, script=[Answer(raw="A"), Answer(raw="B")])

        r1 = self._call(model, sample_slot="v1")
        r2 = self._call(model, sample_slot="v2")

        assert model.backend.calls == 2
        assert (r1.value, r2.value) == ({"text": "A"}, {"text": "B"})
        key, _ = model._build_cache_key("fake", "p", "binary", "v1", "ds")
        answers = model.cache.get_all(key=key, model_id="m", **SLOT_DEFAULTS)
        assert answers == {"1:v1": {"text": "A"}, "1:v2": {"text": "B"}}

    def test_replay_matches_slots_exactly(self, tmp_path):
        """Slot "1" and slot "1:v1" resolve independently — no prefix match."""
        model = self._model(tmp_path)
        key, _ = model._build_cache_key("fake", "p", "binary", "v1", "ds")
        for slot, value in (("1", {"text": "plain"}), ("1:v1", {"text": "sampled"})):
            model.cache.get_or_set(
                key=key, slot=slot, prompt="p", value=value, model_id="m",
                **SLOT_DEFAULTS,
            )

        with_slot = self._call(model, sample_slot="v1")
        without = self._call(model)

        assert model.backend.calls == 0
        assert with_slot.value == {"text": "sampled"}
        assert without.value == {"text": "plain"}

    def test_prompt_nbr_prefix_keeps_replications_apart(self, tmp_path):
        """Same suffix under prompt_nbr 1 and 2 → two draws, not a replay.

        This is the replication axis staying alive: run p2 must not read
        back p1's votes just because both sampled slot "v1".
        """
        results = []
        for nbr in (1, 2):
            model = HarnessModel(
                model_cfg(),
                meta_cfg(enable_cache=True, prompt_nbr=nbr),
                backend=FakeBackend(script=[Answer(raw=f"draw{nbr}")]),
                cache_dir=tmp_path,
            )
            results.append(self._call(model, sample_slot="v1"))
            assert model.backend.calls == 1, nbr
        assert [r.value for r in results] == [{"text": "draw1"}, {"text": "draw2"}]

    def test_the_task_dict_carries_the_slot(self, tmp_path):
        """``sample_slot`` travels from the task fields to the cache slot."""
        model = self._model(tmp_path, script=[Answer(raw="A"), Answer(raw="B")])

        results = model.run_llm_tasks_with(
            template=_TEMPLATE,
            parser=FakeParser(),
            tasks=[
                {"question": "q", "dataset_id": "ds", "sample_slot": f"v{n}"}
                for n in (1, 2)
            ],
            desc="votes",
        )

        assert model.backend.calls == 2
        assert [r.value for r in results] == [{"text": "A"}, {"text": "B"}]
