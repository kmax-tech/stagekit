"""Replay: every answer from the cache, no backend built or called.

``meta.cache_mode='replay'`` (stagekit/models/llm_cache.py, CACHE_MODES)
runs the processing again on the answers an earlier run stored. A miss
stops the run, naming item and key — except for an item the caller lists in
``meta.replay_failed_items`` (the replayed run's own failed items), whose
miss counts as failed, as in that run. The mode reaches
pipeline_summary.json and stays out of the repro hash.
"""

import pytest

import stagekit.models.base as base_module
from stagekit.analysis.repro_hash import compute_repro_hash
from stagekit.config.classes import AppConfig, MetaConfig
from stagekit.config.validation import _check_meta_config
from stagekit.main_pipeline import run_summary_fields
from stagekit.models.base import LLMBaseModel
from stagekit.models.llm_cache import CacheMiss
from stagekit.prompts.base import PromptTemplate
from tests.llm_harness import FakeBackend, FakeParser, HarnessModel, meta_cfg, model_cfg

_TEMPLATE = PromptTemplate(name="t", template_str="Q: $question", required_fields=["question"])


def _call(model, item_id="i1", prompt="p"):
    return model._execute_llm_call(
        prompt=prompt, parser_fields={}, max_retries=3,
        prompt_version="v1", dataset_id="ds", item_id=item_id,
    )


def _replay_model(tmp_path, backend, failed=()):
    return HarnessModel(
        model_cfg(),
        meta_cfg(enable_cache=True, cache_mode="replay", replay_failed_items=list(failed)),
        backend=backend,
        cache_dir=tmp_path,
    )


def _fill(tmp_path, prompt="p", item_id="i1"):
    """A normal run stores one answer, as the replayed run did."""
    model = HarnessModel(model_cfg(), meta_cfg(enable_cache=True), cache_dir=tmp_path)
    assert _call(model, item_id=item_id, prompt=prompt).success


# ------------------------------------------------------------------
# The model
# ------------------------------------------------------------------
class TestReplayReadsTheCache:
    def test_a_hit_is_read_without_the_backend(self, tmp_path):
        _fill(tmp_path)
        backend = FakeBackend()

        result = _call(_replay_model(tmp_path, backend))

        assert (result.success, result.source, result.value) == (True, "cache", {"text": "fine"})
        assert backend.calls == 0

    def test_a_miss_stops_naming_item_and_key(self, tmp_path):
        backend = FakeBackend()
        model = _replay_model(tmp_path, backend)
        key, _ = model._build_cache_key("fake", "p", "binary", "v1", "ds")
        assert key is not None

        with pytest.raises(CacheMiss) as err:
            _call(model, item_id="bigbench:7")

        assert "'bigbench:7'" in str(err.value) and key in str(err.value)
        assert backend.calls == 0

    def test_a_listed_item_counts_as_failed(self, tmp_path):
        backend = FakeBackend()
        model = _replay_model(tmp_path, backend, failed=["bigbench:692"])

        result = _call(model, item_id="bigbench:692")

        assert (result.success, result.source, result.attempts) == (False, "replay", 0)
        assert result.error_type == "replay_listed_failure"
        assert model.llm_stats.summary()["failed_items"] == {"fake": ["bigbench:692"]}
        assert model.llm_stats.summary()["retry_exhausted_items"] == {}
        assert backend.calls == 0

    def test_the_list_excuses_only_its_items(self, tmp_path):
        model = _replay_model(tmp_path, FakeBackend(), failed=["bigbench:692"])

        with pytest.raises(CacheMiss):
            _call(model, item_id="bigbench:693")

    def test_a_listed_items_hit_is_still_read(self, tmp_path):
        """The list excuses misses only; a stored answer is the replayed run's."""
        _fill(tmp_path, item_id="bigbench:692")

        result = _call(_replay_model(tmp_path, FakeBackend(), failed=["bigbench:692"]), item_id="bigbench:692")

        assert result.success and result.source == "cache"

    def test_a_miss_in_a_batch_stops_the_batch(self, tmp_path):
        tasks = [
            {"question": q, "item_id": f"i{n}", "dataset_id": "ds"}
            for n, q in enumerate(["a", "b"])
        ]
        first = HarnessModel(model_cfg(), meta_cfg(enable_cache=True), cache_dir=tmp_path)
        first.run_llm_tasks_with(template=_TEMPLATE, parser=FakeParser(), tasks=tasks[:1], desc="d")
        backend = FakeBackend(max_safe_workers=None)
        model = _replay_model(tmp_path, backend)
        assert model.meta_cfg is not None
        model.meta_cfg.max_workers = 4

        with pytest.raises(CacheMiss, match="'i1'"):
            model.run_llm_tasks_with(template=_TEMPLATE, parser=FakeParser(), tasks=tasks, desc="d")
        assert backend.calls == 0

    def test_a_replay_without_the_cache_refuses(self):
        model = HarnessModel(
            model_cfg(), meta_cfg(enable_cache=False, cache_mode="replay"), backend=FakeBackend(),
        )

        with pytest.raises(RuntimeError, match="cache is off"):
            _call(model)


class TestReplayBuildsNoBackend:
    def test_no_backend_is_built_and_the_stand_in_refuses(self, monkeypatch):
        """An ollama model without a server: normal mode would reach for it on creation."""
        def must_not_build(_cfg):
            raise AssertionError("build_backend called in a replay")

        monkeypatch.setattr(base_module, "build_backend", must_not_build)
        monkeypatch.delenv("OLLAMA_SERVER", raising=False)

        model = LLMBaseModel(
            model_cfg(backend="ollama", model_name="llama3.2"),
            None,
            MetaConfig(cache_mode="replay"),
        )

        assert model.backend.name == "ollama"  # it keys the cache
        with pytest.raises(CacheMiss, match="calls no backend"):
            model.backend.generate("p")

    def test_normal_mode_still_builds_it(self, monkeypatch):
        built = []
        monkeypatch.setattr(base_module, "build_backend", lambda cfg: built.append(cfg) or FakeBackend())

        LLMBaseModel(model_cfg(), None, MetaConfig())

        assert len(built) == 1


# ------------------------------------------------------------------
# Config, summary, identity
# ------------------------------------------------------------------
def _meta_errors(**meta) -> list[str]:
    errors: list[str] = []
    _check_meta_config(AppConfig(meta=MetaConfig(**meta)), errors)
    return errors


@pytest.mark.parametrize(
    ("meta", "fragment"),
    [
        (dict(cache_mode="repeat"), "not a cache mode"),
        (dict(cache_mode="replay", enable_cache=False), "enable_cache=false"),
        (dict(replay_failed_items=["x"]), "replay only"),
    ],
)
def test_the_config_check_refuses(meta, fragment):
    errors = _meta_errors(**meta)
    assert len(errors) == 1 and fragment in errors[0]


def test_the_config_check_accepts_both_modes():
    assert _meta_errors() == []
    assert _meta_errors(cache_mode="replay", replay_failed_items=["x"]) == []


def test_the_summary_names_mode_list_and_labkit_run():
    cfg = AppConfig(meta=MetaConfig(
        cache_mode="replay", replay_failed_items=["bigbench:692"], labkit_run_id="run-1",
    ))

    assert run_summary_fields(cfg, {"stage": {}}) == {
        "llm_cache": {"enabled": True, "mode": "replay", "replay_failed_items": ["bigbench:692"]},
        "labkit_run_id": "run-1",
    }
    assert run_summary_fields(AppConfig(), {})["labkit_run_id"] is None


def test_a_stage_named_like_a_summary_entry_is_refused():
    with pytest.raises(ValueError, match="llm_cache"):
        run_summary_fields(AppConfig(), {"llm_cache": {}})


def test_the_mode_is_not_identity():
    """A replay is the same configured run as the one it replays."""
    plain = compute_repro_hash(AppConfig())
    replay = compute_repro_hash(AppConfig(meta=MetaConfig(
        cache_mode="replay", replay_failed_items=["x"], labkit_run_id="run-1",
    )))
    assert replay == plain
