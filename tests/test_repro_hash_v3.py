"""Formula 3: a field at its default and a field gone from the schema do not
enter the repro hash; everything that can change the output still does.

The configs are synthetic. ``_stored_in_may`` rebuilds the shape of a
config stored before fields were added to and removed from the schema.
"""

from __future__ import annotations

import copy
from dataclasses import asdict

import pytest

from stagekit.analysis import repro_hash as rh
from stagekit.analysis.repro_hash import (
    REPRO_HASH_VERSION, compute_repro_hash, repro_hash_from_dict, repro_hash_version,
)
from stagekit.config import classes
from stagekit.config.classes import (
    AppConfig, DataConfig, LLMConfig, MetaConfig, ModelConfig, PipelineConfig,
    PipelineStageConfig,
)
from stagekit.run.manager import ReproducibilityTracker


def _app() -> AppConfig:
    return AppConfig(
        meta=MetaConfig(experiment_name="toy-eval", experiment_config="eval/toy-exp",
                        data_config="toy-data", pipeline_config="eval/toy-zs",
                        run_group="toy-group", max_workers=50),
        data=DataConfig(data_loader="toy_loader", data_name="toy",
                        dataset_path="data/toy.jsonl"),
        model=ModelConfig(backend="ollama", model_name="tiny-model", temperature=0.3),
        llm_task=LLMConfig(task_name="toy_zs", template="toy_zs", parser="toy_parser"),
        pipeline=PipelineConfig(
            name="toy-zs", mode="neutral", variant="both",
            stages=[PipelineStageConfig(name="classify", config_override={"table_method": "zs"})],
        ),
    )


def _composed_today(tmp_path) -> dict:
    """The dict a run composed today stores as config_used.yaml."""
    return ReproducibilityTracker(tmp_path)._config_to_repro_dict(_app())


def _stored_in_may(today: dict) -> dict:
    """The same run as a config stored before the schema changed: fields
    since removed are present, fields since added are absent, the volatile
    meta differs, and there is no instrument snapshot yet."""
    old = copy.deepcopy(today)
    for key in ("instrument", "topics"):
        old.pop(key)
    for key in ("cache_mode", "replay_failed_items", "labkit_run_id"):
        old["meta"].pop(key)
    old["meta"].update(run_id="another-run", timestamp="2026-05-25_10-09-32", max_workers=10)
    for key in ("max_tokens", "num_ctx", "reasoning", "stop", "supports_temperature"):
        old["model"].pop(key)
    old["model"].update(fewshot=False, inference_batch_size=16, max_length=256, num_labels=0)
    for key in ("preload_stages", "derived_artifacts"):
        old["pipeline"].pop(key)
    old["pipeline"].update(artifact_queries={}, virtual_stages={}, resume_from="")
    return old


# --------------------------------------------------
# Version
# --------------------------------------------------

def test_todays_formula_is_3():
    assert REPRO_HASH_VERSION == 3
    assert repro_hash_from_dict({"pipeline": {"stages": []}}).startswith("v3-")


def test_formula_2_stays_computable_and_unchanged(tmp_path):
    """A frozen literal: formula 2 must still give what it gave on 2026-11-03."""
    cfg = _composed_today(tmp_path)
    assert repro_hash_from_dict(cfg, version=2) == (
        "v2-6a1ffd035c756c6bae1d2b120eaa55cd10297231ce55a65a7514e3f58ee2b33e")


def test_formula_3_literal(tmp_path):
    cfg = _composed_today(tmp_path)
    assert repro_hash_from_dict(cfg) == (
        "v3-e0984a7332b64255372365cd938cb1a9931bf8f0e2060e681872a82fab76dcc6")


def test_formula_1_cannot_be_computed():
    with pytest.raises(ValueError, match="formula 1 cannot be computed"):
        repro_hash_from_dict({}, version=1)


def test_a_stored_v2_hash_is_recognised_as_formula_2(tmp_path):
    v2 = repro_hash_from_dict(_composed_today(tmp_path), version=2)
    assert repro_hash_version(v2) == 2
    assert v2 != repro_hash_from_dict(_composed_today(tmp_path))


# --------------------------------------------------
# A schema change alone keeps the identity
# --------------------------------------------------

def test_composed_today_and_stored_in_may_hash_alike(tmp_path):
    today = _composed_today(tmp_path)
    may = _stored_in_may(today)
    # formula 2 told them apart — the reason formula 3 exists
    assert repro_hash_from_dict(may, version=2) != repro_hash_from_dict(today, version=2)
    assert repro_hash_from_dict(may) == repro_hash_from_dict(today)


def test_a_field_at_its_default_hashes_as_absent(tmp_path):
    today = _composed_today(tmp_path)
    without = copy.deepcopy(today)
    without["pipeline"].pop("feature_views")
    without["model"].pop("model_dir")
    without["meta"].pop("prompt_nbr")
    assert repro_hash_from_dict(without) == repro_hash_from_dict(today)


# --------------------------------------------------
# What can change the output still changes the hash
# --------------------------------------------------

def _changed(cfg: dict, path: tuple, value) -> dict:
    out = copy.deepcopy(cfg)
    node = out
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return out


@pytest.mark.parametrize("path,value", [
    (("model", "temperature"), 0.9),
    (("model", "temperature"), 1),  # moved to the default: still another run
    (("model", "model_name"), "other-model"),
    (("data", "dataset_path"), "data/other.jsonl"),
    (("llm_task", "template"), "other_template"),
    (("pipeline", "stages", 0, "config_override"), {"table_method": "fs"}),
    (("instrument", "prompts", "toy_zs"), "a-new-version"),
    (("instrument", "parsers", "toy_parser"), "toy_parser_v2"),
    (("random_state", ), 7),
], ids=["temperature", "temperature-to-default", "model_name", "dataset_path",
        "template", "stage-option", "prompt-version", "parser-cache-id", "seed"])
def test_an_output_changing_field_changes_the_hash(tmp_path, path, value):
    today = _composed_today(tmp_path)
    assert repro_hash_from_dict(_changed(today, path, value)) != repro_hash_from_dict(today)


def test_temperature_changes_the_hash_of_an_appconfig():
    a, b = _app(), _app()
    b.model.temperature = 0.9
    assert compute_repro_hash(a) != compute_repro_hash(b)


# --------------------------------------------------
# The defaults formula 3 reads are frozen
# --------------------------------------------------

#: The default of every field that can enter formula 3's canon. A run
#: stored at an old default hashes as if it ran at today's, so changing a
#: default here is a formula change: a new REPRO_HASH_VERSION. Adding a
#: field (at a default meaning "not used") or removing one is what formula
#: 3 absorbs: extend or shrink the literal.
FROZEN_DEFAULTS = {
    "MetaConfig": {"experiment_name": "", "experiment_config": "", "data_config": "",
                   "pipeline_config": "", "prompt_nbr": 1},
    "DataConfig": {"data_loader": "", "dataset_path": "default_dataset",
                   "data_name": "default_name", "label_info_path": None,
                   "split_id": "no_split", "split_dir": "split", "mode": "dataset",
                   "file_path": None},
    "ModelConfig": {"model_type": "llm", "backend": "", "model_name": "", "model_dir": None,
                    "temperature": 1, "supports_temperature": True, "max_tokens": None,
                    "reasoning": False, "stop": None, "num_ctx": None},
    "LLMConfig": {"task_name": "", "template": None, "parser": None},
    "PipelineConfig": {"name": "", "stages": [], "mode": "", "variant": "",
                       "source_model": "", "method": "", "artifact_root": "",
                       "feature_sources": {}, "feature_source_scopes": {},
                       "feature_views": {}, "preload_stages": {}, "source_modes": [],
                       "preload_artifacts": {}, "derived_artifacts": {},
                       "preload_expect": {}},
    "PipelineStageConfig": {"name": "(none)", "stage_type": "", "model": None,
                            "llm_task": None, "config_override": None, "depends_on": None},
    "AppConfig.random_state": 42,
}


def test_the_defaults_formula_3_reads_are_frozen():
    volatile = {"MetaConfig": rh._VOLATILE_META, "ModelConfig": rh._VOLATILE_MODEL,
                "PipelineStageConfig": rh._VOLATILE_STAGE}
    today: dict = {}
    for name in ("MetaConfig", "DataConfig", "ModelConfig", "LLMConfig", "PipelineConfig",
                 "PipelineStageConfig"):
        today[name] = {
            k: "(none)" if v is rh._NO_DEFAULT else v
            for k, v in rh._defaults(getattr(classes, name)).items()
            if k not in volatile.get(name, ())
        }
    today["AppConfig.random_state"] = rh._defaults(AppConfig)["random_state"]
    assert today == FROZEN_DEFAULTS


def test_asdict_of_the_app_config_hashes_as_its_dataclass():
    assert repro_hash_from_dict(asdict(_app())) == compute_repro_hash(_app())
