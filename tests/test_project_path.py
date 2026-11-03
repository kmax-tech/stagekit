"""A relative path in a config is a path below the project root, wherever the run started.

Resolved where it is used (``stagekit.config.paths.project_path``), never
written back: the config keeps its text, which the repro hash reads. The
case it answers: a run started from an experiment area inside the project
(fallacy-detection, experiments/01-pilot-bigbench-zs) read the dataset and
the LLM cache relative to the area.
"""

import pytest

from stagekit.analysis.repro_hash import compute_repro_hash
from stagekit.config import paths
from stagekit.config.classes import AppConfig, DataConfig, MetaConfig, PipelineStageConfig
from stagekit.config.validation import _check_data_paths
from stagekit.models.llm_model import GenericLLMModel
from stagekit.run.utils import make_run_context
from stagekit.stages.base import PipelineStage
from tests.llm_harness import model_cfg


@pytest.fixture()
def project(monkeypatch, tmp_path):
    """A project named by the variable, the run started elsewhere."""
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "config" / "config.yaml").write_text("topics: []\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv(paths.ENV_VAR, str(root))
    monkeypatch.chdir(elsewhere)
    paths.find_project_root.cache_clear()
    yield root.resolve()
    paths.find_project_root.cache_clear()


def test_a_relative_path_is_below_the_root(project):
    assert paths.project_path("data/x.jsonl") == project / "data" / "x.jsonl"


@pytest.mark.usefixtures("project")
def test_an_absolute_path_stays(tmp_path):
    assert paths.project_path(tmp_path / "abs.jsonl") == tmp_path / "abs.jsonl"


def test_without_a_project_a_relative_path_is_refused(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    paths.find_project_root.cache_clear()
    try:
        with pytest.raises(paths.ProjectRootError):
            paths.project_path("data/x.jsonl")
    finally:
        paths.find_project_root.cache_clear()


def test_the_llm_cache_lives_below_the_root(project):
    model = GenericLLMModel(
        model_cfg(), None, MetaConfig(cache_dir="data/cache/llm", cache_mode="replay"),
    )
    assert model.cache is not None
    assert model.cache.cache_dir == project / "data" / "cache" / "llm"


def test_the_run_directory_is_below_the_root(project):
    cfg = AppConfig(output_dir="out")
    cfg.meta.run_name = "r"
    assert make_run_context(cfg, run_type="pipeline").run_dir == project / "out" / "pipeline" / "r"


def test_the_dataset_check_looks_below_the_root(project):
    (project / "data").mkdir()
    (project / "data" / "x.jsonl").write_text("")
    errors: list[str] = []
    _check_data_paths(AppConfig(data=DataConfig(dataset_path="data/x.jsonl")), errors)
    assert errors == []
    _check_data_paths(AppConfig(data=DataConfig(dataset_path="data/missing.jsonl")), errors)
    assert len(errors) == 1


class _Stage(PipelineStage):
    input_path_config_keys = ["input_path"]
    output_path_config_keys = ["output_path"]


def test_a_stage_gets_its_declared_paths_resolved_and_the_config_keeps_its_text(project):
    override = {"input_path": "data/in.jsonl", "output_path": "data/out.jsonl", "label": "data/x"}
    stage_cfg = PipelineStageConfig(name="s", config_override=override)

    co = _Stage().get_config(stage_cfg)

    assert co == {
        "input_path": str(project / "data" / "in.jsonl"),
        "output_path": str(project / "data" / "out.jsonl"),
        "label": "data/x",  # not declared as a path: untouched
    }
    assert stage_cfg.config_override == {
        "input_path": "data/in.jsonl", "output_path": "data/out.jsonl", "label": "data/x",
    }


@pytest.mark.usefixtures("project")
def test_resolution_leaves_the_repro_hash_alone():
    """The hash reads the config's text; resolving at use changes no text."""
    cfg = AppConfig(data=DataConfig(dataset_path="data/x.jsonl"), meta=MetaConfig(cache_dir="data/cache/llm"))
    before = compute_repro_hash(cfg)
    paths.project_path(cfg.data.dataset_path)
    make_run_context(cfg, run_type="pipeline")
    assert compute_repro_hash(cfg) == before
    assert cfg.data.dataset_path == "data/x.jsonl"
