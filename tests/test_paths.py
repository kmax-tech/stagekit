"""Which directory is the project, and which .env is its own.

The case 4a of the stagekit plan needs: a run started in an experiment
area inside the project (cwd = <project>/experiments/<area>) uses the
project's root and .env, not the area's.
"""

import os

import pytest

from stagekit.config import paths


@pytest.fixture()
def fresh(monkeypatch, tmp_path):
    """No variable, an empty lookup cache, the cache emptied again afterwards."""
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    paths.find_project_root.cache_clear()
    yield tmp_path
    paths.find_project_root.cache_clear()


def make_project(root):
    (root / "config").mkdir(parents=True)
    (root / "config" / "config.yaml").write_text("topics: []\n")
    return root


def test_the_search_from_an_area_finds_the_project(fresh, monkeypatch):
    project = make_project(fresh / "project")
    area = project / "experiments" / "04-pilot"
    (area / "config").mkdir(parents=True)
    (area / "config" / "experiment.yaml").write_text("{}\n")  # no config.yaml: not a project
    monkeypatch.chdir(area)
    assert paths.find_project_root() == project.resolve()
    assert paths.CONFIG_DIR == project.resolve() / "config"
    assert paths.PINS_FILE == project.resolve() / "published" / "pins.yaml"


def test_the_nearest_config_yaml_wins(fresh, monkeypatch):
    outer = make_project(fresh / "outer")
    inner = make_project(outer / "inner")
    monkeypatch.chdir(inner)
    assert paths.find_project_root() == inner.resolve()


def test_the_variable_wins_over_the_working_directory(fresh, monkeypatch):
    named = make_project(fresh / "named")
    monkeypatch.chdir(make_project(fresh / "here"))
    monkeypatch.setenv(paths.ENV_VAR, str(named))
    assert paths.find_project_root() == named.resolve()


def test_a_variable_without_config_yaml_is_refused(fresh, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(fresh))
    with pytest.raises(paths.ProjectRootError, match=f"{paths.ENV_VAR}=.* holds no config/config.yaml"):
        paths.find_project_root()


def test_no_project_is_refused_on_use_not_on_import(fresh, monkeypatch):
    monkeypatch.chdir(fresh)
    with pytest.raises(paths.ProjectRootError, match="no config/config.yaml in .* or a parent"):
        _ = paths.PROJECT_ROOT
    with pytest.raises(AttributeError):  # other names never look the root up
        _ = paths.__path__  # type: ignore[attr-defined]


def test_the_env_of_the_root_is_loaded_not_the_areas(fresh, monkeypatch):
    project = make_project(fresh / "project")
    (project / ".env").write_text("STAGEKIT_TEST_SERVER=root\n")
    area = project / "experiments" / "04-pilot"
    area.mkdir(parents=True)
    (area / ".env").write_text("STAGEKIT_TEST_SERVER=area\n")
    monkeypatch.chdir(area)
    monkeypatch.delenv("STAGEKIT_TEST_SERVER", raising=False)
    assert paths.load_project_env() == project.resolve() / ".env"
    assert os.environ["STAGEKIT_TEST_SERVER"] == "root"
    monkeypatch.delenv("STAGEKIT_TEST_SERVER")


def test_the_env_follows_the_variable(fresh, monkeypatch):
    named = make_project(fresh / "named")
    (named / ".env").write_text("STAGEKIT_TEST_SERVER=named\n")
    monkeypatch.chdir(make_project(fresh / "here"))
    monkeypatch.setenv(paths.ENV_VAR, str(named))
    monkeypatch.delenv("STAGEKIT_TEST_SERVER", raising=False)
    assert paths.load_project_env() == named.resolve() / ".env"
    assert os.environ["STAGEKIT_TEST_SERVER"] == "named"
    monkeypatch.delenv("STAGEKIT_TEST_SERVER")


def test_no_project_or_no_env_loads_nothing(fresh, monkeypatch):
    monkeypatch.chdir(fresh)
    assert paths.load_project_env() is None
    paths.find_project_root.cache_clear()
    monkeypatch.chdir(make_project(fresh / "bare"))
    assert paths.load_project_env() is None
