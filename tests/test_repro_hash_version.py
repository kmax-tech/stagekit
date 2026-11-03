"""The repro hash says which formula made it."""

import pytest

from stagekit.analysis.repro_hash import REPRO_HASH_VERSION, repro_hash_from_dict, repro_hash_version


def test_a_new_hash_carries_todays_version():
    h = repro_hash_from_dict({"pipeline": {"stages": []}})
    assert h.startswith(f"v{REPRO_HASH_VERSION}-") and len(h) == len(f"v{REPRO_HASH_VERSION}-") + 64
    assert repro_hash_version(h) == REPRO_HASH_VERSION


def test_an_unprefixed_stored_hash_is_formula_1():
    assert repro_hash_version("c8f5bb690aa454837d307a4e2403fdf341ea5f827b58babf7acff146dce21191") == 1


@pytest.mark.parametrize("bad", ["", "v2-abc", "C8F5" * 16, "v-" + "a" * 64])
def test_anything_else_is_refused(bad):
    with pytest.raises(ValueError, match="not a repro hash"):
        repro_hash_version(bad)


def test_the_run_index_counts_runs_per_version(tmp_path):
    from types import SimpleNamespace

    from stagekit.analysis.run_index import RunIndex

    index = RunIndex(pipeline_root=tmp_path, catalog_path=tmp_path / "c.json", pins_path=tmp_path / "p.yaml")
    run = lambda h, status="success": SimpleNamespace(  # noqa: E731
        run_type="pipeline", status=status, pipeline=SimpleNamespace(repro_hash=h))
    index._records = [run("a" * 64), run("a" * 64), run(f"v{REPRO_HASH_VERSION}-" + "b" * 64),
                      run("c" * 64, status="failed"), run("")]
    assert index.formula_versions() == {1: 2, REPRO_HASH_VERSION: 1}
