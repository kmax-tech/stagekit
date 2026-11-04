"""The repro hash says which formula made it; RunIndex compares runs across
formulas by today's formula over their stored config."""

from types import SimpleNamespace

import pytest
import yaml

from stagekit.analysis import repro_hash as rh
from stagekit.analysis.repro_hash import (
    REPRO_HASH_VERSION, compute_repro_hash, repro_hash_from_dict, repro_hash_version,
)
from stagekit.analysis.run_index import RunIndex
from stagekit.analysis.run_record import PipelineFields
from tests.test_repro_hash_v3 import _app, _changed, _composed_today, _stored_in_may


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
    index = RunIndex(pipeline_root=tmp_path, catalog_path=tmp_path / "c.json", pins_path=tmp_path / "p.yaml")
    run = lambda h, status="success": SimpleNamespace(  # noqa: E731
        run_type="pipeline", status=status, pipeline=SimpleNamespace(repro_hash=h))
    index._records = [run("a" * 64), run("a" * 64), run(f"v{REPRO_HASH_VERSION}-" + "b" * 64),
                      run("c" * 64, status="failed"), run("")]
    assert index.formula_versions() == {1: 2, REPRO_HASH_VERSION: 1}


# --------------------------------------------------
# RunIndex: today's formula over the stored config
# --------------------------------------------------

def _write_run(root, name: str, cfg: dict) -> None:
    run_dir = root / "pipeline" / name
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        f'{{"run_type": "pipeline", "run_id": "{name}", "experiment_name": "toy-eval"}}')
    (run_dir / "pipeline_status.json").write_text('{"status": "success"}')
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg))


def _index(root) -> RunIndex:
    return RunIndex(pipeline_root=root / "pipeline", catalog_path=root / "catalog.json",
                    pins_path=root / "pins.yaml").scan(force=True)


def test_a_formula_1_run_with_an_instrument_is_found(tmp_path):
    stored = _composed_today(tmp_path)
    stored["repro_hash"] = "a" * 64  # formula 1, made by some earlier canon
    _write_run(tmp_path, "old", stored)
    index = _index(tmp_path)
    assert index.has_equivalent(_app())
    assert index.formula_versions() == {1: 1}
    assert index.incomparable_runs() == {}


def test_a_run_without_an_instrument_is_not_compared_and_counted(tmp_path):
    may = _stored_in_may(_composed_today(tmp_path))
    may["repro_hash"] = "b" * 64
    _write_run(tmp_path, "may", may)
    index = _index(tmp_path)
    assert not index.has_equivalent(_app())
    assert index.incomparable_runs() == {1: 1}


def test_a_stored_hash_of_todays_formula_does_not_outrank_the_recomputation(tmp_path):
    """Stored at another temperature but carrying today's target hash (as a
    run stored at an earlier default would): the stored config decides."""
    stored = _changed(_composed_today(tmp_path), ("model", "temperature"), 0.9)
    stored["repro_hash"] = compute_repro_hash(_app())
    _write_run(tmp_path, "other-temperature", stored)
    assert not _index(tmp_path).has_equivalent(_app())


def test_the_catalog_is_rescanned_under_another_formula(tmp_path, monkeypatch):
    _write_run(tmp_path, "run", {**_composed_today(tmp_path), "repro_hash": "c" * 64})
    _index(tmp_path)  # writes the catalog
    monkeypatch.setattr(rh, "REPRO_HASH_VERSION", 4)
    monkeypatch.setitem(rh._CANON_BUILDERS, 4, rh._build_canonical_v3)
    index = RunIndex(pipeline_root=tmp_path / "pipeline", catalog_path=tmp_path / "catalog.json",
                     pins_path=tmp_path / "pins.yaml")
    assert not index._load_cache()


def test_the_index_counts_from_records(tmp_path):
    index = _index(tmp_path)
    run = lambda h, recomputed="": SimpleNamespace(  # noqa: E731
        run_type="pipeline", status="success",
        pipeline=PipelineFields(repro_hash=h, recomputed_repro_hash=recomputed))
    index._records = [run("a" * 64), run("a" * 64, f"v3-{'d' * 64}"),
                      run(f"v{REPRO_HASH_VERSION}-" + "e" * 64)]
    assert index.formula_versions() == {1: 2, 3: 1}
    assert index.incomparable_runs() == {1: 1}
