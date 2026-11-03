"""Tests for PipelineTableAggregator (stagekit/analysis/table_aggregator.py).

Uses synthetic fixtures — no real pipeline runs needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from stagekit.analysis.repro_hash import repro_hash_from_dict
from stagekit.analysis.run_index import RunIndex
from stagekit.analysis.table_aggregator import PipelineTableAggregator
from stagekit.config.classes import (
    AppConfig, MetaConfig, DataConfig, ModelConfig, LLMConfig,
    PipelineConfig, PipelineStageConfig,
)


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture()
def tmp_root(tmp_path):
    (tmp_path / "pipeline").mkdir()
    return tmp_path


def _make_index(tmp_root, **kwargs):
    return RunIndex(
        pipeline_root=tmp_root / "pipeline",
        catalog_path=tmp_root / "catalog.json",
        **kwargs,
    )


def _make_pipeline_run(
    root: Path,
    experiment_name: str = "exp1",
    run_id: str = "run_001",
    status: str = "success",
    finished_at: str = "2025-01-15T10:00:00",
    model_name: str = "gpt-4",
    data_name: str = "hong",
    pipeline_name: str = "default",
    table_summary: list[dict] | None = None,
    extra_metrics: dict | None = None,
) -> Path:
    """Create a synthetic pipeline run with pipeline_summary.json."""
    run_dir = root / "pipeline" / f"{experiment_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "run_manifest.json").write_text(json.dumps({
        "run_id": run_id,
        "experiment_name": experiment_name,
        "combo_name": "combo",
        "run_type": "pipeline",
    }), encoding="utf-8")

    (run_dir / "pipeline_status.json").write_text(json.dumps({
        "status": status,
        "completed_stages": ["stage1"],
        "total_stages": 1,
        "finished_at": finished_at,
    }), encoding="utf-8")

    (run_dir / "model_usage.json").write_text(json.dumps({
        "stages": {
            "stage1": {
                "model": {"name": model_name},
                "llm_task": {"task_name": "classify", "template": "v1"},
            },
        },
    }), encoding="utf-8")

    from dataclasses import asdict as _asdict
    app = AppConfig(
        meta=MetaConfig(experiment_name=experiment_name, run_group="test"),
        data=DataConfig(data_name=data_name),
        model=ModelConfig(model_type="llm", backend="openai", model_name=model_name),
        llm_task=LLMConfig(task_name="classify", template="v1"),
        pipeline=PipelineConfig(
            name=pipeline_name,
            stages=[PipelineStageConfig(name="stage1", stage_type="llm_enrich")],
        ),
    )
    cfg_dict = _asdict(app)
    for stage in cfg_dict["pipeline"]["stages"]:
        stage.pop("model_cfg", None)
        stage.pop("llm_task_cfg", None)
    cfg_dict["data"].pop("component_label2id", None)
    cfg_dict["repro_hash"] = repro_hash_from_dict(cfg_dict)
    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg_dict, f)

    # pipeline_summary.json
    summary = extra_metrics or {}
    if table_summary is not None:
        summary["table_summary"] = table_summary
    (run_dir / "pipeline_summary.json").write_text(
        json.dumps(summary), encoding="utf-8",
    )

    return run_dir


def _table_entry(
    method: str = "svm",
    stage: str = "classify_svm",
    dataset: str = "hong",
    model: str | None = "gpt-4",
    eval_mode: str = "cv",
    f1: float = 0.85,
    accuracy: float = 0.80,
    mode: str | None = None,
    variant: str | None = None,
) -> dict:
    """Build a minimal table_summary entry."""
    entry = {
        "method": method,
        "stage": stage,
        "dataset": dataset,
        "model": model,
        "eval_mode": eval_mode,
        "metrics": {"f1": f1, "accuracy": accuracy},
    }
    if mode:
        entry["mode"] = mode
    if variant:
        entry["variant"] = variant
    return entry


# --------------------------------------------------
# Tests
# --------------------------------------------------

class TestCollect:
    def test_basic_collect(self, tmp_root):
        _make_pipeline_run(
            tmp_root, run_id="r1",
            table_summary=[_table_entry(f1=0.85)],
        )
        index = _make_index(tmp_root).scan(force=True)
        agg = PipelineTableAggregator(index=index)

        rows = agg.collect(experiment_name="exp1")
        assert len(rows) == 1
        assert rows[0].method == "svm"
        assert rows[0].metrics["f1"] == 0.85
        assert rows[0].run_id == "r1"

    def test_multiple_stages_per_run(self, tmp_root):
        _make_pipeline_run(
            tmp_root, run_id="r1",
            table_summary=[
                _table_entry(method="svm", stage="svm_nli", f1=0.80),
                _table_entry(method="svm", stage="svm_combined", f1=0.88),
            ],
        )
        index = _make_index(tmp_root).scan(force=True)
        rows = PipelineTableAggregator(index=index).collect()

        assert len(rows) == 2
        stages = {r.stage for r in rows}
        assert stages == {"svm_nli", "svm_combined"}

    def test_multiple_runs(self, tmp_root):
        _make_pipeline_run(
            tmp_root, run_id="r1", model_name="llama",
            table_summary=[_table_entry(model="llama", f1=0.80)],
        )
        _make_pipeline_run(
            tmp_root, run_id="r2", model_name="gpt-4",
            table_summary=[_table_entry(model="gpt-4", f1=0.90)],
        )
        index = _make_index(tmp_root).scan(force=True)
        rows = PipelineTableAggregator(index=index).collect(deduplicate=False)

        assert len(rows) == 2

    def test_skips_failed_runs(self, tmp_root):
        _make_pipeline_run(
            tmp_root, run_id="r1", status="failed",
            table_summary=[_table_entry()],
        )
        index = _make_index(tmp_root).scan(force=True)
        rows = PipelineTableAggregator(index=index).collect()

        assert len(rows) == 0

    def test_includes_failed_when_requested(self, tmp_root):
        _make_pipeline_run(
            tmp_root, run_id="r1", status="failed",
            table_summary=[_table_entry()],
        )
        index = _make_index(tmp_root, include_failed=True).scan(force=True)
        rows = PipelineTableAggregator(index=index).collect(status=None)

        assert len(rows) == 1

    def test_no_summary_file(self, tmp_root):
        """Run without pipeline_summary.json is silently skipped."""
        run_dir = _make_pipeline_run(
            tmp_root, run_id="r1", table_summary=[_table_entry()],
        )
        (run_dir / "pipeline_summary.json").unlink()
        index = _make_index(tmp_root).scan(force=True)
        rows = PipelineTableAggregator(index=index).collect()

        assert len(rows) == 0

    def test_no_table_summary_key(self, tmp_root):
        """pipeline_summary.json without table_summary is skipped."""
        _make_pipeline_run(
            tmp_root, run_id="r1", table_summary=None,
            extra_metrics={"stage1": {"f1": 0.5}},
        )
        index = _make_index(tmp_root).scan(force=True)
        rows = PipelineTableAggregator(index=index).collect()

        assert len(rows) == 0

    def test_filter_by_dataset(self, tmp_root):
        _make_pipeline_run(
            tmp_root, run_id="r1", data_name="hong",
            table_summary=[_table_entry(dataset="hong")],
        )
        _make_pipeline_run(
            tmp_root, run_id="r2", data_name="argotario",
            table_summary=[_table_entry(dataset="argotario")],
        )
        index = _make_index(tmp_root).scan(force=True)
        agg = PipelineTableAggregator(index=index)

        rows = agg.collect(dataset_name="hong", deduplicate=False)
        assert len(rows) == 1
        assert rows[0].dataset == "hong"


class TestDeduplicate:
    def test_keeps_latest_by_repro_hash(self, tmp_root):
        """Two runs with same config → same repro_hash → keep latest."""
        _make_pipeline_run(
            tmp_root, run_id="r1",
            finished_at="2025-01-01T10:00:00",
            table_summary=[_table_entry(f1=0.80)],
        )
        _make_pipeline_run(
            tmp_root, run_id="r2",
            finished_at="2025-01-02T10:00:00",
            table_summary=[_table_entry(f1=0.85)],
        )
        index = _make_index(tmp_root).scan(force=True)
        rows = PipelineTableAggregator(index=index).collect(deduplicate=True)

        assert len(rows) == 1
        assert rows[0].metrics["f1"] == 0.85  # latest run

    def test_no_dedup_when_disabled(self, tmp_root):
        _make_pipeline_run(
            tmp_root, run_id="r1",
            finished_at="2025-01-01T10:00:00",
            table_summary=[_table_entry(f1=0.80)],
        )
        _make_pipeline_run(
            tmp_root, run_id="r2",
            finished_at="2025-01-02T10:00:00",
            table_summary=[_table_entry(f1=0.85)],
        )
        index = _make_index(tmp_root).scan(force=True)
        rows = PipelineTableAggregator(index=index).collect(deduplicate=False)

        assert len(rows) == 2


class TestProvenance:
    def test_row_has_provenance(self, tmp_root):
        _make_pipeline_run(
            tmp_root, run_id="r1", experiment_name="my-exp",
            finished_at="2025-01-15T10:00:00",
            table_summary=[_table_entry()],
        )
        index = _make_index(tmp_root).scan(force=True)
        rows = PipelineTableAggregator(index=index).collect()

        row = rows[0]
        assert row.run_id == "r1"
        assert row.experiment_name == "my-exp"
        assert row.repro_hash  # non-empty
        assert "my-exp" in row.run_dir

    def test_optional_fields_propagated(self, tmp_root):
        _make_pipeline_run(
            tmp_root, run_id="r1",
            table_summary=[_table_entry(mode="neutral", variant="true_to_false")],
        )
        index = _make_index(tmp_root).scan(force=True)
        rows = PipelineTableAggregator(index=index).collect()

        assert rows[0].mode == "neutral"
        assert rows[0].variant == "true_to_false"


class TestOutput:
    def test_to_dict(self, tmp_root):
        _make_pipeline_run(
            tmp_root, run_id="r1",
            table_summary=[_table_entry(f1=0.85)],
        )
        index = _make_index(tmp_root).scan(force=True)
        rows = PipelineTableAggregator(index=index).collect()
        d = rows[0].to_dict()

        assert isinstance(d, dict)
        assert d["method"] == "svm"
        assert d["metrics"]["f1"] == 0.85

    def test_collect_as_dicts(self, tmp_root):
        _make_pipeline_run(
            tmp_root, run_id="r1",
            table_summary=[_table_entry()],
        )
        index = _make_index(tmp_root).scan(force=True)
        dicts = PipelineTableAggregator(index=index).collect_as_dicts()

        assert isinstance(dicts[0], dict)

    def test_summary_string(self, tmp_root):
        _make_pipeline_run(
            tmp_root, run_id="r1",
            table_summary=[_table_entry(f1=0.85, accuracy=0.80)],
        )
        index = _make_index(tmp_root).scan(force=True)
        text = PipelineTableAggregator(index=index).summary()

        assert "svm" in text
        assert "0.850" in text

    def test_summary_empty(self, tmp_root):
        index = _make_index(tmp_root).scan(force=True)
        text = PipelineTableAggregator(index=index).summary()

        assert "No table rows" in text
