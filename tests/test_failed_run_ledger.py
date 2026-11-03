"""A crashed pipeline run must still enter the run ledger.

Covers the failure path added for audit finding A3 (docs/audit-2026-09-02.md):
without a pipeline_status.json and run_manifest.json, a failed run is
invisible to the scanner and ``include_failed`` can never match anything.
"""

import json
import threading

from stagekit.analysis.scanner import scan_pipeline_runs
from stagekit.main_pipeline import _finalize_failed_run
from stagekit.run.manager import RunManager


class _LedgerOnlyManager(RunManager):
    """RunManager stripped to the ledger pieces this test exercises."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.tracker = None
        self._model_usage = {"stages": {}}
        self._usage_lock = threading.Lock()
        self.finish_called = False

    def finish(self):
        self.finish_called = True


class _StubExecutor:
    topological_order = ["s1", "s2", "s3"]
    completed_stages = ["s1"]


class TestWritePipelineStatusFailed:
    def test_failed_status_carries_error_and_planned_total(self, tmp_path):
        mgr = _LedgerOnlyManager(tmp_path)
        mgr.write_pipeline_status(
            ["s1"], status="failed", total_stages=3, error="ValueError: boom",
        )

        payload = json.loads((tmp_path / "pipeline_status.json").read_text())
        assert payload["status"] == "failed"
        assert payload["completed_stages"] == ["s1"]
        assert payload["total_stages"] == 3
        assert payload["error"] == "ValueError: boom"

    def test_success_payload_unchanged(self, tmp_path):
        """The success path must keep its exact pre-A3 shape (no error key)."""
        mgr = _LedgerOnlyManager(tmp_path)
        mgr.write_pipeline_status(["s1", "s2"])

        payload = json.loads((tmp_path / "pipeline_status.json").read_text())
        assert payload["status"] == "success"
        assert payload["total_stages"] == 2
        assert "error" not in payload


class TestFinalizeFailedRun:
    def _run_dir(self, tmp_path):
        run_dir = tmp_path / "pipeline" / "exp_run"
        run_dir.mkdir(parents=True)
        return run_dir

    def test_writes_status_and_finishes(self, tmp_path):
        run_dir = self._run_dir(tmp_path)
        mgr = _LedgerOnlyManager(run_dir)

        _finalize_failed_run(mgr, _StubExecutor(), ValueError("stage s2 died"))

        payload = json.loads((run_dir / "pipeline_status.json").read_text())
        assert payload["status"] == "failed"
        assert payload["completed_stages"] == ["s1"]
        assert payload["total_stages"] == 3
        assert payload["error"] == "ValueError: stage s2 died"
        assert mgr.finish_called

    def test_never_raises(self, tmp_path):
        """Recording the failure must not mask the original exception."""
        mgr = _LedgerOnlyManager(self._run_dir(tmp_path))
        mgr.write_pipeline_status = None  # type: ignore[assignment] — force a TypeError inside

        _finalize_failed_run(mgr, _StubExecutor(), ValueError("boom"))  # must not raise

    def test_failed_run_visible_to_scanner_via_include_failed(self, tmp_path):
        """End-to-end: the written files make the run discoverable."""
        run_dir = self._run_dir(tmp_path)
        mgr = _LedgerOnlyManager(run_dir)
        # _finalize_failed_run calls the real finish() in production; here the
        # manifest is written directly since _LedgerOnlyManager stubs finish().
        _finalize_failed_run(mgr, _StubExecutor(), RuntimeError("boom"))
        (run_dir / "run_manifest.json").write_text(json.dumps({
            "run_id": "r1", "experiment_name": "exp", "combo_name": "c",
            "run_type": "pipeline",
        }), encoding="utf-8")

        assert scan_pipeline_runs(tmp_path / "pipeline") == []
        failed = scan_pipeline_runs(tmp_path / "pipeline", include_failed=True)
        assert len(failed) == 1
        assert failed[0].status == "failed"


class TestRealFinishWritesManifest:
    """The production _write_run_manifest works even after a crash, when
    pipeline_summary.json does not exist — it is the scanner's discovery
    anchor, so it must never depend on success-only files."""

    def test_manifest_written_without_summary(self, tmp_path):
        run_dir = tmp_path / "group" / "run"
        run_dir.mkdir(parents=True)
        mgr = _LedgerOnlyManager(run_dir)

        class _Meta:
            run_id = "r1"
            experiment_name = "exp"

        class _Cfg:
            meta = _Meta()

        class _Ctx:
            pass

        _Ctx.run_dir = run_dir
        mgr.cfg = _Cfg()
        mgr.ctx = _Ctx()
        mgr.run_type = "pipeline"
        RunManager._write_run_manifest(mgr)

        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        assert manifest["run_type"] == "pipeline"
        assert manifest["experiment_name"] == "exp"
        assert manifest["paths"] == {}  # nothing produced — still a manifest
