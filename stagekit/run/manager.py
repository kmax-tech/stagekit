# stagekit/run/manager.py
from __future__ import annotations
import json
import logging
import sys
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

from stagekit.analysis.repro_hash import instrument_snapshot, repro_hash_from_dict
from stagekit.config.classes import AppConfig
from stagekit.run.artifact_manager import ArtifactManager
from stagekit.run.context import RunContext
from stagekit.run.tracker import RunTracker
from stagekit.pipeline.data_store import PipelineDataStore
from stagekit.utils.serialization import save_json, save_yaml, normalize_config

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Writes per-stage metric files under <run_dir>/metrics/."""

    def __init__(self, metrics_dir: Path):
        self.dir = metrics_dir

    def save_stage_metrics(self, stage_name: str, metrics: dict):
        self.dir.mkdir(parents=True, exist_ok=True)
        save_json(metrics, self.dir / f"{stage_name}.json")


class ReproducibilityTracker:
    """
    Persists static experiment reproducibility artifacts.

    Called once at pipeline start (not per-stage).

    - config_used.yaml:
        Canonical, fully composed declarative configuration
        (after Hydra composition, defaults, and overrides),
        with runtime-resolved fields stripped.

    - environment.json:
        Execution environment metadata.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    # -------------------------------------------------
    def _config_to_repro_dict(self, cfg: AppConfig) -> dict:
        """Convert AppConfig to a reproducibility-safe dict."""
        cfg_dict = asdict(cfg)

        # ----------------------------------
        # Strip runtime-resolved stage fields
        # ----------------------------------
        stages = cfg_dict.get("pipeline", {}).get("stages", [])
        for stage in stages:
            stage.pop("model_cfg", None)
            stage.pop("llm_task_cfg", None)

        # Instrument snapshot: prompt versions + parser ids the stages will
        # use, taken now so config_used.yaml re-hashes identically later.
        cfg_dict["instrument"] = instrument_snapshot(cfg_dict)

        return cfg_dict

    # -------------------------------------------------
    def save_config_snapshot(self, cfg: AppConfig):
        """Save static reproducibility artifacts (config + environment). Call once."""
        if not is_dataclass(cfg):
            raise ValueError("Config must be a dataclass AppConfig instance.")

        cfg_dict = self._config_to_repro_dict(cfg)
        cfg_dict["repro_hash"] = repro_hash_from_dict(cfg_dict)
        save_yaml(cfg_dict, self.run_dir / "config_used.yaml")

        from stagekit.utils.reproducibility import _collect_git_info

        env = {
            "python_version": sys.version,
            **_collect_git_info(),
        }
        save_json(env, self.run_dir / "environment.json")


class RunManager:
    """
    Facade over experiment sub-components.  The executor calls only this
    class — never the sub-components directly.

    Architecture:

        Executor (orchestration: what runs in what order)
            │  calls semantic methods only
            ▼
        RunManager (facade: where state lives)
            │  delegates to sub-components
            ├─ ArtifactManager   — stage dirs, completion markers, artifacts
            ├─ MetricsLogger     — per-stage metric files
            ├─ ReproducibilityTracker — config + environment snapshots
            └─ RunTracker        — MLflow / W&B integration

    The facade methods (stage_dir, mark_stage_completed, etc.) look like
    trivial delegation, but they ARE the abstraction boundary.  The
    executor says "this is done" without knowing which sub-component
    tracks that.  If completion tracking later moves from
    ArtifactManager to a database or another sub-component, only this
    class's internals change — orchestration code is untouched.

    Coordination methods (save_stage_outputs, save_llm_stage_stats) are
    where the facade adds real logic: they fan out a single call to
    multiple sub-components (metrics + artifacts + tracker) atomically.

    Reproducibility lifecycle:
      1. save_config_snapshot()       — once at pipeline start
      2. record_stage_execution()     — per stage (in-memory)
      3. finish()                     — flushes model_usage.json, manifest, trackers
    """

    def __init__(self, cfg: AppConfig, ctx: RunContext):
        self.cfg = cfg
        self.ctx = ctx
        self.run_dir = ctx.run_dir
        self.run_type = ctx.run_type

        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = MetricsLogger(self.run_dir / "metrics")
        self.artifacts = ArtifactManager.for_run(self.run_dir)
        self.repro = ReproducibilityTracker(self.run_dir)

        # In-memory accumulator for per-stage model/config usage
        self._model_usage = {
            "experiment_name": cfg.meta.experiment_name,
            "run_id": cfg.meta.run_id,
            "combo_name": ctx.run_dir.parent.name,
            "stages": {},
        }
        self._usage_lock = threading.Lock()

        self.tracker = self._init_tracker()

    # -------------------------------------------------
    def _init_tracker(self):
        """Start MLflow/W&B runs if enabled in config."""
        params = self.cfg.to_dict()

        run_name = self.cfg.meta.run_name or self.run_dir.name
        tags = {
            "experiment": self.cfg.meta.experiment_name,
            "combo": self.ctx.run_dir.parent.name,
            "task": self.run_type,
            "run_type": self.run_type,
        }

        tracker = RunTracker(
            cfg=self.cfg.tracking,
            run_name=run_name,
            run_type=self.run_type,
            run_dir=self.run_dir,
            params=params,
            tags=tags,
        )
        return tracker

    # =================================================
    # Reproducibility
    # =================================================

    def save_config_snapshot(self):
        """Save static reproducibility artifacts. Call once before the stage loop."""
        self.repro.save_config_snapshot(self.cfg)

        if self.tracker:
            for file in ["config_used.yaml", "environment.json"]:
                path = self.run_dir / file
                if path.exists():
                    self.tracker.log_artifact(path, artifact_path="repro")

    def record_stage_execution(
        self,
        stage_name: str,
        stage_cfg=None,
        model_cfg=None,
        llm_task_cfg=None,
        model=None,
    ):
        """Record per-stage model/config usage in memory. Flushed at finish().

        Args:
            stage_name: Stage this entry belongs to.
            stage_cfg: The stage config, for stage_type and override source.
            model_cfg: The model config used.
            llm_task_cfg: The LLM task config used.
            model: The built model, if any. Only its backend's instrument
                digest is read — an observation about the run, which is why
                it is recorded here and not in config_used.yaml: the config
                is input and feeds the repro hash, the digest is output.
        """
        stage_type = stage_cfg.stage_type if stage_cfg else None

        model_entry = None
        if model_cfg is not None:
            model_entry = {
                "name": model_cfg.model_name,
                "type": model_cfg.model_type,
                "source": (
                    "stage_override"
                    if stage_cfg is not None and stage_cfg.model_cfg is not None
                    else "global"
                ),
            }
            # WHICH weights answered, not just what they were called: an
            # ollama tag is mutable and a vLLM alias means whatever its
            # launch command currently says. Absent for backends that
            # report none — getattr guards non-LLM models too.
            backend = getattr(model, "backend", None)
            digest = getattr(backend, "model_digest", None)
            if digest:
                model_entry["digest"] = digest
            # The serving software is part of the instrument too (ollama:
            # quantised runtime, version-dependent defaults such as the
            # context window). Only backends that probe it carry the field.
            server_version = getattr(backend, "server_version", None)
            if server_version:
                model_entry["server_version"] = server_version

        llm_task_entry = None
        if llm_task_cfg is not None:
            llm_task_entry = {
                "task_name": llm_task_cfg.task_name,
                "template": llm_task_cfg.template,
                "parser": llm_task_cfg.parser,
                "source": (
                    "stage_override"
                    if stage_cfg is not None and stage_cfg.llm_task_cfg is not None
                    else "global"
                ),
            }

        stage_config_entry = None
        if stage_cfg is not None and stage_cfg.config_override:
            stage_config_entry = normalize_config(stage_cfg.config_override)

        with self._usage_lock:
            self._model_usage["stages"][stage_name] = {
                "stage_type": stage_type,
                "stage_config": stage_config_entry,
                "model": model_entry,
                "llm_task": llm_task_entry,
            }

    def _flush_model_usage(self):
        """Write accumulated model_usage to disk."""
        model_usage_path = self.run_dir / "model_usage.json"
        with self._usage_lock:
            save_json(self._model_usage, model_usage_path)

        if self.tracker:
            self.tracker.log_artifact(model_usage_path, artifact_path="repro")

    # =================================================
    # Lifecycle
    # =================================================

    def finish(self):
        self._flush_model_usage()
        self._write_run_manifest()
        self._write_run_artifacts()

        if self.tracker:
            self.tracker.finish()


    # =================================================
    # Stage outputs
    # =================================================

    def save_stage_outputs(self, stage_name: str, metrics: dict, artifacts: dict):
        self.metrics.save_stage_metrics(stage_name, metrics)
        stage_dir = self.artifacts.save_stage_artifacts(stage_name, artifacts)

        if self.tracker:
            self.tracker.log_metrics(metrics, prefix=f"{stage_name}/")
            if stage_dir:
                self.tracker.log_artifact(
                    stage_dir, artifact_path=f"artifacts/{stage_name}"
                )

    def load_stage_artifacts(self, stage_name: str) -> dict:
        return self.artifacts.load_stage_artifacts(stage_name)

    # =================================================
    # Stage lifecycle (delegates to ArtifactManager)
    # =================================================

    def stage_dir(self, stage_name: str) -> Path:
        return self.artifacts.stage_dir(stage_name)

    def mark_stage_completed(self, stage_name: str) -> None:
        self.artifacts.mark_stage_completed(stage_name)

    # =================================================
    # Pipeline data
    # =================================================

    def save_pipeline_data(self, pipeline_data: PipelineDataStore):
        if not self.cfg.meta.enable_pipeline_outputs:
            return
        if not pipeline_data._dirty:
            return
        if pipeline_data.is_empty():
            return
        save_json(pipeline_data.to_dict(), self.run_dir / "pipeline_outputs.json")
        pipeline_data._dirty = False

    # =================================================
    # LLM diagnostics
    # =================================================

    def save_llm_stage_stats(self, stage_name: str, llm_stats):
        summary = llm_stats.summary()
        if summary.get("total_calls", 0) == 0:
            return

        path = self.run_dir / "llm_stats" / f"{stage_name}.json"
        path.parent.mkdir(exist_ok=True)
        save_json(summary, path)

        if self.tracker:
            self.tracker.log_artifact(path, artifact_path="llm_stats")

    def save_parse_diagnostics(self, stage_name: str, parse_diagnostics):

        # --------------------------------------------------
        # GLOBAL ON/OFF SWITCH
        # --------------------------------------------------
        if not self.cfg.meta.store_parse_diagnostics:
            return

        if not parse_diagnostics:
            return

        # ------------------------------------
        # STORAGE POLICY (filter -> cap)
        # ------------------------------------
        if self.cfg.meta.store_only_failed_parses:
            parse_diagnostics = [
                d for d in parse_diagnostics if d.parsed_output is None
            ]

        if not parse_diagnostics:
            return

        max_diags = self.cfg.meta.max_parse_diagnostics_per_stage
        if max_diags is not None:
            parse_diagnostics = parse_diagnostics[:max_diags]

        if not parse_diagnostics:
            return

        # ------------------------------------
        # PERSIST
        # ------------------------------------
        diag_dir = self.run_dir / "diagnostics" / "parse"
        diag_dir.mkdir(parents=True, exist_ok=True)

        path = diag_dir / f"{stage_name}.jsonl"

        with open(path, "w", encoding="utf-8") as f:
            for d in parse_diagnostics:
                f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")

        if self.tracker:
            self.tracker.log_artifact(path, artifact_path="diagnostics/parse")

    # =================================================
    # Run manifest
    # =================================================

    def _write_run_manifest(self):
        """
        Write a lightweight run manifest that indexes
        all top-level artifacts produced by this run.
        """

        manifest = {
            "run_id": self.cfg.meta.run_id,
            "experiment_name": self.cfg.meta.experiment_name,
            "combo_name": self.ctx.run_dir.parent.name,
            "run_type": self.run_type,
            "paths": {},
        }

        # Known, canonical files (only add if they exist)
        candidates = {
            "model_usage": "model_usage.json",
            "pipeline_outputs": "pipeline_outputs.json",
            "pipeline_summary": "pipeline_summary.json",
            "config_used": "config_used.yaml",
            "environment": "environment.json",
        }

        for key, filename in candidates.items():
            path = self.run_dir / filename
            if path.exists():
                manifest["paths"][key] = filename

        # Optional directories
        if (self.run_dir / "stage_outputs").exists():
            manifest["paths"]["stage_outputs_dir"] = "stage_outputs"

        # Write manifest
        manifest_path = self.run_dir / "run_manifest.json"
        save_json(manifest, manifest_path)

        # Track it if tracking is enabled
        if self.tracker:
            self.tracker.log_artifact(manifest_path, artifact_path="repro")

    def _write_run_artifacts(self):
        """Write run_artifacts.json — a machine-readable manifest of all
        artifacts produced by this run (part of the run ledger)."""
        from stagekit.utils.serialization import compute_file_hash, load_json

        stage_outputs_dir = self.run_dir / "stage_outputs"
        if not stage_outputs_dir.is_dir():
            return

        with self._usage_lock:
            stages_usage = self._model_usage.get("stages", {})

        entries = []
        for stage_dir in sorted(stage_outputs_dir.iterdir()):
            if not stage_dir.is_dir():
                continue
            if not (stage_dir / "_completed").exists():
                continue

            keys_path = stage_dir / "_keys.json"
            if not keys_path.exists():
                continue
            key_map = load_json(keys_path)
            if not isinstance(key_map, dict):
                continue

            stage_name = stage_dir.name
            stage_info = stages_usage.get(stage_name, {})
            model_entry = stage_info.get("model") or {}
            llm_task_entry = stage_info.get("llm_task") or {}

            for safe_key, artifact_key in key_map.items():
                file_path = stage_dir / f"{safe_key}.json"
                if not file_path.exists():
                    file_path = stage_dir / f"{safe_key}.txt"
                    if not file_path.exists():
                        continue

                parts = artifact_key.split(".", 1)
                entries.append({
                    "artifact_key": artifact_key,
                    "stage_name": stage_name,
                    "field_name": parts[1] if len(parts) > 1 else parts[0],
                    "file_path": str(file_path.relative_to(self.run_dir)),
                    "content_hash": compute_file_hash(file_path),
                    "model_name": model_entry.get("name", ""),
                    "llm_template": llm_task_entry.get("template", ""),
                })

        cfg = self.cfg
        manifest = {
            "version": 1,
            "run_id": cfg.meta.run_id,
            "run_meta": {
                "data_name": cfg.data.data_name,
                "pipeline_name": cfg.pipeline.name,
                "pipeline_mode": getattr(cfg.pipeline, "mode", ""),
                "prompt_nbr": cfg.meta.prompt_nbr,
            },
            "artifacts": entries,
        }

        save_json(manifest, self.run_dir / "run_artifacts.json")

    def write_pipeline_status(
        self,
        stage_names: list[str],
        *,
        status: str = "success",
        total_stages: int | None = None,
        error: str | None = None,
    ) -> None:
        """Write pipeline_status.json — the scanner's success/failure signal.

        Args:
            stage_names: Stages that actually completed.
            status: "success" or "failed"; anything but "success" makes the
                run visible only via ``include_failed``.
            total_stages: Planned stage count; defaults to
                ``len(stage_names)``, which is only correct for a
                successful run.
            error: For a failed run, ``"ExcType: message"`` of the cause.
        """
        payload = {
            "status": status,
            "completed_stages": stage_names,
            "total_stages": len(stage_names) if total_stages is None else total_stages,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        if error is not None:
            payload["error"] = error

        path = self.run_dir / "pipeline_status.json"
        save_json(payload, path)
