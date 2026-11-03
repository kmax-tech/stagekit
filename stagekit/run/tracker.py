# stagekit/run/tracker.py
from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, cast, Literal

from stagekit.config.classes import TrackingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _flatten_dict(
    data: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def _sanitize_value(val: Any) -> Any:
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    if isinstance(val, Path):
        return str(val)
    return str(val)


VALID_WANDB_MODES = {"online", "offline", "disabled", "shared"}


# ---------------------------------------------------------------------
# RunTracker
# ---------------------------------------------------------------------
class RunTracker:
    """
    Lightweight bridge to optionally send metrics, params,
    and artifacts to MLflow and/or Weights & Biases.
    """

    def __init__(
        self,
        cfg: TrackingConfig,
        run_name: str,
        run_type: str,
        run_dir: Path,
        params: Optional[dict] = None,
        tags: Optional[dict] = None,
    ):
        self.cfg = cfg
        self.run_name = run_name
        self.run_type = run_type
        self.run_dir = run_dir

        self.params = params or {}
        self.tags = tags or {}
        self.tags.setdefault("run_type", run_type)
        self.tags.setdefault("run_name", run_name)

        self._mlflow = None
        self._mlflow_active = False

        self._wandb = None
        self._wandb_run = None
        self._wandb_active = False

        self._init_mlflow()
        self._init_wandb()

        if self.params:
            self.log_params(self.params)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_mlflow(self):
        if not self.cfg.enable_mlflow:
            return

        try:
            import mlflow
        except ImportError:
            logger.warning("MLflow not installed; skipping MLflow tracking.")
            return

        tracking_uri = self.cfg.mlflow_tracking_uri or str(
            Path(self.run_dir).parent / "mlruns"
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow_experiment)

        self._mlflow = mlflow
        self._mlflow_active = True

        mlflow.start_run(run_name=self.run_name)
        mlflow.set_tags(self.tags)

    def _init_wandb(self):
        if not self.cfg.enable_wandb:
            return

        try:
            import wandb
        except ImportError:
            logger.warning("Weights & Biases not installed; skipping W&B tracking.")
            return

        self._wandb = wandb

        tags = [f"{k}:{v}" for k, v in self.tags.items()]

        mode_str = self.cfg.wandb_mode
        if mode_str not in VALID_WANDB_MODES:
            raise ValueError(
                f"Invalid wandb_mode='{mode_str}'. "
                f"Expected one of {sorted(VALID_WANDB_MODES)}"
            )

        mode = cast(
            Literal["online", "offline", "disabled", "shared"],
            mode_str,
        )

        self._wandb_run = wandb.init(
            project=self.cfg.wandb_project,
            entity=self.cfg.wandb_entity,
            name=self.run_name,
            tags=tags,
            dir=str(self.run_dir),
            mode=mode,
        )
        self._wandb_active = True

    # ------------------------------------------------------------------
    # Params
    # ------------------------------------------------------------------
    def log_params(self, params: dict):
        if is_dataclass(params):
            params = asdict(params)

        flat = {k: _sanitize_value(v) for k, v in _flatten_dict(params).items()}

        if self._mlflow_active and self._mlflow is not None:
            self._mlflow.log_params(flat)

        if self._wandb_active and self._wandb_run is not None:
            self._wandb_run.config.update(flat, allow_val_change=True)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def log_metric(self, name: str, value: Any, step: Optional[int] = None):
        """Log one scalar metric to the active trackers.

        Non-numeric values are skipped with a debug line — an explicit
        tolerance, not an accident: StageResult.metrics legitimately
        carries structured entries (e.g. ``skipped_by_reason`` dicts) that
        belong to stage_metrics.json; feeding them to ``mlflow.log_metric``
        would crash the run at persist time.
        """
        if not isinstance(value, (int, float)):  # bool passes: it IS an int
            logger.debug(
                "log_metric: skipping non-scalar metric %r (%s)",
                name, type(value).__name__,
            )
            return

        if self._mlflow_active and self._mlflow is not None:
            if step is not None:
                self._mlflow.log_metric(name, value, step=step)
            else:
                self._mlflow.log_metric(name, value)

        if self._wandb_active and self._wandb_run is not None:
            # step is a wandb.log() keyword, not a payload key — "_step"
            # in the payload is undocumented and silently ignored.
            if step is not None:
                self._wandb_run.log({name: value}, step=step)
            else:
                self._wandb_run.log({name: value})

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
    ):
        for k, v in metrics.items():
            name = f"{prefix}{k}" if prefix else k
            self.log_metric(name, v, step=step)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    def log_artifact(self, path: Path, artifact_path: Optional[str] = None):
        if self._mlflow_active and self._mlflow is not None:
            if path.is_dir():
                self._mlflow.log_artifacts(str(path), artifact_path=artifact_path)
            else:
                self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

        if self._wandb_active and self._wandb_run is not None:
            self._wandb_run.save(str(path))

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------
    def finish(self):
        if self._mlflow_active and self._mlflow is not None:
            self._mlflow.end_run()
            self._mlflow_active = False

        if self._wandb_active and self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_active = False
