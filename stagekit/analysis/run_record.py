"""Data model for the unified run index.

RunRecord is a frozen dataclass representing one discoverable unit:
either a pipeline run directory or a single fusion model result file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class PipelineFields:
    """Pipeline-specific metadata extracted from run directory."""
    run_id: str = ""
    combo_name: str = ""
    run_group: str = ""
    pipeline_name: str = ""
    model_name: str = ""
    data_name: str = ""
    prompt_nbr: int = 1
    llm_task_name: Optional[str] = None
    llm_template: Optional[str] = None
    repro_hash: str = ""
    # Today's formula over the stored config_used.yaml; "" when the
    # snapshot carries no instrument (stored before 2026-09-24), whose
    # prompt versions today's registries would only guess.
    recomputed_repro_hash: str = ""
    completed_stages: tuple[str, ...] = ()
    total_stages: int = 0


@dataclass(frozen=True)
class RunRecord:
    """One discoverable unit in the run index: one pipeline run directory."""
    run_type: str  # "pipeline"
    run_dir: Path
    experiment_name: str
    status: str = "unknown"
    timestamp: Optional[str] = None

    pipeline: Optional[PipelineFields] = None
    # Coordinates in published/pins.yaml that point at this run. A pinned
    # run feeds a downstream pipeline and is exempt from every cleanup of
    # output/ (stagekit/run/pins.py).
    pinned: tuple[str, ...] = ()


@dataclass
class FilterSpec:
    """All filter dimensions in one object. None = don't filter.

    Accepts either a scalar or an iterable for set-based matching.
    Supports both `index.filter(seed=42)` kwargs and
    `index.filter(spec=FilterSpec(seed=42))` for composition.
    """
    run_type: Optional[str] = None
    experiment_name: Optional[str | Iterable[str]] = None
    model_name: Optional[str | Iterable[str]] = None
    dataset_name: Optional[str | Iterable[str]] = None
    prompt_nbr: Optional[int | Iterable[int]] = None
    llm_task: Optional[str | Iterable[str]] = None
    llm_template: Optional[str | Iterable[str]] = None
    pipeline_name: Optional[str | Iterable[str]] = None
    run_group: Optional[str | Iterable[str]] = None
    status: Optional[str | Iterable[str]] = None
    within_dir: Optional[Path | str] = None
    since: Optional[str | datetime] = None
    until: Optional[str | datetime] = None


# Declarative mapping: FilterSpec field → RunRecord attribute path(s).
# Each entry is (spec_field, (dotted_path, ...)). Paths are tried in
# order; the first non-None value wins. This replaces the lambda table
# in run_index.py and keeps the accessor logic with the data model.
FILTER_FIELD_MAP: dict[str, tuple[str, ...]] = {
    "run_type":        ("run_type",),
    "status":          ("status",),
    "experiment_name": ("experiment_name",),
    "model_name":      ("pipeline.model_name",),
    "dataset_name":    ("pipeline.data_name",),
    "prompt_nbr":      ("pipeline.prompt_nbr",),
    "llm_task":        ("pipeline.llm_task_name",),
    "llm_template":    ("pipeline.llm_template",),
    "pipeline_name":   ("pipeline.pipeline_name",),
    "run_group":       ("pipeline.run_group",),
}
