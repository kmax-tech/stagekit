"""Filesystem scanners for pipeline runs.

Each scanner walks a directory tree and returns a list of RunRecords.
The RunIndex orchestrates these scanners and caches the results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from stagekit.analysis.repro_hash import repro_hash_from_dict
from stagekit.analysis.run_record import RunRecord, PipelineFields
from stagekit.utils.serialization import load_json_safe, load_yaml_safe

logger = logging.getLogger(__name__)



# --------------------------------------------------
# Common scan loop
# --------------------------------------------------

def _scan_root(
    root: Path,
    glob: str,
    loader: Callable[[Path], Optional[RunRecord]],
    *,
    skip_names: frozenset[str] = frozenset(),
) -> list[RunRecord]:
    """Walk *root*, apply *loader* to every match, collect non-None results."""
    if not root.exists():
        return []

    records: list[RunRecord] = []
    for path in root.rglob(glob):
        if path.name in skip_names:
            continue
        try:
            record = loader(path)
            if record is not None:
                records.append(record)
        except (OSError, ValueError, KeyError) as exc:
            # WARNING, not debug: a run silently missing from the index
            # falls out of every comparison — the operator must see it.
            logger.warning("Skipping unreadable run %s: %s", path, exc)
    return records


# --------------------------------------------------
# Pipeline scanner
# --------------------------------------------------

def scan_pipeline_runs(
    root: Path,
    *,
    include_failed: bool = False,
    pinned: Optional[dict[Path, tuple[str, ...]]] = None,
) -> list[RunRecord]:
    """Discover pipeline runs under *root*.

    Walks recursively, stops at directories containing ``run_manifest.json``.

    Args:
        pinned: Resolved run directory → coordinates pinned to it
            (``pins.pinned_coordinates_by_run_dir``); fills ``RunRecord.pinned``.
    """
    pinned = pinned or {}

    def loader(manifest_path: Path) -> Optional[RunRecord]:
        run_dir = manifest_path.parent
        return _load_pipeline_record(
            run_dir, include_failed=include_failed,
            pinned=pinned.get(run_dir.resolve(), ()),
        )

    return _scan_root(root, "run_manifest.json", loader)


def _load_pipeline_record(
    run_dir: Path,
    *,
    include_failed: bool = False,
    pinned: tuple[str, ...] = (),
) -> Optional[RunRecord]:
    """Build a RunRecord from one pipeline run directory."""
    manifest = load_json_safe(run_dir / "run_manifest.json")
    if manifest is None:
        return None

    # --- pipeline_status.json ---
    status_data = load_json_safe(run_dir / "pipeline_status.json") or {}
    status = status_data.get("status", "unknown")

    if not include_failed and status != "success":
        return None

    # --- model_usage.json (first stage with model info) ---
    model_name = ""
    llm_task_name: Optional[str] = None
    llm_template: Optional[str] = None

    usage = load_json_safe(run_dir / "model_usage.json")
    if usage:
        for stage_info in usage.get("stages", {}).values():
            if not model_name and (m := stage_info.get("model")):
                model_name = m.get("name", "")
            if llm_task_name is None and (lt := stage_info.get("llm_task")):
                llm_task_name = lt.get("task_name")
                llm_template = lt.get("template")

    # --- config_used.yaml (fallback for remaining fields) ---
    cfg = load_yaml_safe(run_dir / "config_used.yaml")

    # Repro hash: use stored value if present, otherwise compute from dict
    repro_hash = cfg.get("repro_hash", "") if cfg else ""
    if not repro_hash and cfg:
        repro_hash = repro_hash_from_dict(cfg)

    return RunRecord(
        run_type=manifest.get("run_type", "pipeline"),
        run_dir=run_dir,
        experiment_name=manifest.get("experiment_name", ""),
        status=status,
        timestamp=status_data.get("finished_at"),
        pipeline=PipelineFields(
            run_id=manifest.get("run_id", ""),
            combo_name=manifest.get("combo_name", ""),
            run_group=cfg.get("meta", {}).get("run_group", ""),
            pipeline_name=cfg.get("pipeline", {}).get("name", ""),
            model_name=model_name,
            data_name=cfg.get("data", {}).get("data_name", ""),
            prompt_nbr=cfg.get("meta", {}).get("prompt_nbr", 1),
            llm_task_name=llm_task_name,
            llm_template=llm_template,
            repro_hash=repro_hash,
            completed_stages=tuple(status_data.get("completed_stages", [])),
            total_stages=status_data.get("total_stages", 0),
        ),
        pinned=tuple(pinned),
    )
