"""Aggregate table_summary entries from pipeline runs via RunIndex.

Canonical path for building evaluation tables from pipeline results:
    RunIndex (discover runs) → pipeline_summary.json → table_summary

Role split: comparing runs/configurations happens exclusively here
(output/ + RunIndex, keyed by repro_hash).  ``published/pins.yaml`` is
the publication layer for pipeline inputs (``preload_artifacts`` →
``pin:<coordinate>``) and is never a comparison source; see stagekit/run/pins.py.

Usage::

    from stagekit.analysis.table_aggregator import PipelineTableAggregator

    agg = PipelineTableAggregator()
    rows = agg.collect(experiment_name="hong-counterfactual-svm-llama")
    for row in rows:
        print(row.method, row.metrics["f1"])

    # Or with a FilterSpec:
    rows = agg.collect(spec=FilterSpec(
        dataset_name="hong", pipeline_name="counterfactual/classify-svm",
    ))
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from stagekit.analysis.run_index import RunIndex
from stagekit.analysis.run_record import FilterSpec, RunRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TableRow:
    """One normalized table entry with run provenance."""

    # From table_summary
    method: str
    stage: str
    dataset: str
    model: Optional[str]
    eval_mode: Optional[str]
    metrics: dict[str, float]

    # Optional table_summary fields
    mode: Optional[str] = None
    variant: Optional[str] = None
    source_model: Optional[str] = None
    n_folds: Optional[int] = None

    # Run provenance (added by aggregator)
    run_id: str = ""
    run_dir: str = ""
    repro_hash: str = ""
    experiment_name: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "stage": self.stage,
            "dataset": self.dataset,
            "model": self.model,
            "eval_mode": self.eval_mode,
            "metrics": dict(self.metrics),
            "mode": self.mode,
            "variant": self.variant,
            "source_model": self.source_model,
            "n_folds": self.n_folds,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "repro_hash": self.repro_hash,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
        }


class PipelineTableAggregator:
    """Collect table_summary entries from pipeline runs discovered by RunIndex.

    Workflow:
        1. RunIndex.filter() to find matching pipeline runs
        2. Read pipeline_summary.json from each run directory
        3. Extract table_summary entries, enrich with run provenance
        4. Deduplicate by repro_hash (keep latest)
    """

    def __init__(self, *, index: Optional[RunIndex] = None):
        self._index = index

    def _get_index(self) -> RunIndex:
        if self._index is None:
            self._index = RunIndex().scan()
        return self._index

    def collect(
        self,
        spec: Optional[FilterSpec] = None,
        *,
        deduplicate: bool = True,
        **kwargs,
    ) -> list[TableRow]:
        """Collect table_summary rows from runs matching the filter.

        Filters are passed to RunIndex.filter(). Only pipeline runs with
        status="success" are included by default — pass ``status=None``
        to include all.

        When *deduplicate* is True (default), runs with the same
        repro_hash keep only the latest entry.
        """
        if spec is None:
            kwargs.setdefault("run_type", "pipeline")
            kwargs.setdefault("status", "success")
            spec = FilterSpec(**kwargs)

        index = self._get_index()
        runs = index.filter(spec)

        rows: list[TableRow] = []
        for run in runs:
            run_rows = self._extract_from_run(run)
            rows.extend(run_rows)

        if deduplicate:
            rows = self._deduplicate(rows)

        logger.info(
            "Collected %d table rows from %d runs",
            len(rows), len(runs),
        )
        return rows

    def _extract_from_run(self, run: RunRecord) -> list[TableRow]:
        """Read pipeline_summary.json and extract table_summary entries."""
        summary_path = run.run_dir / "pipeline_summary.json"
        if not summary_path.exists():
            logger.debug("No pipeline_summary.json in %s", run.run_dir)
            return []

        try:
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", summary_path, e)
            return []

        table_entries = summary.get("table_summary", [])
        if not table_entries:
            return []

        pipeline = run.pipeline
        run_id = pipeline.run_id if pipeline else ""
        repro_hash = pipeline.repro_hash if pipeline else ""

        rows = []
        for entry in table_entries:
            row = TableRow(
                method=entry["method"],
                stage=entry["stage"],
                dataset=entry["dataset"],
                model=entry.get("model"),
                eval_mode=entry.get("eval_mode"),
                metrics=entry.get("metrics", {}),
                mode=entry.get("mode"),
                variant=entry.get("variant"),
                source_model=entry.get("source_model"),
                n_folds=entry.get("n_folds"),
                run_id=run_id,
                run_dir=str(run.run_dir),
                repro_hash=repro_hash,
                experiment_name=run.experiment_name,
                timestamp=run.timestamp or "",
            )
            rows.append(row)

        return rows

    @staticmethod
    def _deduplicate(rows: list[TableRow]) -> list[TableRow]:
        """Keep only the latest row per (repro_hash, stage) pair.

        Runs with the same repro_hash are equivalent configurations;
        only the newest result matters.
        """
        best: dict[tuple[str, str], TableRow] = {}
        for row in rows:
            if not row.repro_hash:
                # No repro_hash → can't deduplicate, keep all
                best[("", row.run_id + row.stage)] = row
                continue

            key = (row.repro_hash, row.stage)
            existing = best.get(key)
            if existing is None or row.timestamp > existing.timestamp:
                best[key] = row

        return list(best.values())

    # --------------------------------------------------
    # Convenience
    # --------------------------------------------------

    def collect_as_dicts(
        self,
        spec: Optional[FilterSpec] = None,
        **kwargs,
    ) -> list[dict]:
        """Like :meth:`collect` but returns plain dicts."""
        return [r.to_dict() for r in self.collect(spec=spec, **kwargs)]

    def summary(
        self,
        spec: Optional[FilterSpec] = None,
        **kwargs,
    ) -> str:
        """Human-readable summary of collected table rows."""
        rows = self.collect(spec=spec, **kwargs)
        if not rows:
            return "No table rows found."

        lines = [f"{'method':<20} {'dataset':<12} {'mode':<16} {'f1':>6} {'acc':>6}  run_id"]
        lines.append("-" * 80)
        for r in sorted(rows, key=lambda x: (x.dataset, x.method, x.mode or "")):
            f1 = r.metrics.get("f1", float("nan"))
            acc = r.metrics.get("accuracy", float("nan"))
            lines.append(
                f"{r.method:<20} {r.dataset:<12} {(r.mode or '-'):<16} "
                f"{f1:>6.3f} {acc:>6.3f}  {r.run_id[:8]}"
            )
        return "\n".join(lines)
