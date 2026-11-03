"""Unified run index for discovering, caching, and querying experiment runs.

Replaces 6 parallel discovery patterns with a single scan-once, query-many API.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from stagekit.analysis.run_record import RunRecord, FilterSpec, PipelineFields, FILTER_FIELD_MAP
from stagekit.analysis.scanner import scan_pipeline_runs

logger = logging.getLogger(__name__)

# 3: RunRecord.pinned + pins.yaml in the cache key (2026-10-08)
CATALOG_VERSION = 3


class RunIndex:
    """Scan once, query many.

    Discovers pipeline runs, caches the catalog to disk, and serves
    filtered queries without re-walking the filesystem.

    Usage::

        index = RunIndex().scan()
        runs = index.filter(experiment_name="hong-counterfactual-svm")
        latest = index.latest_run("my-experiment", run_group="default")
    """

    def __init__(
        self,
        *,
        pipeline_root: Optional[Path] = None,
        catalog_path: Optional[Path] = None,
        include_failed: bool = False,
        pins_path: Optional[Path] = None,
    ):
        import stagekit.config.paths as s

        self._pipeline_root = pipeline_root or (s.PROJECT_ROOT / "output" / "pipeline")
        self._catalog_path = catalog_path or (s.PROJECT_ROOT / "output" / "run_catalog.json")
        self._include_failed = include_failed
        # published/pins.yaml: which runs feed downstream pipelines
        # (RunRecord.pinned). Part of the cache key — a new pin must show.
        self._pins_path = pins_path or s.PINS_FILE
        self._records: list[RunRecord] = []

    # --------------------------------------------------
    # Scan
    # --------------------------------------------------

    def scan(self, force: bool = False) -> RunIndex:
        """Populate the index. Loads from cache unless *force* or cache is stale."""
        if not force and self._load_cache():
            return self
        return self.refresh()

    def refresh(self) -> RunIndex:
        """Force a full filesystem scan, bypassing cache."""
        from stagekit.run.pins import pinned_coordinates_by_run_dir

        pipeline = scan_pipeline_runs(
            self._pipeline_root, include_failed=self._include_failed,
            pinned=pinned_coordinates_by_run_dir(pins_path=self._pins_path),
        )
        self._records = pipeline
        self._write_cache()
        logger.info("RunIndex scanned: %d pipeline runs", len(pipeline))
        return self

    # --------------------------------------------------
    # Accessors
    # --------------------------------------------------

    def all(self) -> list[RunRecord]:
        return list(self._records)

    def pipeline_runs(self) -> list[RunRecord]:
        return [r for r in self._records if r.run_type == "pipeline"]

    # --------------------------------------------------
    # Filter
    # --------------------------------------------------

    def filter(self, spec: Optional[FilterSpec] = None, **kwargs) -> list[RunRecord]:
        """Return records matching all filter criteria.

        Accepts either a ``FilterSpec`` object or keyword arguments
        (which are used to construct one). All filters combine with AND.
        """
        if spec is None:
            spec = FilterSpec(**kwargs)

        results = self._records

        # Set-membership filters (driven by FILTER_FIELD_MAP)
        for attr, paths in FILTER_FIELD_MAP.items():
            value = getattr(spec, attr)
            if value is not None:
                vals = _as_set(value)
                results = [r for r in results if _extract(r, paths) in vals]

        # Path containment
        if spec.within_dir is not None:
            base = Path(spec.within_dir).resolve()
            results = [
                r for r in results
                if base in r.run_dir.resolve().parents or r.run_dir.resolve() == base
            ]

        # Datetime range
        if spec.since is not None:
            cutoff = _parse_datetime(spec.since)
            results = [
                r for r in results
                if r.timestamp and _parse_datetime(r.timestamp) >= cutoff
            ]

        if spec.until is not None:
            cutoff = _parse_datetime(spec.until)
            results = [
                r for r in results
                if r.timestamp and _parse_datetime(r.timestamp) <= cutoff
            ]

        return results

    # --------------------------------------------------
    # Convenience queries
    # --------------------------------------------------

    def has_equivalent(self, cfg) -> bool:
        """Check if a completed pipeline run matching *cfg* exists.

        Compares the canonical reproducibility hash of *cfg* against
        hashes stored in (or computed from) existing run records. A run
        whose hash was made by another formula version never matches
        (``formula_versions`` counts them, so a caller can say why).
        """
        from stagekit.analysis.repro_hash import compute_repro_hash

        target = compute_repro_hash(cfg)
        return any(
            r.run_type == "pipeline"
            and r.status == "success"
            and r.pipeline is not None
            and r.pipeline.repro_hash == target
            for r in self._records
        )

    def formula_versions(self) -> dict[int, int]:
        """Successful pipeline runs per repro-hash formula version.

        Runs of another version than today's ``REPRO_HASH_VERSION`` cannot
        be found equivalent by ``has_equivalent``; this says how many.
        Runs without a hash are not counted.
        """
        from collections import Counter

        from stagekit.analysis.repro_hash import repro_hash_version

        return dict(Counter(
            repro_hash_version(r.pipeline.repro_hash)
            for r in self._records
            if r.run_type == "pipeline" and r.status == "success"
            and r.pipeline is not None and r.pipeline.repro_hash
        ))

    def latest_run(
        self,
        experiment_name: str,
        run_group: Optional[str] = None,
        *,
        status: Optional[str] = "success",
    ) -> Optional[RunRecord]:
        """Find the most recent pipeline run matching *experiment_name*.

        *status* defaults to ``"success"``; pass ``None`` to match any status
        (useful for resume).
        """
        candidates = [
            r for r in self._records
            if r.run_type == "pipeline"
            and r.experiment_name == experiment_name
            and (status is None or r.status == status)
        ]

        if run_group is not None:
            candidates = [
                r for r in candidates
                if r.pipeline and r.pipeline.run_group == run_group
            ]

        if not candidates:
            return None

        # Records without a timestamp sort FIRST (empty string), never
        # ahead of dated ones — mixing ISO timestamps with directory
        # names lexicographically let a timestamp-less record win always
        # (letters compare greater than digits).
        return max(
            candidates,
            key=lambda r: r.timestamp or "",
        )

    # --------------------------------------------------
    # Cache
    # --------------------------------------------------

    def _count_discoverable(self) -> dict[str, dict]:
        """Quick count and newest mtime of discoverable entries."""
        stats: dict[str, dict] = {}

        if self._pipeline_root.exists():
            manifests = list(self._pipeline_root.rglob("run_manifest.json"))
            newest = max((p.stat().st_mtime for p in manifests), default=0.0)
            stats["pipeline"] = {"count": len(manifests), "newest_mtime": newest}
        else:
            stats["pipeline"] = {"count": 0, "newest_mtime": 0.0}

        if self._pins_path.exists():
            st = self._pins_path.stat()
            stats["pins"] = {"count": st.st_size, "newest_mtime": st.st_mtime}
        else:
            stats["pins"] = {"count": 0, "newest_mtime": 0.0}

        return stats

    def _load_cache(self) -> bool:
        """Load catalog from disk if valid. Returns True on cache hit."""
        if not self._catalog_path.exists():
            return False

        try:
            with open(self._catalog_path, encoding="utf-8") as f:
                catalog = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False

        if catalog.get("version") != CATALOG_VERSION:
            return False

        stored_roots = catalog.get("scan_roots", {})
        current_stats = self._count_discoverable()

        for key, root in (("pipeline", self._pipeline_root), ("pins", self._pins_path)):
            stored = stored_roots.get(key, {})
            current = current_stats.get(key, {"count": 0, "newest_mtime": 0.0})
            # Two RunIndex instances with different roots share one catalog
            # file — a catalog written for another root must never be
            # served (count/mtime alone can coincide).
            if stored.get("path") != str(root):
                logger.debug(
                    "Cache stale: %s root changed (%s -> %s)",
                    key, stored.get("path"), self._pipeline_root,
                )
                return False
            if stored.get("record_count", -1) != current["count"]:
                logger.debug("Cache stale: %s count changed", key)
                return False
            if stored.get("newest_mtime", 0.0) != current["newest_mtime"]:
                logger.debug("Cache stale: %s mtime changed", key)
                return False

        self._records = [
            _record_from_dict(d) for d in catalog.get("records", [])
        ]
        logger.debug("Loaded %d records from cache", len(self._records))
        return True

    def _write_cache(self) -> None:
        """Persist catalog to disk."""
        stats = self._count_discoverable()

        catalog = {
            "version": CATALOG_VERSION,
            "scan_roots": {
                "pipeline": {
                    "path": str(self._pipeline_root),
                    "record_count": stats["pipeline"]["count"],
                    "newest_mtime": stats["pipeline"]["newest_mtime"],
                },
                # size + mtime of pins.yaml: a new pin invalidates the cache
                "pins": {
                    "path": str(self._pins_path),
                    "record_count": stats["pins"]["count"],
                    "newest_mtime": stats["pins"]["newest_mtime"],
                },
            },
            "records": [_record_to_dict(r) for r in self._records],
        }

        from stagekit.utils.serialization import save_json
        save_json(catalog, self._catalog_path)


# --------------------------------------------------
# Helpers
# --------------------------------------------------


def _extract(r: RunRecord, paths: tuple[str, ...]) -> object:
    """Resolve a dotted attribute path on a RunRecord.

    Tries each path in order; returns the first non-None value.
    Example: ``_extract(r, ("pipeline.model_name",))``
    """
    for path in paths:
        val = r
        for part in path.split("."):
            if val is None:
                break
            val = getattr(val, part, None)
        if val is not None:
            return val
    return None


def _as_set(value) -> set:
    """Normalize a filter value to a set."""
    if isinstance(value, (str, int, float)):
        return {value}
    return set(value)


def _parse_datetime(value: str | datetime) -> datetime:
    """Parse a datetime string or pass through a datetime object."""
    if isinstance(value, datetime):
        return value
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d_%H-%M-%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {value!r}")


def _record_to_dict(r: RunRecord) -> dict:
    """Serialize a RunRecord to a JSON-safe dict via dataclasses.asdict."""
    d = asdict(r)
    # Convert Path → str for JSON
    d["run_dir"] = str(d["run_dir"])
    return d


def _record_from_dict(d: dict) -> RunRecord:
    """Deserialize a RunRecord from a JSON dict."""
    pipeline = None
    if d.get("pipeline"):
        p = d["pipeline"]
        p["completed_stages"] = tuple(p.get("completed_stages", ()))
        pipeline = PipelineFields(**p)

    return RunRecord(
        run_type=d["run_type"],
        run_dir=Path(d["run_dir"]),
        experiment_name=d.get("experiment_name", ""),
        status=d.get("status", "unknown"),
        timestamp=d.get("timestamp"),
        pipeline=pipeline,
        pinned=tuple(d.get("pinned", ())),
    )
