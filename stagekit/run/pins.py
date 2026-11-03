"""Pins: which run feeds a downstream pipeline.

``published/pins.yaml`` (version-controlled) maps a stable COORDINATE —
the address a consumer YAML interpolates, e.g.
``counterfactual/hong/llama3.2/neutral/p1`` or
``baselines/hong/llama3.2/zs`` — to exactly one run under ``output/``:
its directory, ``run_id``, ``repro_hash`` and a SHA-256 per pinned
artifact file. Consumers write ``preload_artifacts`` values as
``pin:<coordinate>``; the executor resolves the pin and reads the
artifact from the run itself. Nothing is copied.

Pinning is an explicit act after a run (``python -m scripts.pin``), not
a side effect of every run: a re-run with another prompt must never
swap a downstream input silently. A missing pin is an error that lists
candidate runs — there is no "take the newest".

Roles (docs/published-pins-statt-kopien.md):

- ``output/`` is the run ledger; comparing runs goes through RunIndex.
- ``pins.yaml`` is the publication layer: the decision which run feeds
  which consumer, with the hashes as the backlink.
- Pinned runs are exempt from every cleanup of ``output/``
  (``RunRecord.pinned`` carries the coordinates).

Pin entry::

    counterfactual/hong/llama3.2/neutral/p1:
      run_dir: output/pipeline/<group>/<combo>/<run>
      run_id: 1a2b3c4d-…
      repro_hash: 05a3d6b5…
      artifacts:
        counterfactual_extract.nli_features:
          file: stage_outputs/counterfactual_extract/…__dot__nli_features.json
          sha256: …
      pinned_at: 2026-10-08T14:02:11
      pinned_by: max
      note: ""

A pin names only what the run wrote. Values a consumer computes from a
pinned artifact (the baseline feature view, likelihoods for runs that
predate the field) are that consumer's ``derived_artifacts``
(stagekit/stages/derived_artifacts.py), never part of the pin.
"""

from __future__ import annotations

import getpass
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

import stagekit.config.paths as paths
from stagekit.run.artifact_manager import ArtifactManager
from stagekit.utils.serialization import compute_file_hash, load_json, load_json_safe

logger = logging.getLogger(__name__)

PIN_PREFIX = "pin:"


class PinError(ValueError):
    """A pin cannot be built, read or honoured."""


class PinNotFoundError(PinError, LookupError):
    """No pin for the coordinate; the message lists candidate runs."""


# ----------------------------------------------------------------------
# What a run offers for pinning
# ----------------------------------------------------------------------

# Stage types whose `publishes` declaration ({field: filename}) marks
# artifacts that cross the run boundary. WHAT each stage publishes is
# declared on the stage class; this list only names the contributors.
PUBLISHER_STAGE_TYPES = [
    "counterfactual_generate",
    "counterfactual_extract",
    "counterfactual_evaluate",
    "bayes_generate",
    "tournament_scenario_sample",
    "tournament_pair_judge",
    "bt_score",
    "casa_intervene",
    "casa_score",
]

# Baseline pipelines: whichever of these stages ran, its predictions are
# pinned under ONE role key — consumers address a baseline by the method
# axis of the coordinate and must not need to know the stage name.
BASELINE_PREDICTION_STAGES = (
    "fallacies_binary",
    "explanation_combined_binary",
    "explanation_binary_classify",
    # stage NAME of the 2026-05 ex_combined baseline runs that the paper
    # pins; the stage type was renamed to explanation_combined_binary later
    "ex_combined_binary",
)
BASELINE_PREDICTIONS_KEY = "baseline.predictions"

# Filename of the role-keyed baseline artifact (`publish_filenames`).
_BASELINE_FILENAMES = {BASELINE_PREDICTIONS_KEY: "predictions.json"}


@dataclass(frozen=True)
class PinnedArtifact:
    """One artifact of a pinned run: file relative to run_dir + its hash."""
    file: str
    sha256: str

    def to_dict(self) -> dict:
        return {"file": self.file, "sha256": self.sha256}


@dataclass(frozen=True)
class Pin:
    coordinate: str
    run_dir: str                 # relative to the project root when inside it
    run_id: str
    repro_hash: str
    artifacts: dict[str, PinnedArtifact] = field(default_factory=dict)
    pinned_at: str = ""
    pinned_by: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        """YAML / provenance form (coordinate is the mapping key, not a field)."""
        return {
            "run_dir": self.run_dir,
            "run_id": self.run_id,
            "repro_hash": self.repro_hash,
            "artifacts": {k: a.to_dict() for k, a in self.artifacts.items()},
            "pinned_at": self.pinned_at,
            "pinned_by": self.pinned_by,
            "note": self.note,
        }

    def resolved_run_dir(self, project_root: Path | None = None) -> Path:
        return (project_root or paths.PROJECT_ROOT) / self.run_dir


def _publisher_fields() -> list[tuple[str, str, str]]:
    """(stage_type, artifact_key, filename) from the stages' `publishes`."""
    from stagekit.pipeline.registry import get_stage_fn

    entries = []
    for stage_type in PUBLISHER_STAGE_TYPES:
        publishes = getattr(get_stage_fn(stage_type), "publishes", {}) or {}
        for field_name, filename in publishes.items():
            entries.append((stage_type, f"{stage_type}.{field_name}", filename))
    return entries


def publish_filenames() -> dict[str, str]:
    """artifact_key → canonical filename of every pinnable artifact.

    The filename is the stage's `publishes` declaration (the role-keyed
    baseline predictions: a fixed name). Filenames must be unique: they
    were one flat directory per export and a future `pin --copy` would
    put them side by side again.

    Raises:
        ValueError: On a duplicate filename.
    """
    names = {key: fn for _, key, fn in _publisher_fields()}
    names.update(_BASELINE_FILENAMES)
    filenames = list(names.values())
    if len(set(filenames)) != len(filenames):
        dupes = sorted({fn for fn in filenames if filenames.count(fn) > 1})
        raise ValueError(f"publish filenames must be unique, duplicated: {dupes}")
    return names


def pinnable_artifacts(run_dir: Path) -> dict[str, PinnedArtifact]:
    """Artifacts of *run_dir* that may be pinned, with live hashes.

    Publisher stages are looked up under their stage TYPE as stage name
    (the convention every publishing pipeline follows). Baseline
    predictions appear under the role keys.

    Raises:
        PinError: If more than one baseline prediction stage produced
            output — the role key would be ambiguous.
    """
    am = ArtifactManager.for_run(run_dir)
    found: dict[str, PinnedArtifact] = {}

    for stage_type, key, _ in _publisher_fields():
        path = am.resolve_artifact_path(stage_type, key)
        if path is not None:
            found[key] = PinnedArtifact(
                file=str(path.relative_to(run_dir)), sha256=compute_file_hash(path),
            )

    baseline_paths = {
        stage: path
        for stage in BASELINE_PREDICTION_STAGES
        if (path := am.resolve_artifact_path(stage, f"{stage}.predictions")) is not None
    }
    if len(baseline_paths) > 1:
        raise PinError(
            f"{run_dir}: {sorted(baseline_paths)} all produced predictions — "
            f"cannot pin one as '{BASELINE_PREDICTIONS_KEY}'"
        )
    for path in baseline_paths.values():
        found[BASELINE_PREDICTIONS_KEY] = PinnedArtifact(
            file=str(path.relative_to(run_dir)), sha256=compute_file_hash(path),
        )
    return found


def suggest_coordinate(cfg, artifacts: dict[str, PinnedArtifact]) -> str | None:
    """Coordinate in today's address scheme for a run of *cfg*, or None.

    A suggestion for the human who pins — never applied automatically.
    Baselines: ``baselines/<data>/<model>/<table_method>``; everything
    else: ``counterfactual/<data>/<model>/<mode>/p<prompt_nbr>``.
    """
    if not artifacts:
        return None
    data_name = cfg.data.data_name
    model_name = cfg.model.model_name if cfg.model else "unknown"
    if BASELINE_PREDICTIONS_KEY in artifacts:
        method = next(
            (
                (stage.config_override or {}).get("table_method")
                for stage in cfg.pipeline.stages
                if (stage.config_override or {}).get("table_method")
            ),
            None,
        )
        return f"baselines/{data_name}/{model_name}/{method}" if method else None
    if not cfg.pipeline.mode:
        return None
    return f"counterfactual/{data_name}/{model_name}/{cfg.pipeline.mode}/p{cfg.meta.prompt_nbr}"


# ----------------------------------------------------------------------
# The pins file
# ----------------------------------------------------------------------

_REQUIRED_PIN_KEYS = ("run_dir", "run_id", "repro_hash", "artifacts")


def pins_file(pins_path: Path | None) -> Path:
    return pins_path or paths.PINS_FILE


def read_pins(pins_path: Path | None = None) -> dict[str, Pin]:
    """All pins, keyed by coordinate. A missing file is an empty store.

    Raises:
        PinError: If the file is not a mapping or an entry lacks a
            required key — a half-written pin must not pass as "no pin".
    """
    path = pins_file(pins_path)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise PinError(f"{path}: expected a mapping coordinate → pin, got {type(raw).__name__}")

    pins: dict[str, Pin] = {}
    for coordinate, entry in raw.items():
        if not isinstance(entry, dict):
            raise PinError(f"{path}: pin '{coordinate}' is not a mapping")
        missing = [k for k in _REQUIRED_PIN_KEYS if k not in entry]
        if missing:
            raise PinError(f"{path}: pin '{coordinate}' lacks {missing}")
        artifacts = {}
        for key, art in (entry["artifacts"] or {}).items():
            if not isinstance(art, dict) or "file" not in art or "sha256" not in art:
                raise PinError(
                    f"{path}: pin '{coordinate}', artifact '{key}' needs file + sha256"
                )
            if set(art) - {"file", "sha256"}:
                raise PinError(
                    f"{path}: pin '{coordinate}', artifact '{key}': unexpected keys "
                    f"{sorted(set(art) - {'file', 'sha256'})} — a pin names files, it does not compute"
                )
            artifacts[key] = PinnedArtifact(file=str(art["file"]), sha256=str(art["sha256"]))
        pins[str(coordinate)] = Pin(
            coordinate=str(coordinate),
            run_dir=str(entry["run_dir"]),
            run_id=str(entry["run_id"]),
            repro_hash=str(entry["repro_hash"]),
            artifacts=artifacts,
            pinned_at=str(entry.get("pinned_at", "")),
            pinned_by=str(entry.get("pinned_by", "")),
            note=str(entry.get("note", "") or ""),
        )
    return pins


def write_pins(pins: dict[str, Pin], pins_path: Path | None = None) -> Path:
    """Write the whole store, sorted by coordinate, atomically."""
    path = pins_file(pins_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {c: pins[c].to_dict() for c in sorted(pins)}
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=120)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False,
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)
    return path


# ----------------------------------------------------------------------
# Resolve and load
# ----------------------------------------------------------------------

def coordinate_from_spec(spec: str, *, artifact_key: str = "") -> str:
    """Strip the ``pin:`` prefix of a preload value; refuse anything else.

    Raises:
        PinError: If *spec* is not ``pin:<coordinate>`` — file paths are
            no longer a preload source (docs/published-pins-statt-kopien.md).
    """
    where = f" for '{artifact_key}'" if artifact_key else ""
    if not isinstance(spec, str) or not spec.startswith(PIN_PREFIX):
        raise PinError(
            f"preload_artifacts{where}: expected 'pin:<coordinate>', got {spec!r}. "
            f"Preloads are resolved through {paths.PINS_FILE.name}, not file paths."
        )
    coordinate = spec[len(PIN_PREFIX):].strip().strip("/")
    if not coordinate or ".." in coordinate.split("/"):
        raise PinError(f"preload_artifacts{where}: invalid coordinate in {spec!r}")
    return coordinate


def _candidate_runs(coordinate: str, limit: int = 10) -> list[str]:
    """Successful runs whose identity fields contain the coordinate's segments."""
    from stagekit.analysis.run_index import RunIndex

    segments = [
        s for s in coordinate.split("/")[1:]
        if not (s.startswith("p") and s[1:].isdigit())
    ]
    hits = []
    for r in RunIndex().scan().pipeline_runs():
        if r.status != "success" or r.pipeline is None:
            continue
        haystack = f"{r.pipeline.data_name}|{r.pipeline.model_name}|{r.pipeline.run_group}"
        if all(s in haystack for s in segments):
            hits.append(r)
    hits.sort(key=lambda r: r.timestamp or "", reverse=True)
    return [
        f"{r.run_dir}  ({r.experiment_name}, {r.timestamp or 'no timestamp'})"
        for r in hits[:limit]
    ]


def resolve(coordinate: str, *, pins_path: Path | None = None) -> Pin:
    """The pin for *coordinate*.

    Raises:
        PinNotFoundError: With candidate runs (by data/model/mode match)
            and the command to pin one. Never falls back to the newest.
    """
    pins = read_pins(pins_path)
    pin = pins.get(coordinate)
    if pin is not None:
        return pin
    try:
        candidates = _candidate_runs(coordinate)
    except Exception as exc:  # noqa: BLE001 — the candidate list is a courtesy
        candidates = [f"(could not list candidates: {exc})"]
    hint = "\n  ".join(candidates) if candidates else "(no successful run matches the coordinate)"
    raise PinNotFoundError(
        f"No pin for '{coordinate}' in {pins_file(pins_path)}.\n"
        f"Candidate runs:\n  {hint}\n"
        f"Pin one with: python -m scripts.pin <run_dir> {coordinate}"
    )


def load(pin: Pin, artifact_key: str, *, project_root: Path | None = None) -> Any:
    """Read one pinned artifact from its run, hash-verified.

    Raises:
        PinError: If the key is not pinned under this coordinate.
        FileNotFoundError: If the run or the file is gone — the pin is
            broken (`python -m scripts.pin --list` shows it red).
        ValueError: If the live hash differs from the pinned one.
    """
    from stagekit.stages.artifact_schemas import verify_recorded_hashes

    art = pin.artifacts.get(artifact_key)
    if art is None:
        raise PinError(
            f"pin '{pin.coordinate}' has no artifact '{artifact_key}' "
            f"(pinned: {sorted(pin.artifacts)})"
        )
    path = pin.resolved_run_dir(project_root) / art.file
    if not path.exists():
        raise FileNotFoundError(
            f"pin '{pin.coordinate}': {path} is missing — the pinned run was "
            f"moved or deleted. Re-pin or restore the run."
        )
    verify_recorded_hashes(
        {"artifact_hashes": {artifact_key: art.sha256}, "run_id": pin.run_id},
        {artifact_key: compute_file_hash(path)},
        context=f"Pinned artifact integrity check failed (pin '{pin.coordinate}')",
        remedy="Re-pin the run if the change is intended: python -m scripts.pin <run_dir> <coordinate>",
    )
    return load_json(path)


def check_preload(
    spec: str, artifact_key: str, *, pins_path: Path | None = None,
    project_root: Path | None = None,
) -> list[str]:
    """Config-time check of one preload entry; returns error strings.

    Pin present, run directory present, artifact pinned. Hashes are
    verified at load time (`load`).
    """
    try:
        coordinate = coordinate_from_spec(spec, artifact_key=artifact_key)
    except PinError as exc:
        return [str(exc)]
    pin = read_pins(pins_path).get(coordinate)
    if pin is None:
        return [
            f"pipeline.preload_artifacts['{artifact_key}']: no pin for "
            f"'{coordinate}' (pin a run: python -m scripts.pin <run_dir> {coordinate})"
        ]
    errors = []
    run_dir = pin.resolved_run_dir(project_root)
    if not run_dir.is_dir():
        errors.append(
            f"pipeline.preload_artifacts['{artifact_key}']: pin '{coordinate}' "
            f"points to a missing run: {run_dir}"
        )
    if artifact_key not in pin.artifacts:
        errors.append(
            f"pipeline.preload_artifacts['{artifact_key}']: pin '{coordinate}' "
            f"has no such artifact (pinned: {sorted(pin.artifacts)})"
        )
    return errors


# ----------------------------------------------------------------------
# Pinning
# ----------------------------------------------------------------------

def stored_repro_hash(run_dir: Path) -> str:
    """The run's authoritative repro_hash from its config snapshot.

    The pin must carry the SAME hash the ledger recorded; recomputing it
    from a post-run cfg is a second derivation that can drift.

    Raises:
        PinError: If the snapshot or its repro_hash is missing.
    """
    snapshot = run_dir / "config_used.yaml"
    if not snapshot.exists():
        raise PinError(f"cannot pin: {snapshot} does not exist, so the run's repro_hash is unknown")
    data = yaml.safe_load(snapshot.read_text(encoding="utf-8")) or {}
    repro_hash = data.get("repro_hash")
    if not repro_hash:
        raise PinError(f"cannot pin: {snapshot} carries no repro_hash")
    return str(repro_hash)


def build_pin(
    run_dir: Path,
    coordinate: str,
    *,
    note: str = "",
    project_root: Path | None = None,
    now: datetime | None = None,
    user: str | None = None,
) -> Pin:
    """A Pin for *run_dir* under *coordinate* — nothing written yet.

    Raises:
        PinError: If the run is not a successful pipeline run or offers
            no pinnable artifact.
    """
    root = project_root or paths.PROJECT_ROOT
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise PinError(f"cannot pin: {run_dir} is not a directory")
    status = (load_json_safe(run_dir / "pipeline_status.json") or {}).get("status")
    if status != "success":
        raise PinError(f"cannot pin: {run_dir} has status {status!r}, need 'success'")
    manifest = load_json_safe(run_dir / "run_manifest.json") or {}
    run_id = manifest.get("run_id")
    if not run_id:
        raise PinError(f"cannot pin: {run_dir}/run_manifest.json carries no run_id")

    artifacts = pinnable_artifacts(run_dir)
    if not artifacts:
        raise PinError(
            f"cannot pin: {run_dir} offers no pinnable artifact (publisher stages: "
            f"{PUBLISHER_STAGE_TYPES}; baseline stages: {list(BASELINE_PREDICTION_STAGES)})"
        )

    try:
        stored_dir = str(run_dir.relative_to(root.resolve()))
    except ValueError:
        stored_dir = str(run_dir)

    return Pin(
        coordinate=coordinate.strip().strip("/"),
        run_dir=stored_dir,
        run_id=str(run_id),
        repro_hash=stored_repro_hash(run_dir),
        artifacts=artifacts,
        pinned_at=(now or datetime.now()).replace(microsecond=0).isoformat(),
        pinned_by=user or getpass.getuser(),
        note=note,
    )


def pin(
    run_dir: Path,
    coordinate: str,
    *,
    note: str = "",
    pins_path: Path | None = None,
    project_root: Path | None = None,
) -> tuple[Pin, Pin | None]:
    """Pin *run_dir* under *coordinate*; returns (new pin, replaced pin or None).

    An existing entry is replaced — the previous one lives on in the
    pins file's git history.
    """
    new = build_pin(run_dir, coordinate, note=note, project_root=project_root)
    pins = read_pins(pins_path)
    old = pins.get(new.coordinate)
    pins[new.coordinate] = new
    write_pins(pins, pins_path)
    if old is not None:
        logger.info("Replaced pin '%s': %s -> %s", new.coordinate, old.run_dir, new.run_dir)
    return new, old


# ----------------------------------------------------------------------
# Status and ledger support
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class PinStatus:
    pin: Pin
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.problems


def list_pins(
    *, pins_path: Path | None = None, project_root: Path | None = None,
) -> list[PinStatus]:
    """Every pin with its live state: run present, files present, hashes equal."""
    statuses = []
    pins = read_pins(pins_path)
    for coordinate in sorted(pins):
        p = pins[coordinate]
        problems: list[str] = []
        run_dir = p.resolved_run_dir(project_root)
        if not run_dir.is_dir():
            problems.append(f"run missing: {run_dir}")
        else:
            for key, art in p.artifacts.items():
                path = run_dir / art.file
                if not path.exists():
                    problems.append(f"{key}: file missing ({art.file})")
                elif compute_file_hash(path) != art.sha256:
                    problems.append(f"{key}: hash changed")
        statuses.append(PinStatus(pin=p, problems=tuple(problems)))
    return statuses


def pinned_coordinates_by_run_dir(
    *, pins_path: Path | None = None, project_root: Path | None = None,
) -> dict[Path, tuple[str, ...]]:
    """Resolved run directory → coordinates pinned to it (for the RunIndex)."""
    by_dir: dict[Path, list[str]] = {}
    for coordinate, p in read_pins(pins_path).items():
        by_dir.setdefault(p.resolved_run_dir(project_root).resolve(), []).append(coordinate)
    return {d: tuple(sorted(c)) for d, c in by_dir.items()}
