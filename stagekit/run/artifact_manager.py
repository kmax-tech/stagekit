# stagekit/run/artifact_manager.py
import json
from pathlib import Path
from typing import Any

from stagekit.utils.serialization import load_json, load_json_safe, save_json


STAGE_OUTPUTS_DIRNAME = "stage_outputs"


class ArtifactManager:
    """Manages per-stage artifact directories and completion markers.

    Owns the run's stage_outputs/ directory tree.  All stage persistence
    (artifacts, completion state) goes through this class.  Cross-run
    inputs are not copied anywhere: a pin (stagekit/run/pins.py) points at a
    run, and the artifact is read from that run's tree.
    """

    def __init__(self, stage_outputs_dir: Path):
        self.dir = stage_outputs_dir

    @classmethod
    def for_run(cls, run_dir: Path) -> "ArtifactManager":
        """Manager for a run directory's stage-output tree.

        The run's own manifest names the directory (``paths.stage_outputs_dir``;
        runs before the 2026-07 rename wrote ``paths.artifacts_dir`` =
        ``artifacts``). Without a manifest — a run still in progress —
        the current default applies.
        """
        manifest = load_json_safe(run_dir / "run_manifest.json") or {}
        manifest_paths = manifest.get("paths", {}) or {}
        dirname = (
            manifest_paths.get("stage_outputs_dir")
            or manifest_paths.get("artifacts_dir")
            or STAGE_OUTPUTS_DIRNAME
        )
        return cls(run_dir / dirname)

    # -------------------------------------------------
    # Stage directory
    # -------------------------------------------------

    def stage_dir(self, stage_name: str) -> Path:
        """Return (and create) the canonical directory for a stage."""
        d = self.dir / stage_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -------------------------------------------------
    # Completion lifecycle
    # -------------------------------------------------

    def is_stage_completed(self, stage_name: str) -> bool:
        return (self.dir / stage_name / "_completed").exists()

    def mark_stage_completed(self, stage_name: str) -> None:
        (self.stage_dir(stage_name) / "_completed").write_text("done")

    # -------------------------------------------------
    # Artifact persistence
    # -------------------------------------------------

    def save_stage_artifacts(self, stage_name: str, artifacts: dict):
        stage_dir = self.stage_dir(stage_name)

        key_map = {}  # filename_stem -> original_key

        for key, val in artifacts.items():
            if key == "pipeline_data":
                continue

            safe_key = key.replace(".", "__dot__")
            out_path = stage_dir / f"{safe_key}.json"
            key_map[safe_key] = key

            try:
                save_json(val, out_path)
            except TypeError:
                (stage_dir / f"{safe_key}.txt").write_text(str(val), encoding="utf-8")

        # Persist key mapping for round-trip fidelity
        if key_map:
            save_json(key_map, stage_dir / "_keys.json")

        return stage_dir

    def resolve_artifact_path(self, stage_name: str, artifact_key: str) -> Path | None:
        """Return the on-disk path for an artifact by its logical key, or None."""
        stage_dir = self.dir / stage_name
        keys_path = stage_dir / "_keys.json"
        if not keys_path.exists():
            return None
        key_map = load_json(keys_path)
        for safe_key, original_key in key_map.items():
            if original_key == artifact_key:
                json_path = stage_dir / f"{safe_key}.json"
                if json_path.exists():
                    return json_path
                txt_path = stage_dir / f"{safe_key}.txt"
                if txt_path.exists():
                    return txt_path
        return None

    def load_artifact(self, stage_name: str, artifact_key: str) -> Any:
        """Load a single artifact by its logical key, or None if missing."""
        path = self.resolve_artifact_path(stage_name, artifact_key)
        if path is None:
            return None
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return path.read_text(encoding="utf-8")

    def load_stage_artifacts(self, stage_name: str):
        stage_dir = self.dir / stage_name
        if not stage_dir.exists():
            return {}

        keys_path = stage_dir / "_keys.json"
        if not keys_path.exists():
            return {}
        key_map = load_json(keys_path)

        data = {}
        for safe_key, original_key in key_map.items():
            json_path = stage_dir / f"{safe_key}.json"
            txt_path = stage_dir / f"{safe_key}.txt"
            if json_path.exists():
                data[original_key] = json.loads(json_path.read_text(encoding="utf-8"))
            elif txt_path.exists():
                data[original_key] = txt_path.read_text(encoding="utf-8")
        return data
