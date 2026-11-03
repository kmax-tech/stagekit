# stagekit/utils/serialization.py
import hashlib
import json
import os
import tempfile
import yaml
from pathlib import Path
from typing import Any, Iterable, Iterator
import numpy as np
from dataclasses import is_dataclass, asdict


def _to_jsonable(obj: Any):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)

    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))

    return obj


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _to_jsonable(obj)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as tmp:
        json.dump(clean, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_safe(path: Path) -> dict | None:
    """Load a JSON file as dict. Returns None if missing, malformed, or not a dict."""
    if not path.exists():
        return None
    try:
        data = load_json(path)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def save_yaml(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_yaml_safe(path: Path) -> dict:
    """Load a YAML file as dict. Returns {} if missing, malformed, or not a dict."""
    if not path.exists():
        return {}
    try:
        data = load_yaml(path)
        return data if isinstance(data, dict) else {}
    except (yaml.YAMLError, OSError):
        return {}


# ── file integrity ──────────────────────────────────────────────────────────

def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()



def verify_file_hash(path: Path) -> None:
    """Verify file against its .sha256 sidecar.

    Silently skips if no sidecar exists (file was created without hashing).
    Raises ValueError if the sidecar exists but the hash doesn't match.
    """
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.exists():
        return
    stored = sidecar.read_text().strip()
    current = compute_file_hash(path)
    if stored != current:
        raise ValueError(
            f"Integrity check failed for {path.name}: "
            f"expected {stored[:16]}..., got {current[:16]}... — "
            f"file may have been modified or corrupted"
        )


# ── JSONL ─────────────────────────────────────────────────────────────────

def iter_jsonl(path: Path) -> Iterator[dict]:
    """Stream records from a JSONL file.

    Verifies integrity if a .sha256 sidecar exists alongside the file.
    Raises ValueError on malformed lines (with line number).
    """
    path = Path(path)
    verify_file_hash(path)
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} in {path}"
                ) from e


def load_jsonl(path: Path, *, errors: str = "raise") -> list:
    """Load a JSONL file into a list of dicts.

    Verifies integrity if a .sha256 sidecar exists alongside the file.

    Args:
        errors: "raise" (default) raises on malformed lines,
                "skip" silently drops them.
    """
    if errors == "raise":
        return list(iter_jsonl(path))

    path = Path(path)
    verify_file_hash(path)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def save_jsonl(
    records: Iterable[dict], path: Path, *, sidecar_hash: bool = True,
) -> int:
    """Atomically write records to a JSONL file. Returns the record count.

    Atomic like ``save_json`` (temp file + os.replace) so an interrupt
    can never leave a half-written file. ``sidecar_hash=False`` skips
    the ``.sha256`` sidecar for internal working files (e.g. resume
    snapshots) that are not published artifacts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as tmp:
        for record in records:
            tmp.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")
            count += 1
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)
    if sidecar_hash:
        sidecar = Path(str(path) + ".sha256")
        sidecar.write_text(compute_file_hash(path), encoding="utf-8")
    return count


def normalize_config(cfg: Any) -> Any:
    """Normalize a config object (dataclass, dict, or None) for JSON serialization."""
    if cfg is None:
        return None

    if is_dataclass(cfg) and not isinstance(cfg, type):
        cfg = asdict(cfg)

    if isinstance(cfg, dict):
        normalized = {}
        for k, v in cfg.items():
            if isinstance(v, Path):
                normalized[k] = str(v)
            elif is_dataclass(v) and not isinstance(v, type):
                normalized[k] = asdict(v)
            else:
                normalized[k] = v
        return normalized

    return cfg
