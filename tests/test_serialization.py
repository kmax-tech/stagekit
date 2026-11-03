"""Tests for shared serialization, including complex-type handling."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stagekit.utils.serialization import (
    _to_jsonable,
    compute_file_hash,
    load_json,
    load_jsonl,
    save_json,
    save_jsonl,
)


def test_json_roundtrip():
    data = {"key": "value", "nested": {"a": 1}}
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(data, f)
        path = Path(f.name)

    loaded = load_json(path)
    assert loaded == data
    path.unlink()


def test_jsonl_roundtrip():
    records = [{"id": "a", "val": 1}, {"id": "b", "val": 2}]
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        path = Path(f.name)

    loaded = load_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0]["id"] == "a"
    assert loaded[1]["val"] == 2
    path.unlink()


def test_save_json_creates_file():
    data = {"test": True}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.json"
        save_json(data, path)
        assert path.exists()
        loaded = load_json(path)
        assert loaded == data


def test_load_jsonl_empty_lines_skipped():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        f.write('{"id": "a"}\n\n{"id": "b"}\n')
        path = Path(f.name)

    loaded = load_jsonl(path)
    assert len(loaded) == 2
    path.unlink()


def test_save_jsonl_writes_sha256_sidecar(tmp_path):
    path = tmp_path / "dataset.jsonl"
    records = [{"id": "a", "val": 1}, {"id": "b", "val": 2}]

    written = save_jsonl(records, path)

    sidecar = Path(str(path) + ".sha256")
    assert written == 2
    assert sidecar.exists()
    assert sidecar.read_text().strip() == compute_file_hash(path)


# ------------------------------------------------------------------
# _to_jsonable: complex types
# ------------------------------------------------------------------


def test_to_jsonable_path():
    assert _to_jsonable(Path("/foo/bar")) == "/foo/bar"


def test_to_jsonable_numpy_int():
    assert _to_jsonable(np.int64(42)) == 42
    assert isinstance(_to_jsonable(np.int64(42)), int)


def test_to_jsonable_numpy_float():
    result = _to_jsonable(np.float32(3.14))
    assert isinstance(result, float)


def test_to_jsonable_numpy_array():
    arr = np.array([1, 2, 3])
    assert _to_jsonable(arr) == [1, 2, 3]


def test_to_jsonable_dataclass():
    @dataclass
    class Sample:
        name: str
        value: int

    result = _to_jsonable(Sample(name="test", value=7))
    assert result == {"name": "test", "value": 7}


def test_to_jsonable_nested():
    """Mixed nested structure with Path, numpy, and plain types."""
    data = {
        "path": Path("/data/out"),
        "score": np.float64(0.95),
        "counts": np.array([10, 20]),
        "nested": {"inner_path": Path("a/b")},
    }
    result = _to_jsonable(data)
    assert result["path"] == "/data/out"
    assert isinstance(result["score"], float)
    assert result["counts"] == [10, 20]
    assert result["nested"]["inner_path"] == "a/b"


def test_save_json_handles_complex_types(tmp_path):
    """save_json should handle Path, numpy, dataclass without raising."""
    data = {
        "path": Path("/tmp/out"),
        "np_int": np.int32(5),
        "np_arr": np.array([1.0, 2.0]),
    }
    path = tmp_path / "complex.json"
    save_json(data, path)
    loaded = load_json(path)
    assert loaded["path"] == "/tmp/out"
    assert loaded["np_int"] == 5
    assert loaded["np_arr"] == [1.0, 2.0]

