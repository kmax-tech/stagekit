"""stagekit stands alone: no module imports a topic, the registries know the framework only.

The boundary fallacy-detection held as a test before the split
(tests/test_core_boundary.py there, 2026-11-03): a topic enters through
what it registers (stagekit/run/topics.py), never by an import from here.
Imports inside functions count.
"""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "stagekit"


def modules() -> dict[str, Path]:
    out = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        out[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path
    return out


def foreign_imports(path: Path) -> list[str]:
    """Imports of ``src`` (where topics live) anywhere in *path*."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names = [node.module]
        found += [f"{path.relative_to(ROOT)}:{node.lineno} {n}" for n in names
                  if n == "src" or n.startswith("src.")]
    return found


def test_no_module_imports_a_topic():
    found = [hit for path in modules().values() for hit in foreign_imports(path)]
    assert not found, found


def test_the_scan_sees_imports():
    """Probe: a known stagekit import is found by the same walk, so an empty result means something."""
    tree = ast.parse((PACKAGE / "pipeline" / "executor.py").read_text(encoding="utf-8"))
    assert any(isinstance(n, ast.ImportFrom) and (n.module or "").startswith("stagekit.")
               for n in ast.walk(tree))


def test_every_module_imports_alone():
    """In a fresh interpreter each module imports without a project or a topic."""
    names = sorted(modules())
    code = "import importlib, sys\nfor n in sys.argv[1:]:\n    importlib.import_module(n)\n"
    result = subprocess.run([sys.executable, "-c", code, *names], cwd="/", capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]


def test_framework_maps_name_framework_modules_only():
    from stagekit.run.topics import REGISTRY_HOMES, registry_of

    outside = {
        kind: {name: m for name, m in registry_of(kind)._module_map.items()
               if not m.startswith("stagekit.")}
        for kind in REGISTRY_HOMES
    }
    assert not any(outside.values()), outside
    assert importlib.import_module("stagekit.run.topics").loaded_topics() == ()
