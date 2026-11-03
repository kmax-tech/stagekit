"""Where the project lives: the directory a run reads its configs from and writes its output to.

The project root is the directory named by the environment variable
``STAGEKIT_PROJECT_ROOT``, else the nearest of the working directory and
its parents that holds ``config/config.yaml``. It is looked up on first use
of a name below (``PROJECT_ROOT``, ``CONFIG_DIR``,
``EXPERIMENTS_CONFIG_DIR``, ``PINS_FILE``), so stagekit imports outside a
project; using one of these names there raises ``ProjectRootError``.
"""

import functools
import os
from pathlib import Path

ENV_VAR = "STAGEKIT_PROJECT_ROOT"
MARKER = Path("config") / "config.yaml"


class ProjectRootError(RuntimeError):
    """No project root: the variable is unset and no config/config.yaml is in reach."""


@functools.cache
def find_project_root() -> Path:
    """The project root, by ``STAGEKIT_PROJECT_ROOT`` or by search from the working directory.

    Raises:
        ProjectRootError: If the variable names a directory without
            ``config/config.yaml``, or it is unset and neither the working
            directory nor a parent holds one.
    """
    named = os.environ.get(ENV_VAR)
    if named:
        root = Path(named).resolve()
        if not (root / MARKER).is_file():
            raise ProjectRootError(f"{ENV_VAR}={named} holds no {MARKER}")
        return root
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / MARKER).is_file():
            return candidate
    raise ProjectRootError(f"no {MARKER} in {here} or a parent, and {ENV_VAR} is not set")


#: name -> its path below the project root
_BELOW_ROOT = {
    "PROJECT_ROOT": Path("."),
    "CONFIG_DIR": Path("config"),
    "EXPERIMENTS_CONFIG_DIR": Path("config") / "experiments",
    # The publication layer: which run feeds which downstream pipeline
    # (stagekit/run/pins.py). Version-controlled in the project.
    "PINS_FILE": Path("published") / "pins.yaml",
}


def __getattr__(name: str) -> Path:
    # Only these names look the root up; anything else (Python asks a
    # module for __path__ and the like) is a plain AttributeError.
    if name not in _BELOW_ROOT:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    root = find_project_root()
    return root if name == "PROJECT_ROOT" else root / _BELOW_ROOT[name]
