"""Where the project lives: the directory a run reads its configs from and writes its output to.

The project root is the directory named by the environment variable
``STAGEKIT_PROJECT_ROOT``, else the one of the working directory and its
parents that holds ``config/config.yaml`` — exactly one: with two on the
way up (an experiment area with its own config directory needs a
``config.yaml``, stagekit/run/utils.py) the search does not guess, the
variable must name the root. It is looked up on first use
of a name below (``PROJECT_ROOT``, ``CONFIG_DIR``,
``EXPERIMENTS_CONFIG_DIR``, ``PINS_FILE``), so stagekit imports outside a
project; using one of these names there raises ``ProjectRootError``.
The project's ``.env`` is the one in that root (``load_project_env``).
"""

import functools
import os
from pathlib import Path

from dotenv import load_dotenv

ENV_VAR = "STAGEKIT_PROJECT_ROOT"
MARKER = Path("config") / "config.yaml"


class ProjectRootError(RuntimeError):
    """No project root, or more than one: the variable is unset and the search
    finds no config/config.yaml, or several."""


@functools.cache
def find_project_root() -> Path:
    """The project root, by ``STAGEKIT_PROJECT_ROOT`` or by search from the working directory.

    Raises:
        ProjectRootError: If the variable names a directory without
            ``config/config.yaml``, or it is unset and the working directory
            and its parents hold none, or more than one.
    """
    named = os.environ.get(ENV_VAR)
    if named:
        root = Path(named).resolve()
        if not (root / MARKER).is_file():
            raise ProjectRootError(f"{ENV_VAR}={named} holds no {MARKER}")
        return root
    here = Path.cwd().resolve()
    found = [c for c in (here, *here.parents) if (c / MARKER).is_file()]
    if not found:
        raise ProjectRootError(f"no {MARKER} in {here} or a parent, and {ENV_VAR} is not set")
    if len(found) > 1:
        raise ProjectRootError(
            f"ambiguous project root: {MARKER} in {[str(c) for c in found]} — "
            f"set {ENV_VAR} to the project's directory"
        )
    return found[0]


def load_project_env() -> Path | None:
    """Load ``<project root>/.env`` (backend addresses, keys) into the environment.

    Its values override variables already set. The root is the one
    ``find_project_root`` finds, so the variable and the search apply to
    the ``.env`` as well; a ``.env`` in a subdirectory of the project (an
    experiment area) is not read.

    Returns:
        The file loaded, or None when stagekit runs outside a project (no
        root — importing stagekit there is allowed) or the project has no
        ``.env``.
    """
    try:
        root = find_project_root()
    except ProjectRootError:
        return None
    env = root / ".env"
    if not env.is_file():
        return None
    load_dotenv(env, override=True)
    return env


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
