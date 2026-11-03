import logging
import os
import random
import subprocess
import numpy as np
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None


def set_global_seed(seed: int) -> int:
    """Sets seed for Python, NumPy, and Torch (if installed)."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def collect_reproducibility_info(seed: int) -> Dict[str, Any]:
    """Return a dictionary with seed + environment reproducibility info."""

    py_state = random.getstate()
    np_state = np.random.get_state()

    # Extract state arrays - both return tuples where index 1 contains the state
    py_state_array = py_state[1]  # type: ignore[index]
    np_state_array = np_state[1]  # type: ignore[index]

    info: Dict[str, Any] = {
        "seed": seed,
        "python_random_state": list(py_state_array[:10]),
        "numpy_random_state": list(np_state_array[:10]),
    }

    if torch is not None:
        info["torch_manual_seed"] = torch.initial_seed()

    info.update(_collect_git_info())

    return info


def _collect_git_info() -> Dict[str, Any]:
    """Capture git commit SHA and dirty-worktree flag."""
    info: Dict[str, Any] = {}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if sha.returncode == 0:
            info["git_sha"] = sha.stdout.strip()

        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            capture_output=True, timeout=5,
        )
        info["git_dirty"] = dirty.returncode != 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("git not available — skipping git state capture")

    return info