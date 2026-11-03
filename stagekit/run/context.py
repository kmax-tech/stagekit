# stagekit/run/context.py

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    run_type: str        # "pipeline" | "train" | "eval" | "test"
    run_name: str
