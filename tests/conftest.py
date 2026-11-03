"""stagekit's tests run in a minimal project without a topic.

The project (a temporary directory with a ``config/config.yaml`` that names
no topics) is set through ``STAGEKIT_PROJECT_ROOT`` before any test looks
the root up, so the registries hold the framework alone.
"""

import os
import tempfile
from pathlib import Path

from stagekit.config.paths import ENV_VAR, MARKER

_project = Path(tempfile.mkdtemp(prefix="stagekit-test-project-"))
(_project / MARKER).parent.mkdir(parents=True)
(_project / MARKER).write_text("defaults:\n  - app_schema\n  - _self_\n\ntopics: []\n")
os.environ[ENV_VAR] = str(_project)
