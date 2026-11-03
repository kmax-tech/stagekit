"""Base class for canonical stages.

Subclasses implement ``transform()`` to convert raw input into a list of
``CanonicalRecord`` dicts.  The base class handles output writing, schema
validation, and ``StageResult`` construction.
"""

import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any

from stagekit.stages.base import PipelineStage, StageResult, StageContext
from stagekit.stages.canonical.schema import validate_canonical_batch
from stagekit.utils.serialization import save_jsonl

logger = logging.getLogger(__name__)


def load_epistemic_set(taxonomy_path: Path) -> set[str]:
    """Return lowercased fallacy names that belong to the epistemic dimension (H).

    The returned set contains all name variants present in the taxonomy,
    lowercased.  Callers should compare with ``name.lower() in epistemic_set``.
    """
    with open(taxonomy_path, encoding="utf-8") as f:
        tax = json.load(f)
    return {k.lower() for k, v in tax["fallacy_to_dimension"].items() if v == "H"}


def is_epistemic(fallacy_name: str, epistemic_set: set[str]) -> bool:
    """Check if a fallacy name matches the epistemic set.

    Tries exact lowercase match first, then underscore-to-space normalization
    to handle datasets that use snake_case names (e.g. touchefallacy).
    """
    name = fallacy_name.lower()
    if name in epistemic_set:
        return True
    return name.replace("_", " ") in epistemic_set


class CanonicalStageBase(PipelineStage):
    """Base for stages that produce canonical JSONL records.

    ``output_path`` is required here and inherited by every subclass —
    required_config unions over the MRO (``required_config_keys()``), so
    subclasses declare only their own additions.
    """

    required_config = ["output_path"]
    output_path_config_keys = ["output_path"]
    optional_config = ["canonical_version"]
    provides_artifacts = ["canonical_output"]

    def run(self, model: Any, data: Any, ctx: StageContext) -> StageResult:  # noqa: ARG002
        co = self.get_config(ctx.stage_cfg)
        output_path = Path(co["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records, metrics = self.transform(co, ctx)

        validate_canonical_batch(records)
        save_jsonl(records, output_path)

        metrics.setdefault("count", len(records))

        stage_name = ctx.stage_cfg.name if ctx.stage_cfg else "unknown"
        logger.info(
            "Canonical stage '%s': %d records -> %s",
            stage_name, len(records), output_path,
        )

        return StageResult(
            metrics=metrics,
            artifacts={f"{stage_name}.canonical_output": str(output_path)},
            data=records,
        )

    @abstractmethod
    def transform(
        self, config: dict, ctx: StageContext
    ) -> tuple[list[dict], dict]:
        """Convert raw input into canonical records.

        Args:
            config: Stage config_override dict (from YAML).
            ctx: Stage context (access to AppConfig, artifacts, stage_dir).

        Returns:
            (records, metrics) where records is a list of CanonicalRecord
            dicts and metrics is a dict of extra stage-specific metrics
            (``count`` is added automatically by the base class).
        """
        ...
