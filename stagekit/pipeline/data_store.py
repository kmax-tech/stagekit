from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PipelineDataStore:
    """
    Stores intermediate pipeline outputs in a structured way.

    Structure:
        stages: {
            "stage_name": {
                "field_name": [value0, value1, value2, ...]
            }
        }

    Constraints:
        - All stored fields must be index-aligned with the dataset.
        - Values must be JSON-serializable (or convertible to string fallback).
    """

    stages: Dict[str, Dict[str, List[Any]]] = field(default_factory=dict)
    _dirty: bool = field(default=False, repr=False)

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def _ensure_stage(self, stage_name: str):
        if stage_name not in self.stages:
            self.stages[stage_name] = {}

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def add_output(self, stage_name: str, field: str, values: List[Any], *, expected_len: int | None = None):
        self._ensure_stage(stage_name)
        if field in self.stages[stage_name]:
            raise ValueError(f"Pipeline output already exists: {stage_name}.{field}")
        if expected_len is not None and len(values) != expected_len:
            raise ValueError(
                f"Pipeline output '{stage_name}.{field}' has {len(values)} values "
                f"but dataset has {expected_len} records"
            )
        self.stages[stage_name][field] = values
        self._dirty = True

    def clear_stage(self, stage_name: str):
        """Remove all outputs for a stage so it can republish cleanly."""
        if stage_name in self.stages:
            self._dirty = True
        self.stages.pop(stage_name, None)

    def get(self, stage_name: str, field: str) -> Optional[List[Any]]:
        """Retrieve a stored field output."""
        return self.stages.get(stage_name, {}).get(field)

    def is_empty(self) -> bool:
        return all(not fields for fields in self.stages.values())

    # ------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary representation.

        Returns the internal mapping directly (not a copy): the only
        caller is ``save_pipeline_data``, which serializes it
        immediately. Mutating the result would corrupt the store.
        """
        return self.stages
