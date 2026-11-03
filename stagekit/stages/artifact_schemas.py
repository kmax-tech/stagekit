"""Artifact schema enforcement for pipeline stages.

Each stage can declare ``artifact_schemas`` — a dict mapping artifact
field names to schema specs.  After ``run()`` returns, the executor
calls ``validate_stage_artifacts()`` to check:

1. Completeness: every ``provides_artifacts`` key is present in result.
2. No undeclared keys in result.artifacts (beyond the stage namespace).
3. Schema: each artifact value matches its declared schema.

Schema spec (simple dict, no jsonschema dependency)::

    {
        "type": "list",              # "list" | "dict" | "str"
        "item_schema": {             # for type="list" only
            "required_keys": [...],
            "key_types": {...},
        },
        "required_keys": [...],      # for type="dict" only
        "key_types": {...},          # for type="dict" only
        "allow_empty": True,         # empty list/str ok?
        "custom_validator": fn,      # (value, stage_name, artifact_key) -> list[str]
    }
"""

import logging
from typing import Any, Callable

from stagekit.utils.registry import LazyRegistry

logger = logging.getLogger(__name__)


# ── Exception ─────────────────────────────────────────────────────────

class ArtifactSchemaError(Exception):
    """Raised when a stage's output artifacts fail schema validation."""

    def __init__(self, stage_name: str, errors: list[str]):
        bullet_list = "\n".join(f"  - {e}" for e in errors)
        super().__init__(
            f"Stage '{stage_name}' artifact schema validation failed:\n{bullet_list}"
        )
        self.stage_name = stage_name
        self.errors = errors


# ── Schema constants ──────────────────────────────────────────────────

KFOLD_ASSIGNMENTS: dict = {
    "type": "dict",
    "required_keys": ["n_folds", "seed", "folds"],
    "key_types": {"n_folds": int, "seed": int, "folds": list},
}

def _classifier_results_validator(value, *, stage_name: str = "", artifact_key: str = "") -> list[str]:
    """Deep validation of a k-fold classifier results artifact.

    Two legitimate shapes:
      - insufficient-data form: exactly ``{"error": <str>}``
      - full form: model/folds/n_examples, per-metric {mean, std} stat
        dicts, and per_fold entries with predictions aligned to test_ids.
    """
    prefix = f"{stage_name}.{artifact_key}" if stage_name else artifact_key
    if not isinstance(value, dict):
        return [f"{prefix}: expected dict, got {type(value).__name__}"]

    # Insufficient-data form (see classify_base.InsufficientDataError)
    if set(value.keys()) == {"error"}:
        if not isinstance(value["error"], str):
            return [f"{prefix}['error']: expected str, got {type(value['error']).__name__}"]
        return []

    errors: list[str] = []
    for key in ("model", "folds", "n_examples", "accuracy", "f1", "auc", "per_fold"):
        if key not in value:
            errors.append(f"{prefix}: missing required key '{key}'")

    # Metric stats must be {mean, std} with numeric values
    for stat_key in ("accuracy", "f1", "auc", "precision", "recall"):
        stat = value.get(stat_key)
        if stat is None:
            continue
        if not isinstance(stat, dict) or not {"mean", "std"} <= set(stat):
            errors.append(f"{prefix}['{stat_key}']: expected {{mean, std}} dict")
            continue
        for part in ("mean", "std"):
            if not isinstance(stat[part], (int, float)):
                errors.append(
                    f"{prefix}['{stat_key}']['{part}']: expected number, "
                    f"got {type(stat[part]).__name__}"
                )

    per_fold = value.get("per_fold")
    if isinstance(per_fold, list):
        for i, fold in enumerate(per_fold):
            if not isinstance(fold, dict):
                errors.append(f"{prefix}.per_fold[{i}]: expected dict")
                continue
            for key in ("fold", "train_ids", "test_ids", "predictions",
                        "probabilities", "labels", "metrics"):
                if key not in fold:
                    errors.append(f"{prefix}.per_fold[{i}]: missing required key '{key}'")
            n_test = len(fold.get("test_ids", []))
            for aligned in ("predictions", "probabilities", "labels"):
                seq = fold.get(aligned)
                if isinstance(seq, list) and len(seq) != n_test:
                    errors.append(
                        f"{prefix}.per_fold[{i}]['{aligned}']: length {len(seq)} "
                        f"!= {n_test} test_ids"
                    )
    elif per_fold is not None:
        errors.append(f"{prefix}['per_fold']: expected list, got {type(per_fold).__name__}")

    return errors


CLASSIFIER_RESULTS: dict = {
    "type": "dict",
    "custom_validator": _classifier_results_validator,
}


def _prediction_batch_validator(value, *, stage_name: str = "", artifact_key: str = "") -> list[str]:
    """Record-level prediction contract — single authority.

    Delegates to ``canonical.schema.validate_prediction_record`` so the
    same rules apply wherever prediction batches are handed off.
    """
    from stagekit.stages.canonical.schema import validate_prediction_record

    errors: list[str] = []
    for i, pred in enumerate(value):
        errors.extend(validate_prediction_record(pred, index=i))
    prefix = f"{stage_name}.{artifact_key}" if stage_name else artifact_key
    return [f"{prefix}: {e}" for e in errors]


PREDICTION_BATCH: dict = {
    "type": "list",
    "item_schema": {
        "required_keys": ["id", "type"],
        "key_types": {"type": dict},
    },
    "custom_validator": _prediction_batch_validator,
}

EXPLANATIONS: dict = {
    "type": "list",
    "item_schema": {
        "required_keys": ["id", "explanation"],
        "key_types": {"explanation": str},
    },
}


def _judgement_validator(value, *, stage_name: str = "", artifact_key: str = "") -> list[str]:
    """Deep validation of a prompt-judge judgement artifact."""
    prefix = f"{stage_name}.{artifact_key}" if stage_name else artifact_key
    if not isinstance(value, dict):
        return [f"{prefix}: expected dict, got {type(value).__name__}"]
    errors: list[str] = []
    scores = value.get("scores")
    if not isinstance(scores, dict) or "overall" not in scores:
        errors.append(f"{prefix}['scores']: expected dict containing 'overall'")
    elif not all(isinstance(v, (int, float)) for v in scores.values()):
        errors.append(f"{prefix}['scores']: all values must be numeric")
    return errors


JUDGEMENT: dict = {
    "type": "dict",
    "required_keys": [
        "judged_prompt", "prompt_version", "judge_model",
        "scores", "weaknesses", "suggested_rules", "sample_ids",
    ],
    "key_types": {
        "judged_prompt": str, "prompt_version": str, "judge_model": str,
        "scores": dict, "weaknesses": list, "suggested_rules": list,
        "sample_ids": list,
    },
    "custom_validator": _judgement_validator,
}

BERT_CHECKPOINT: dict = {
    "type": "dict",
    "required_keys": [
        "checkpoint_path", "weights_sha256", "bert_model_name",
        "hidden_dim", "freeze_bert", "max_length", "text_source", "n_train",
    ],
    "key_types": {
        "checkpoint_path": str, "weights_sha256": str,
        "bert_model_name": str, "hidden_dim": int, "freeze_bert": bool,
        "max_length": int, "text_source": str, "n_train": int,
    },
}

CONSENSUS: dict = {
    "type": "dict",
    "required_keys": [
        "judges", "judged_prompt", "prompt_version", "scores",
        "shared_weaknesses", "shared_suggested_rules",
    ],
    "key_types": {
        "judges": list, "judged_prompt": str, "prompt_version": str,
        "scores": dict, "shared_weaknesses": list,
        "shared_suggested_rules": list,
    },
}

PREDICTION_SCORES: dict = {
    "type": "dict",
    "required_keys": ["n_examples", "accuracy", "precision", "recall", "f1", "f1_macro", "per_record"],
    "key_types": {"n_examples": int, "per_record": list},
}

PREDICTION_SCORES_CV: dict = {
    "type": "dict",
    "required_keys": ["n_examples", "n_folds", "accuracy", "precision", "recall", "f1",
                       "f1_macro", "accuracy_std", "precision_std", "recall_std", "f1_std",
                       "f1_macro_std", "per_fold"],
    "key_types": {"n_examples": int, "n_folds": int, "per_fold": list},
}


# ── Preload validation (cross-run inputs) ────────────────────────────
#
# Artifacts loaded from the artifacts/ publication store are integrity-
# checked by the executor (SHA-256 + provenance) — that proves the FILE
# is unchanged, not that its SHAPE matches what today's code expects.
# The registry assigns each preloadable FIELD name (the part after
# "stage." in the artifact key) its shape schema, so format drift between
# code versions fails loudly at the load boundary instead of deep inside
# feature code.

#: artifact field -> module that registers its schema. The framework
#: registers its own fields below; a topic adds its entries when loaded
#: (stagekit/run/topics.py).
ARTIFACT_SCHEMA_MODULE_MAP: dict[str, str] = {}

ARTIFACT_SCHEMA_REGISTRY = LazyRegistry("artifact schema", ARTIFACT_SCHEMA_MODULE_MAP)

for _field, _schema in {
    "fold_assignments": KFOLD_ASSIGNMENTS,
    "predictions": PREDICTION_BATCH,
    "explanations": EXPLANATIONS,
}.items():
    ARTIFACT_SCHEMA_REGISTRY.register(_field)(_schema)

def validate_preloaded_artifact(artifact_key: str, value: Any, *, source: str) -> None:
    """Shape-check a preloaded (cross-run) artifact against its schema.

    The field name (segment after the last ``.``) selects the schema
    from ``ARTIFACT_SCHEMA_REGISTRY``. Fields without a registered schema
    are accepted with an explicit log line naming the loaded topics — which
    fields have a schema depends on them (a run loads its config's topics
    before the executor preloads; code that skips that checks the
    framework's own fields only).

    Args:
        artifact_key: Namespaced key, e.g. ``counterfactual_extract.nli_features``.
        value: The deserialized artifact.
        source: File path for the error message.

    Raises:
        ArtifactSchemaError: If the value violates the registered schema.
    """
    field = artifact_key.rsplit(".", 1)[-1]
    if field not in ARTIFACT_SCHEMA_REGISTRY.available():
        from stagekit.run.topics import loaded_topics
        logger.info(
            "preload: no schema registered for field '%s' ('%s') — shape unchecked "
            "(topics loaded: %s)",
            field, artifact_key, list(loaded_topics()) or "none",
        )
        return

    errors = validate_artifact(
        value, ARTIFACT_SCHEMA_REGISTRY.get(field), stage_name="preload", artifact_key=artifact_key,
    )
    if errors:
        raise ArtifactSchemaError(
            "preload", [f"{e} (source: {source})" for e in errors],
        )


def verify_recorded_hashes(
    meta: dict,
    live_hashes: dict[str, str],
    *,
    context: str,
    remedy: str = "Re-pin the run if the change is intended.",
) -> None:
    """Compare live hashes against ``meta["artifact_hashes"]``.

    The one integrity routine for artifacts consumed across runs
    (pin loading, stagekit/run/pins.py): every loaded artifact must have a
    recorded hash, and it must match the live file.

    Args:
        meta: Recorded provenance — ``artifact_hashes`` (name → SHA-256)
            plus whatever identifies the source (``run_id``).
        live_hashes: name → SHA-256 of what was actually loaded.
        context: Error headline, e.g. ``"Pinned artifact integrity
            check failed (pin '<coordinate>')"``.
        remedy: Trailing hint on how to fix the mismatch.

    Raises:
        ValueError: Listing every unrecorded or mismatched artifact, with
            the recorded provenance appended.
    """
    recorded_hashes = meta.get("artifact_hashes", {})
    errors: list[str] = []

    for name, live_hash in live_hashes.items():
        if name not in recorded_hashes:
            errors.append(f"  {name}: loaded but no hash recorded")
        elif live_hash != recorded_hashes[name]:
            errors.append(
                f"  {name}: recorded {recorded_hashes[name][:16]}..., "
                f"live file is {live_hash[:16]}..."
            )

    if errors:
        raise ValueError(
            f"{context}:\n"
            + "\n".join(errors)
            + f"\n\nRecorded provenance: run_id={meta.get('run_id', '?')}\n"
            + remedy
        )


# ── table_summary contract ───────────────────────────────────────────
#
# pipeline_summary.json may contain a "table_summary" key: a list of
# normalized entries, one per scoring stage that declared table_method.
# The table builder reads only this block — never raw per-stage metrics.
#
# Required fields:
#   method    str   Fachliche Methode: "zs", "ex_combined",
#                   "ex_two_step", "svm", "bert", "bert_mlp"
#   stage     str   Stage name that produced the metrics
#   dataset   str   cfg.data.data_name (e.g. "hong", "big-bench")
#   model     str|null  cfg.model.model_name (null for model-free methods)
#   eval_mode str|null  "cv" or "single_run" — how metrics were computed
#   metrics   dict  Metric name → float.  Recognised keys:
#                     accuracy, precision, recall, f1, auc
#                   CV stages also provide {metric}_std (e.g. accuracy_std).
#
# Optional fields:
#   mode          str|null  Pipeline mode (e.g. "neutral", "skeptical")
#   variant       str|null  Variant (e.g. "true_to_false", "false_to_true")
#   source_model  str|null  Source model for counterfactual features
#   n_folds       int|null  Number of CV folds (only when eval_mode="cv")
#
# Stages opt in via `table_method` + `eval_mode`:
#   - Class attribute:   table_method = "svm", eval_mode = "cv"
#   - config_override:   table_method: zs (eval_predictions, eval_mode="single_run")
#
TABLE_SUMMARY_REQUIRED_FIELDS = {"method", "stage", "dataset", "model", "eval_mode", "metrics"}
TABLE_SUMMARY_OPTIONAL_FIELDS = {"mode", "variant", "source_model", "n_folds"}
TABLE_SUMMARY_METRIC_KEYS = {"accuracy", "precision", "recall", "f1", "auc"}


# ── Validation engine ─────────────────────────────────────────────────

def validate_artifact(
    value: Any,
    schema: dict,
    *,
    stage_name: str = "",
    artifact_key: str = "",
) -> list[str]:
    """Validate a single artifact value against its schema.

    Returns a list of error strings (empty means valid).
    """
    errors: list[str] = []
    prefix = f"{stage_name}.{artifact_key}" if stage_name else artifact_key

    expected_type = schema.get("type")
    if expected_type is None:
        return errors

    # Type check
    if expected_type == "list":
        if not isinstance(value, list):
            return [f"{prefix}: expected list, got {type(value).__name__}"]
        if not schema.get("allow_empty", False) and len(value) == 0:
            return [f"{prefix}: empty list not allowed"]
        # Item-level validation
        item_schema = schema.get("item_schema")
        if item_schema and value:
            errors.extend(_validate_list_items(value, item_schema, prefix))

    elif expected_type == "dict":
        if not isinstance(value, dict):
            return [f"{prefix}: expected dict, got {type(value).__name__}"]
        errors.extend(_validate_dict(value, schema, prefix))

    elif expected_type == "str":
        if not isinstance(value, str):
            return [f"{prefix}: expected str, got {type(value).__name__}"]
        if not schema.get("allow_empty", False) and not value:
            return [f"{prefix}: empty string not allowed"]

    # Custom validator
    custom: Callable | None = schema.get("custom_validator")
    if custom is not None:
        custom_errors = custom(value, stage_name=stage_name, artifact_key=artifact_key)
        if custom_errors:
            errors.extend(custom_errors)

    return errors


def _validate_dict(value: dict, schema: dict, prefix: str) -> list[str]:
    """Validate a dict artifact against required_keys and key_types."""
    errors: list[str] = []

    for key in schema.get("required_keys", []):
        if key not in value:
            errors.append(f"{prefix}: missing required key '{key}'")

    for key, expected in schema.get("key_types", {}).items():
        if key in value and not isinstance(value[key], expected):
            errors.append(
                f"{prefix}['{key}']: expected {expected.__name__}, "
                f"got {type(value[key]).__name__}"
            )

    return errors


def _validate_list_items(
    items: list,
    item_schema: dict,
    prefix: str,
) -> list[str]:
    """Validate every item in a list against item_schema."""
    errors: list[str] = []
    required_keys = item_schema.get("required_keys", [])
    key_types = item_schema.get("key_types", {})

    for i, item in enumerate(items):
        item_prefix = f"{prefix}[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{item_prefix}: expected dict, got {type(item).__name__}")
            continue

        for key in required_keys:
            if key not in item:
                errors.append(f"{item_prefix}: missing required key '{key}'")

        for key, expected in key_types.items():
            if key in item and not isinstance(item[key], expected):
                errors.append(
                    f"{item_prefix}['{key}']: expected {expected.__name__}, "
                    f"got {type(item[key]).__name__}"
                )

    return errors


# ── Stage-level completeness check ───────────────────────────────────

def validate_stage_artifacts(
    stage_name: str,
    result: Any,
    stage_instance: Any,
    stage_cfg: Any,
    cfg: Any,
) -> None:
    """Validate artifacts after stage.run() returns.

    Checks:
    1. Every provides_artifacts field is present in result.artifacts.
    2. No undeclared artifact keys.
    3. Each artifact with a declared schema passes validation.

    Raises ArtifactSchemaError on failure.
    """
    errors: list[str] = []

    # Declared artifact fields
    declared_fields = stage_instance.get_provides_artifacts(
        stage_cfg=stage_cfg, cfg=cfg,
    )
    expected_keys = {f"{stage_name}.{f}" for f in declared_fields}
    actual_keys = set(result.artifacts.keys())

    # 1. Completeness — every declared field must be present
    missing = expected_keys - actual_keys
    for key in sorted(missing):
        errors.append(f"declared artifact '{key}' not in result")

    # 2. No undeclared keys
    extra = actual_keys - expected_keys
    for key in sorted(extra):
        errors.append(f"undeclared artifact '{key}' in result")

    # 3. Schema validation
    schemas = getattr(stage_instance, "artifact_schemas", {})
    for field, schema in schemas.items():
        full_key = f"{stage_name}.{field}"
        if full_key not in result.artifacts:
            continue  # already flagged in step 1
        value = result.artifacts[full_key]
        errors.extend(validate_artifact(
            value, schema,
            stage_name=stage_name,
            artifact_key=field,
        ))

    if errors:
        raise ArtifactSchemaError(stage_name, errors)
