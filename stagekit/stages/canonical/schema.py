"""Schema contracts for pipeline records.

Three levels of record contract, each stricter than the last:

    CanonicalRecord
        Base shape for all pipeline records.  Required: id, text.

    ClassificationRecord  (validated by ``validate_classification_record``)
        Adds: non-empty text, binary label, components with ≥1 claim
        and ≥1 premise (each typed as ``ComponentRecord``).

    PredictionRecord  (validated by ``validate_prediction_record``)
        Output contract for eval stages.  Required: id, type (dict
        that passes ``get_binary_label``).
"""

from typing import TypedDict

# Functional syntax required because ``_meta`` starts with underscore.
CanonicalRecord = TypedDict(
    "CanonicalRecord",
    {
        "id": str,
        "text": str,
        "labels": dict,
        "context": dict,
        "_meta": dict,
        "components": list,
        "enrichments": dict,
    },
    total=False,
)
# Mark required keys — id and text must always be present.
CanonicalRecord.__required_keys__ = frozenset({"id", "text"})
CanonicalRecord.__optional_keys__ = frozenset(
    {"labels", "context", "_meta", "components", "enrichments"}
)

REQUIRED_FIELDS = {"id", "text"}
DICT_FIELDS = {"labels", "context", "_meta", "enrichments"}
OPTIONAL_FIELDS = DICT_FIELDS | {"components"}
ALL_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS


def validate_canonical_record(record: dict, *, strict: bool = False) -> list[str]:
    """Validate a single record against the canonical schema.

    Returns a list of error strings (empty means valid).
    If *strict*, unknown top-level keys are rejected.
    """
    errors = []

    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"missing required field '{field}'")

    for field in DICT_FIELDS:
        val = record.get(field)
        if val is not None and not isinstance(val, dict):
            errors.append(f"'{field}' must be a dict, got {type(val).__name__}")

    if "components" in record and not isinstance(record["components"], list):
        errors.append(
            f"'components' must be a list, got {type(record['components']).__name__}"
        )

    if strict:
        unknown = set(record.keys()) - ALL_FIELDS
        if unknown:
            errors.append(f"unknown fields: {unknown}")

    return errors


# ------------------------------------------------------------------
# Label policy
# ------------------------------------------------------------------
# This block is the single source of truth for which label keys are
# recognised, how they map to binary values, and which are eligible
# for stratified splitting.  Canonical builders, analysis scripts,
# and split strategies should all derive their behaviour from these
# definitions — never hard-code label key names elsewhere.
#
# Binary label keys (tried in priority order by get_binary_label):
#   fallacy_status  — str, {"fallacy", "non-fallacy"}
#   fallacy_exists  — int, {0, 1}
#
# Stratify-eligible keys: the data loader's stratify_keys
# (in fallacy-detection: src/data_loader/argument.py).
#
# Canonical builders must only emit values from the allowed sets above.
# Records with unknown or missing labels must be skipped, not written
# with placeholder values (they would fail get_binary_label validation).
# ------------------------------------------------------------------

BINARY_LABEL_KEYS = ("fallacy_status", "fallacy_exists")
"""Label keys that carry binary (fallacy / non-fallacy) semantics,
in resolution-priority order."""

_FALLACY_STATUS_ALLOWED = {"fallacy", "non-fallacy"}


def get_binary_label(labels: dict) -> int:
    """Extract a binary (0/1) label from a canonical labels dict.

    Tries ``BINARY_LABEL_KEYS`` in priority order:
      - fallacy_status: "fallacy" → 1, "non-fallacy" → 0
      - fallacy_exists: int passthrough (strictly 0 or 1)

    Raises KeyError if no known label key is found.
    Raises ValueError if a key has an unexpected value.
    """
    if "fallacy_status" in labels:
        val = labels["fallacy_status"]
        if val not in _FALLACY_STATUS_ALLOWED:
            raise ValueError(
                f"fallacy_status must be one of {_FALLACY_STATUS_ALLOWED}, got {val!r}"
            )
        return 1 if val == "fallacy" else 0
    if "fallacy_exists" in labels:
        val = int(labels["fallacy_exists"])
        if val not in (0, 1):
            raise ValueError(
                f"fallacy_exists must be 0 or 1, got {val!r}"
            )
        return val
    raise KeyError(
        f"No known binary label key in {set(labels)}; "
        f"expected one of {BINARY_LABEL_KEYS}"
    )


def validate_canonical_batch(records: list[dict], *, strict: bool = False) -> None:
    """Validate a list of records. Raises ValueError if any are invalid."""
    all_errors: list[str] = []
    for i, record in enumerate(records):
        rec_errors = validate_canonical_record(record, strict=strict)
        if rec_errors:
            rid = record.get("id", f"index={i}")
            all_errors.append(f"{rid}: {'; '.join(rec_errors)}")
    if all_errors:
        raise ValueError(
            f"Invalid canonical records ({len(all_errors)} error(s)):\n"
            + "\n".join(f"  - {e}" for e in all_errors[:10])
            + (f"\n  ... and {len(all_errors) - 10} more"
               if len(all_errors) > 10 else "")
        )


# ------------------------------------------------------------------
# Component contract
# ------------------------------------------------------------------


class ComponentRecord(TypedDict):
    """One argument component (claim or premise) inside a canonical record."""

    text: str
    type: str  # "claim" | "premise"


_COMPONENT_TYPES = {"claim", "premise"}


def validate_component(component: object, *, index: int | None = None) -> list[str]:
    """Validate a single component dict.

    Returns a list of error strings (empty means valid).
    """
    prefix = f"components[{index}]" if index is not None else "component"
    errors: list[str] = []

    if not isinstance(component, dict):
        errors.append(f"{prefix} must be a dict, got {type(component).__name__}")
        return errors

    ctype = component.get("type")
    if not ctype:
        errors.append(f"{prefix}: missing 'type'")
    elif ctype not in _COMPONENT_TYPES:
        errors.append(f"{prefix}: type '{ctype}' not in {_COMPONENT_TYPES}")

    if not component.get("text", "").strip():
        errors.append(f"{prefix} (type={ctype}): empty text")

    return errors


# ------------------------------------------------------------------
# Classification input contract
# ------------------------------------------------------------------
# Stricter than canonical: requires components with at least one
# claim and one premise, a binary label, and text.  This is what
# the counterfactual + classify pipeline expects as preprocessed
# input.
# ------------------------------------------------------------------


class ClassificationRecord(TypedDict):
    """Input record for the classification pipeline.

    Stricter than CanonicalRecord: requires non-empty text, a binary
    label, and components with at least one claim and one premise.
    """

    id: str
    text: str
    labels: dict                     # must pass get_binary_label()
    components: list[ComponentRecord]  # ≥1 claim + ≥1 premise


def validate_classification_record(record: dict) -> list[str]:
    """Validate a record for the classification pipeline.

    Checks everything ``validate_canonical_record`` checks, plus:
      - ``labels`` must be present with a valid binary label key
      - ``components`` must contain ≥1 claim and ≥1 premise
      - ``text`` must be non-empty
    """
    errors = validate_canonical_record(record)
    if errors:
        return errors

    # text must be non-empty (BERT needs it)
    if not record.get("text", "").strip():
        errors.append("'text' is empty")

    # labels — must resolve via get_binary_label
    labels = record.get("labels")
    if not labels:
        errors.append("missing 'labels'")
    else:
        try:
            get_binary_label(labels)
        except (KeyError, ValueError) as e:
            errors.append(f"labels: {e}")

    # components — need claim + premise, each individually valid
    # (list type-check already done by validate_canonical_record)
    components = record.get("components")
    if not components:
        errors.append("missing 'components'")
    elif isinstance(components, list):
        types_found: set[str] = set()
        for i, c in enumerate(components):
            comp_errors = validate_component(c, index=i)
            errors.extend(comp_errors)
            if isinstance(c, dict) and c.get("type"):
                types_found.add(c["type"])
        for required in _COMPONENT_TYPES:
            if required not in types_found:
                errors.append(f"components missing type '{required}'")

    return errors


def check_classification_batch(records: list[dict]) -> list[str]:
    """Validate a list of records for the classification pipeline.

    Returns a list of error strings (one per invalid record, empty means
    all valid).  Callers decide whether to warn or abort.
    """
    all_errors: list[str] = []
    for i, record in enumerate(records):
        rec_errors = validate_classification_record(record)
        if rec_errors:
            rid = record.get("id", f"index={i}")
            all_errors.append(f"{rid}: {'; '.join(rec_errors)}")
    return all_errors


# ------------------------------------------------------------------
# Prediction output contract
# ------------------------------------------------------------------
# Every eval stage that produces a ``predictions`` artifact must
# emit records conforming to this shape.  The downstream consumer
# (BinaryLLMFeatureSource, analysis scripts) relies on ``type``
# being a dict that passes ``get_binary_label()``.
# ------------------------------------------------------------------


class PredictionRecord(TypedDict):
    """One prediction output from an eval stage."""

    id: str
    type: dict  # must pass get_binary_label()


def validate_prediction_record(record: object, *, index: int | None = None) -> list[str]:
    """Validate a single prediction dict.

    Returns a list of error strings (empty means valid).
    """
    prefix = f"predictions[{index}]" if index is not None else "prediction"
    errors: list[str] = []

    if not isinstance(record, dict):
        return [f"{prefix}: expected dict, got {type(record).__name__}"]

    if "id" not in record:
        errors.append(f"{prefix}: missing 'id'")

    type_info = record.get("type")
    if type_info is None:
        errors.append(f"{prefix}: missing 'type'")
    elif not isinstance(type_info, dict):
        errors.append(
            f"{prefix}: 'type' must be dict, got {type(type_info).__name__}"
        )
    else:
        try:
            get_binary_label(type_info)
        except (KeyError, ValueError) as e:
            errors.append(f"{prefix}: invalid type: {e}")

    return errors


def validate_prediction_batch(predictions: list, *, stage_name: str = "") -> None:
    """Validate a list of prediction dicts.

    Raises ``ValueError`` with all errors if any record is invalid.
    Called automatically by eval stages after building outputs.
    """
    all_errors: list[str] = []
    for i, pred in enumerate(predictions):
        all_errors.extend(validate_prediction_record(pred, index=i))

    if all_errors:
        ctx = f" (stage '{stage_name}')" if stage_name else ""
        sample = all_errors[:10]
        raise ValueError(
            f"Invalid predictions{ctx} — {len(all_errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in sample)
            + (f"\n  ... and {len(all_errors) - 10} more"
               if len(all_errors) > 10 else "")
        )
