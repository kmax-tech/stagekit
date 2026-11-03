"""Artifacts a consumer pipeline derives from a preloaded one.

Declared in the pipeline YAML, so the derivation is THIS run's work and
part of its provenance (``run_inputs.yaml``), not the pin's::

    derived_artifacts:
      derived.cf_likelihoods:
        from: counterfactual_evaluate.probabilities
        derive: cf_likelihoods_from_probabilities

The key follows the ``<stage>.<field>`` convention under the pseudo-stage
``derived``: the field selects the preload schema and the feature-name
prefix (``cf_likelihoods_…``, unchanged from the producer's artifact),
the pseudo-stage says who made it. One function, two call sites: the
executor materialises every declared key into ``ctx.artifacts`` right
after preloading (single-mode consumers), and ``load_mode_artifacts``
derives per pinned mode (multi-mode stacking reads the pins directly).

Why not in the pin: a pin names what a run wrote, hashed. A derivation
there would run with whatever code is current, unhashed, and be labelled
as the producing run's output (NOTES.md 2026-10-08). Why not a derive
stage: the stacked SVM reads three pinned runs, not stage artifacts, so
a stage could only serve the primary mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from stagekit.utils.registry import LazyRegistry

logger = logging.getLogger(__name__)

DERIVED_PREFIX = "derived."

#: derivation name (``derive:`` in the YAML) → module that registers it;
#: each derivation is fn(source artifact data) → derived data and lives
#: with the stage whose output it re-derives. The framework has none of its
#: own; a topic adds its entries when loaded (stagekit/run/topics.py).
DERIVATION_MODULE_MAP: dict[str, str] = {}

DERIVATION_REGISTRY = LazyRegistry("derivation", DERIVATION_MODULE_MAP)


@dataclass(frozen=True)
class DerivedArtifact:
    key: str      # derived.<field>
    source: str   # a preloaded artifact key
    derive: str   # name in DERIVATION_REGISTRY


def parse_derived_artifacts(pipeline) -> dict[str, DerivedArtifact]:
    """``pipeline.derived_artifacts`` as typed declarations.

    Raises:
        ValueError: On a malformed entry or an unknown derivation — the
            config-time check (`check_derived_artifacts`) reports the
            same message without raising.
    """
    raw = getattr(pipeline, "derived_artifacts", None) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"pipeline.derived_artifacts must be a mapping, got {type(raw).__name__}")
    decls: dict[str, DerivedArtifact] = {}
    for key, spec in raw.items():
        if not isinstance(spec, dict) or set(spec) != {"from", "derive"}:
            raise ValueError(
                f"pipeline.derived_artifacts['{key}'] needs exactly the keys "
                f"'from' and 'derive', got {sorted(spec) if isinstance(spec, dict) else type(spec).__name__}"
            )
        if not key.startswith(DERIVED_PREFIX) or key.count(".") != 1:
            raise ValueError(
                f"pipeline.derived_artifacts['{key}']: key must be '{DERIVED_PREFIX}<field>' "
                f"(the pseudo-stage says this run derived it)"
            )
        if spec["derive"] not in DERIVATION_REGISTRY.available():
            raise ValueError(
                f"pipeline.derived_artifacts['{key}']: unknown derive "
                f"'{spec['derive']}' (known: {DERIVATION_REGISTRY.available()})"
            )
        decls[key] = DerivedArtifact(key=key, source=str(spec["from"]), derive=str(spec["derive"]))
    return decls


def check_derived_artifacts(pipeline) -> list[str]:
    """Config-time check: well-formed, source preloaded, key not a preload."""
    try:
        decls = parse_derived_artifacts(pipeline)
    except ValueError as exc:
        return [str(exc)]
    preload = pipeline.preload_artifacts or {}
    errors = []
    for key, decl in decls.items():
        if decl.source not in preload:
            errors.append(
                f"pipeline.derived_artifacts['{key}']: source '{decl.source}' is not in "
                f"preload_artifacts (derivations work on preloaded artifacts)"
            )
        if key in preload:
            errors.append(f"pipeline.derived_artifacts['{key}']: key is also preloaded")
    return errors


def derive(decl: DerivedArtifact, source_data: Any, *, source_label: str) -> Any:
    """Apply the declared derivation and shape-check the result.

    Raises:
        ArtifactSchemaError: If the derived value violates the schema
            registered for its field (same gate as a preloaded artifact).
    """
    from stagekit.stages.artifact_schemas import validate_preloaded_artifact

    data = DERIVATION_REGISTRY.get(decl.derive)(source_data)
    validate_preloaded_artifact(decl.key, data, source=f"{decl.derive}({source_label})")
    logger.info("Derived '%s' = %s(%s)", decl.key, decl.derive, decl.source)
    return data
