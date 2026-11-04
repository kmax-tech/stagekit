"""Canonical reproducibility hash for pipeline configurations.

Computes a SHA-256 digest from the identity-determining fields of a
pipeline configuration, ignoring volatile runtime fields (run_id,
timestamps, worker counts, etc.).

Used at two points:
- Write time: ReproducibilityTracker stores the hash in config_used.yaml
- Read time: scanner reads it (or recomputes from the stored dict)

The hash carries the version of the formula that made it:
``v<N>-<sha256 hex>``. A hash without the prefix (64 hex digits) is
formula 1 — every hash stored up to 2026-11-03, made by a formula that
changed on purpose eight times since 2026-05-20 without saying so
(stagekit plan, step 0: 39 of 44 pins no longer recompute to their stored
hash). Two hashes of different versions are not compared: the runs may
be equivalent, the formula cannot tell. A stored hash is never rewritten
— it is the run's identity of its time. Add a new version (and keep the
old one computable) whenever a canon builder or what it reads changes.

Formulas that can still be computed:

- **2** (2026-11-03): the canon of 2026-09-24 — every schema field a
  config carries enters, except the listed volatile ones. A field that
  left the schema since a run was stored, or one added since (at its
  empty default), therefore changes the stored config's hash.
- **3** (2026-11-04): identity = only what can affect output (NOTES.md
  2026-07-04), applied to the schema as well: a key enters only if it is
  a field of today's schema and differs from that field's default. A
  field that left the schema and one added at its default no longer
  move the hash. Formula 3 reads today's defaults: changing the default
  of a field that enters the canon is a formula change (new version;
  ``tests/test_repro_hash_v3.py`` freezes the defaults and goes red).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import MISSING, asdict, fields
from typing import Any, Callable

#: Version of the formula new hashes are made with (see the module
#: docstring); 1 = unprefixed.
REPRO_HASH_VERSION = 3

_VERSIONED = re.compile(r"v(\d+)-[0-9a-f]{64}")
_UNVERSIONED = re.compile(r"[0-9a-f]{64}")

# Meta keys that do NOT affect pipeline output
_VOLATILE_META = frozenset({
    "run_id", "timestamp", "run_name", "run_group",
    "max_workers", "max_retries", "enable_cache", "cache_dir",
    "cache_mode", "replay_failed_items", "labkit_run_id",
    "save_raw", "store_parse_diagnostics", "store_only_failed_parses",
    "enable_pipeline_outputs", "max_parse_diagnostics_per_stage",
})

# Stage keys populated at runtime (already stripped from config_used.yaml)
_VOLATILE_STAGE = frozenset({"model_cfg", "llm_task_cfg"})

# Model keys that are cosmetic / path-only
_VOLATILE_MODEL = frozenset({"description"})


def _model_canon(model: dict) -> dict:
    """Identity-relevant subset of the model config.

    Only parameters that can change THIS run's output enter the hash —
    a field the backend ignores must not make two identical runs look
    different (backend-aware canon; see NOTES.md 2026-07-04). Which
    fields a backend ignores is declared in BACKEND_CAPABILITIES next to
    the backend registry.

    A denylist on purpose: a ModelConfig field added later lands in the
    hash unless someone declares otherwise (fail-safe). Only the fields in
    BACKEND_CONDITIONAL_FIELDS are decided per backend.

    Args:
        model: The model section of a stored or current config.

    Returns:
        The subset of *model* that can affect this run's output.
    """
    from stagekit.models.backends.factory import (
        BACKEND_CONDITIONAL_FIELDS, backend_consumes,
    )

    backend = model.get("backend")
    volatile = set(_VOLATILE_MODEL)
    volatile |= {
        f for f in BACKEND_CONDITIONAL_FIELDS if not backend_consumes(backend, f)
    }
    # supports_temperature (a conditional field no backend claims) stays
    # out of the canon; temperature itself stays IN it even for models
    # that reject overrides — validation forbids temperature != 1 for
    # them, so no false sensitivity can arise, and stored hashes (which
    # carry temperature: 1) keep matching.
    return {k: v for k, v in model.items() if k not in volatile}

# Pipeline-level keys that don't affect output.  "resume_from" no longer
# exists on PipelineConfig; it stays listed so config dicts stored by old
# runs hash identically to new ones.
_VOLATILE_PIPELINE = frozenset({"resume_from"})

# Pipeline keys added AFTER runs were stored, whose empty default means
# "not used": an empty value hashes as absent so every stored run keeps
# its identity; a non-empty value IS identity (it changes the features).
_EMPTY_IS_ABSENT_PIPELINE = frozenset({"derived_artifacts"})


def instrument_snapshot(cfg_dict: dict) -> dict:
    """Prompt versions and parser ids the pipeline's stages will run with.

    The config names stages, models and datasets, but the instrument a
    stage applies is its prompt TEXT and its parser LOGIC — and those
    live in the registries, not in the config. Without this snapshot two
    runs that differ only in a prompt's wording share one repro hash:
    the runner skipped such a run as "equivalent" and the export refused
    the second on the first one's pinned path (NOTES.md 2026-09-24).

    Resolution per stage: the stage class's ``uses_prompts`` /
    ``uses_parsers`` declarations, plus the template/parser of a stage's
    ``llm_task`` (loaded from config/llm_task/) and of the global
    ``llm_task`` fallback. Returns ``{"prompts": {name: version},
    "parsers": {name: cache_id}}``, or ``{}`` for a pipeline without any
    — so model-free pipelines keep the hash they always had.
    """
    from stagekit.config.classes import LLMConfig
    from stagekit.config.io import load_named_config
    from stagekit.parsers.registry import PARSER_REGISTRY
    from stagekit.pipeline.registry import STAGE_REGISTRY
    from stagekit.prompts.registry import PROMPT_REGISTRY
    from stagekit.run.topics import load_topics
    from stagekit.run.utils import config_topics

    # The names below resolve only once their topics are loaded; without
    # them every name would hash as "unregistered". A config that names no
    # topics — no key (stored before they existed, 2026-11-03) or an empty
    # list (built in code) — gets the root config's; an empty list does not
    # mean "framework only".
    load_topics(cfg_dict.get("topics") or config_topics())

    prompts: dict[str, str] = {}
    parsers: dict[str, str] = {}

    # A name the registries do not know still enters the hash (as
    # "unregistered"): the snapshot must never crash on a stored config
    # of an old run whose prompt or stage has since been renamed — the
    # preflight, not the hash, is where an unknown name is an error.
    def version_of(registry, name, attr) -> str:
        try:
            return str(getattr(registry.get(name), attr))
        except (ValueError, KeyError, AttributeError):
            return "unregistered"

    def add(prompt_names, parser_names) -> None:
        for name in prompt_names or ():
            prompts[name] = version_of(PROMPT_REGISTRY, name, "version")
        for name in parser_names or ():
            parsers[name] = version_of(PARSER_REGISTRY, name, "cache_id")

    def add_task(task) -> None:
        if not task:
            return
        template = task.get("template") if isinstance(task, dict) else task.template
        parser = task.get("parser") if isinstance(task, dict) else task.parser
        add([template] if template else [], [parser] if parser else [])

    for stage in cfg_dict.get("pipeline", {}).get("stages", []) or []:
        stage_type = stage.get("stage_type") or stage.get("name")
        try:
            cls = STAGE_REGISTRY.get(stage_type)
        except (ValueError, KeyError):
            cls = None
        add(getattr(cls, "uses_prompts", None), getattr(cls, "uses_parsers", None))
        if stage.get("llm_task"):
            try:
                add_task(load_named_config(stage["llm_task"], subdir="llm_task", cls=LLMConfig))
            except FileNotFoundError:
                prompts[f"llm_task:{stage['llm_task']}"] = "unregistered"
    add_task(cfg_dict.get("llm_task"))

    snap: dict = {}
    if prompts:
        snap["prompts"] = dict(sorted(prompts.items()))
    if parsers:
        snap["parsers"] = dict(sorted(parsers.items()))
    return snap


def _build_canonical_v2(cfg_dict: dict) -> dict:
    """Formula 2: identity-determining fields into a canonical structure.

    Frozen: stored v2 hashes recompute with it. Changes go into a new
    version.
    """
    canon = {}

    # Meta: only identity fields
    meta = cfg_dict.get("meta", {})
    canon["meta"] = {k: v for k, v in meta.items() if k not in _VOLATILE_META}

    # Data. component_label2id: a DataConfig field until 2026-11-03, never
    # set by a config; stored configs may carry it, it never was identity.
    data = cfg_dict.get("data", {})
    canon["data"] = {k: v for k, v in data.items() if k != "component_label2id"}

    # Global model fallback
    model = cfg_dict.get("model")
    if model:
        canon["model"] = _model_canon(model)

    # Global LLM task fallback
    llm_task = cfg_dict.get("llm_task")
    if llm_task:
        canon["llm_task"] = llm_task

    # Pipeline: name + full stage definitions, minus volatile keys
    pipeline = cfg_dict.get("pipeline", {})
    canon["pipeline"] = {
        k: v for k, v in pipeline.items()
        if k not in _VOLATILE_PIPELINE
        and not (k in _EMPTY_IS_ABSENT_PIPELINE and not v)
    }
    canon["pipeline"]["stages"] = [
        {k: v for k, v in s.items() if k not in _VOLATILE_STAGE}
        for s in pipeline.get("stages", [])
    ]

    # Seed
    canon["random_state"] = cfg_dict.get("random_state", 42)

    instrument = _instrument(cfg_dict)
    if instrument:
        canon["instrument"] = instrument

    return canon


def _instrument(cfg_dict: dict) -> dict:
    """The stored snapshot if the run wrote one (so a stored config hashes
    the same whatever the registries say today), otherwise resolved live
    from the registries (a config about to run)."""
    instrument = cfg_dict.get("instrument")
    if instrument is None:
        instrument = instrument_snapshot(cfg_dict)
    return instrument


_NO_DEFAULT = object()


def _defaults(cls) -> dict[str, Any]:
    """Each field of the dataclass *cls* with its default (``_NO_DEFAULT``
    for a field that has none, so it always differs)."""
    out: dict[str, Any] = {}
    for f in fields(cls):
        if f.default is not MISSING:
            out[f.name] = f.default
        elif f.default_factory is not MISSING:
            out[f.name] = f.default_factory()
        else:
            out[f.name] = _NO_DEFAULT
    return out


def _identity_subset(section: dict, cls, volatile: frozenset[str] = frozenset()) -> dict:
    """The keys of *section* that formula 3 counts as identity.

    A key counts if it is a field of today's *cls*, is not *volatile*, and
    differs from the field's default. A key that is no field (any more)
    and a field at its default drop out: a run stored before a field was
    added (it lacks the key) and one composed after (it carries the
    default) hash alike, as do a run stored with a field since removed and
    one composed without it.
    """
    defaults = _defaults(cls)
    return {
        k: v for k, v in section.items()
        if k in defaults and k not in volatile and v != defaults[k]
    }


def _build_canonical_v3(cfg_dict: dict) -> dict:
    """Formula 3: formula 2's sections, each reduced by ``_identity_subset``.

    Sections and volatile keys are formula 2's. ``resume_from``,
    ``component_label2id`` and the empty ``derived_artifacts`` need no
    rule of their own here: the first two are no fields any more, the
    third is its default.
    """
    from stagekit.config.classes import (
        AppConfig, DataConfig, LLMConfig, MetaConfig, ModelConfig,
        PipelineConfig, PipelineStageConfig,
    )

    canon: dict = {
        "meta": _identity_subset(cfg_dict.get("meta") or {}, MetaConfig, _VOLATILE_META),
        "data": _identity_subset(cfg_dict.get("data") or {}, DataConfig),
    }
    # model / llm_task: None means "no global fallback", which differs from
    # a section at its defaults — so a present section enters, even empty.
    if cfg_dict.get("model") is not None:
        canon["model"] = _identity_subset(_model_canon(cfg_dict["model"]), ModelConfig)
    if cfg_dict.get("llm_task") is not None:
        canon["llm_task"] = _identity_subset(cfg_dict["llm_task"], LLMConfig)

    pipeline = dict(cfg_dict.get("pipeline") or {})
    pipeline["stages"] = [
        _identity_subset(s, PipelineStageConfig, _VOLATILE_STAGE)
        for s in pipeline.get("stages") or []
    ]
    canon["pipeline"] = _identity_subset(pipeline, PipelineConfig)

    canon.update(_identity_subset(
        {k: cfg_dict[k] for k in ("random_state",) if k in cfg_dict}, AppConfig,
    ))

    instrument = _instrument(cfg_dict)
    if instrument:
        canon["instrument"] = instrument

    return canon


_CANON_BUILDERS: dict[int, Callable[[dict], dict]] = {
    2: _build_canonical_v2,
    3: _build_canonical_v3,
}


def _build_canonical(cfg_dict: dict) -> dict:
    """The canon today's formula (``REPRO_HASH_VERSION``) hashes."""
    return _CANON_BUILDERS[REPRO_HASH_VERSION](cfg_dict)


def repro_hash_from_dict(cfg_dict: dict, version: int = REPRO_HASH_VERSION) -> str:
    """Compute reproducibility hash from a config dict (e.g. loaded from YAML).

    Args:
        cfg_dict: A stored or current config dict.
        version: The formula to use; today's by default. An older one
            recomputes a stored hash of its time.

    Returns:
        ``v<version>-<sha256 hex>``.

    Raises:
        ValueError: If *version* names no formula that can be computed —
            formula 1 cannot: it was eight formulas, none recorded.
    """
    try:
        build = _CANON_BUILDERS[version]
    except KeyError:
        raise ValueError(
            f"repro hash formula {version} cannot be computed "
            f"(computable: {sorted(_CANON_BUILDERS)}; formula 1 was eight "
            f"unrecorded formulas)"
        ) from None
    raw = json.dumps(build(cfg_dict), sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"v{version}-{digest}"


def repro_hash_version(repro_hash: str) -> int:
    """The formula version a stored or computed hash was made with.

    Raises:
        ValueError: If *repro_hash* is neither ``v<N>-<64 hex>`` nor 64 hex
            digits (formula 1).
    """
    m = _VERSIONED.fullmatch(repro_hash)
    if m:
        return int(m.group(1))
    if _UNVERSIONED.fullmatch(repro_hash):
        return 1
    raise ValueError(f"not a repro hash: {repro_hash!r}")


def compute_repro_hash(cfg) -> str:
    """Compute reproducibility hash from an AppConfig dataclass instance."""
    cfg_dict = asdict(cfg)
    return repro_hash_from_dict(cfg_dict)
