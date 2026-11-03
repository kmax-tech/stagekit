"""Conformance checks: does an extension fit the framework?

Checks for these extension kinds: data loader, stage, prompt and parser
(alone and as an LLM task's pair), artifact schema. Not checked: the
kinds derivation, pipeline check and run-name label (a topic's
registrations there are held only by the map-completeness test in
tests/test_core_boundary.py). Each check imports the registered module
through its registry and reads the declarations; ``check_data_loader``
also loads a sample the caller provides. Each returns the problems it
found as messages (empty: conforms). A topic runs them over everything it
registers in its tests (here: ``tests/test_conformance.py``); a failure
there says which promise to the framework an extension breaks before a
run finds out.
"""

from __future__ import annotations

from string import Template
from typing import Any, Iterable

from stagekit.config.classes import DataConfig
from stagekit.data_loader.base import BaseDataLoader, Dataset
from stagekit.data_loader.registry import DATA_LOADER_REGISTRY
from stagekit.parsers.base import BaseResponseParser
from stagekit.parsers.registry import PARSER_REGISTRY
from stagekit.pipeline.registry import STAGE_REGISTRY
from stagekit.prompts.base import PromptTemplate
from stagekit.prompts.registry import PROMPT_REGISTRY
from stagekit.stages.base import LLMWorkflowStage, PipelineStage

#: what validate_artifact (stagekit/stages/artifact_schemas.py) reads, by type:
#: a key it does not read for that type would be a check that never runs
SCHEMA_KEYS_BY_TYPE = {
    "list": frozenset({"type", "item_schema", "allow_empty", "custom_validator"}),
    "dict": frozenset({"type", "required_keys", "key_types", "custom_validator"}),
    "str": frozenset({"type", "allow_empty", "custom_validator"}),
}
ITEM_SCHEMA_KEYS = frozenset({"required_keys", "key_types"})


def _lookup(registry, name: str) -> tuple[Any, list[str]]:
    try:
        return registry.get(name), []
    except ValueError as exc:
        return None, [str(exc)]


def _check_keys_and_types(spec: dict, where: str) -> list[str]:
    problems = []
    required = spec.get("required_keys", [])
    if not (isinstance(required, (list, tuple)) and all(isinstance(k, str) for k in required)):
        problems.append(f"{where}: required_keys must be a list of strings")
    key_types = spec.get("key_types", {})
    # plain types only: the validator names the expected type by __name__
    if not (isinstance(key_types, dict) and all(isinstance(v, type) for v in key_types.values())):
        problems.append(f"{where}: key_types must map keys to types")
    return problems


def check_artifact_schema(spec: Any, *, where: str) -> list[str]:
    """A schema spec as ``validate_artifact`` reads it; *where* labels the messages."""
    if not isinstance(spec, dict):
        return [f"{where}: schema must be a dict, got {type(spec).__name__}"]
    kind = spec.get("type")
    if kind not in SCHEMA_KEYS_BY_TYPE:
        # validate_artifact passes any value when type is missing
        return [f"{where}: type must be one of {sorted(SCHEMA_KEYS_BY_TYPE)}, got {kind!r}"]
    problems = []
    unread = sorted(set(spec) - SCHEMA_KEYS_BY_TYPE[kind])
    if unread:
        problems.append(f"{where}: keys not read for type '{kind}': {unread}")
    if kind == "dict":
        problems += _check_keys_and_types(spec, where)
    item = spec.get("item_schema")
    if kind == "list" and item is not None:
        if not isinstance(item, dict):
            problems.append(f"{where}.item_schema: must be a dict")
        else:
            unread = sorted(set(item) - ITEM_SCHEMA_KEYS)
            if unread:
                problems.append(f"{where}.item_schema: keys not read: {unread}")
            problems += _check_keys_and_types(item, f"{where}.item_schema")
    if "custom_validator" in spec and not callable(spec["custom_validator"]):
        problems.append(f"{where}: custom_validator must be callable")
    return problems


def check_stage(name: str) -> list[str]:
    """A registered stage: its class, the prompts and parsers it names, its schemas."""
    cls, problems = _lookup(STAGE_REGISTRY, name)
    if cls is None:
        return problems
    if not (isinstance(cls, type) and issubclass(cls, PipelineStage)):
        return [f"stage '{name}': {cls!r} is not a PipelineStage subclass"]
    for prompt in cls.uses_prompts:
        problems += [f"stage '{name}' uses_prompts: {p}" for p in _lookup(PROMPT_REGISTRY, prompt)[1]]
    for parser in cls.uses_parsers:
        problems += [f"stage '{name}' uses_parsers: {p}" for p in _lookup(PARSER_REGISTRY, parser)[1]]
    if issubclass(cls, LLMWorkflowStage) and not (cls.uses_prompts and cls.uses_parsers):
        # the repro hash learns a workflow stage's instrument only from these
        problems.append(f"stage '{name}': an LLMWorkflowStage declares uses_prompts and uses_parsers")
    both = sorted(set(cls.required_config) & set(cls.optional_config))
    if both:
        problems.append(f"stage '{name}': config keys both required and optional: {both}")
    for field, spec in cls.artifact_schemas.items():
        problems += check_artifact_schema(spec, where=f"stage '{name}' artifact_schemas['{field}']")
    # a stage computing its artifacts per config can only be checked at run time
    if cls.get_provides_artifacts is PipelineStage.get_provides_artifacts:
        unprovided = sorted(set(cls.artifact_schemas) - set(cls.provides_artifacts))
        if unprovided:
            problems.append(f"stage '{name}': schemas for artifacts it does not provide: {unprovided}")
    return problems


def check_prompt(name: str) -> list[str]:
    """A registered prompt: a template whose required fields are its placeholders."""
    template, problems = _lookup(PROMPT_REGISTRY, name)
    if template is None:
        return problems
    if not isinstance(template, PromptTemplate):
        return [f"prompt '{name}': {template!r} is not a PromptTemplate"]
    if template.name != name:
        problems.append(f"prompt '{name}': registered under another name than its own ({template.name!r})")
    parsed = Template(template.template_str)
    if not parsed.is_valid():
        problems.append(f"prompt '{name}': a '$' that is no placeholder (write '$$')")
    else:
        placeholders = set(parsed.get_identifiers())
        if placeholders != set(template.required_fields):
            problems.append(
                f"prompt '{name}': required_fields {sorted(template.required_fields)} "
                f"are not its placeholders {sorted(placeholders)}"
            )
    return problems


def check_parser(name: str) -> list[str]:
    """A registered parser: a parser class with its own cache id and parse()."""
    cls, problems = _lookup(PARSER_REGISTRY, name)
    if cls is None:
        return problems
    if not (isinstance(cls, type) and issubclass(cls, BaseResponseParser)):
        return [f"parser '{name}': {cls!r} is not a BaseResponseParser subclass"]
    if not isinstance(cls.cache_id, str) or cls.cache_id in ("", BaseResponseParser.cache_id):
        # the cache key holds it: without its own id two parsers share entries
        problems.append(f"parser '{name}': needs its own cache_id, has {cls.cache_id!r}")
    if cls.parse is BaseResponseParser.parse:
        problems.append(f"parser '{name}': does not implement parse()")
    return problems


def check_parser_cache_ids(names: Iterable[str]) -> list[str]:
    """No two parser classes among *names* share a cache id (one class may
    stand under several names)."""
    owners: dict[str, set[str]] = {}
    problems: list[str] = []
    for name in names:
        cls, unknown = _lookup(PARSER_REGISTRY, name)
        problems += unknown
        if isinstance(cls, type):
            owners.setdefault(str(cls.cache_id), set()).add(f"{cls.__module__}.{cls.__qualname__}")
    return problems + [
        f"parser classes share cache_id '{cache_id}': {sorted(classes)}"
        for cache_id, classes in sorted(owners.items()) if len(classes) > 1
    ]


def check_prompt_parser_pair(prompt: str, parser: str) -> list[str]:
    """An LLM task's pair (``template`` + ``parser``): both conform."""
    return check_prompt(prompt) + check_parser(parser)


def check_data_loader(name: str, data_cfg: DataConfig) -> list[str]:
    """A registered data loader, run on a sample: a Dataset with unique ids, a
    cache id and the loader's stratify_keys. Only what the sample exercises
    is checked (a sample without components skips the component check)."""
    cls, problems = _lookup(DATA_LOADER_REGISTRY, name)
    if cls is None:
        return problems
    if not (isinstance(cls, type) and issubclass(cls, BaseDataLoader)):
        return [f"data loader '{name}': {cls!r} is not a BaseDataLoader subclass"]
    dataset = cls().load(data_cfg)
    if not isinstance(dataset, Dataset):
        return [f"data loader '{name}': load() returned {type(dataset).__name__}, not a Dataset"]
    if not dataset.records:
        problems.append(f"data loader '{name}': no records from the sample")
    ids = dataset.ids
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        problems.append(f"data loader '{name}': duplicate ids {duplicates[:5]}")
    if not dataset.cache_id:
        # the LLM cache path holds it
        problems.append(f"data loader '{name}': no cache_id")
    if dataset.stratify_keys != cls.stratify_keys:
        # load() must hand them on; otherwise splits fall back to unstratified
        problems.append(
            f"data loader '{name}': its Dataset has stratify_keys {dataset.stratify_keys}, "
            f"the loader declares {cls.stratify_keys}"
        )
    return problems
