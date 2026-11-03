"""Topics: the code an experiment adds to the framework, loaded by name.

The framework's registries know only the framework's own entries. A topic
is a module named in the config's ``topics:`` list; it carries
``REGISTRATIONS``, a mapping registry kind -> {name: module that
registers it}, and loading the topic adds those entries to the
registries::

    REGISTRATIONS = {
        "stage": {"fallacies_binary": "src.stages.llm.fallacies_binary"},
        "parser": {"binary_fallacy": "src.parsers.classification"},
    }

The modules themselves stay unimported until a name is first used, as
before. Loading is idempotent per process: a topic loaded twice is added
once.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence

from stagekit.utils.registry import LazyRegistry

#: registry kind (as a topic writes it: lower-case words separated by
#: spaces, the registry's own ``kind``) -> "module:attribute" of the
#: framework's registry. Every LazyRegistry of the framework is here
#: (tests/test_core_boundary.py).
REGISTRY_HOMES: dict[str, str] = {
    "stage": "stagekit.pipeline.registry:STAGE_REGISTRY",
    "prompt": "stagekit.prompts.registry:PROMPT_REGISTRY",
    "parser": "stagekit.parsers.registry:PARSER_REGISTRY",
    "data loader": "stagekit.data_loader.registry:DATA_LOADER_REGISTRY",
    "split strategy": "stagekit.splits.registry:SPLIT_STRATEGY_REGISTRY",
    "model": "stagekit.models.registry:MODEL_REGISTRY",
    "backend": "stagekit.models.backends.factory:BACKEND_REGISTRY",
    "derivation": "stagekit.stages.derived_artifacts:DERIVATION_REGISTRY",
    "artifact schema": "stagekit.stages.artifact_schemas:ARTIFACT_SCHEMA_REGISTRY",
    "pipeline check": "stagekit.config.validation:PIPELINE_CHECK_REGISTRY",
    "run-name label": "stagekit.run.combinator:RUN_LABEL_REGISTRY",
}

_loaded: list[str] = []


class TopicError(ValueError):
    """A topic in ``topics:`` that cannot be loaded; the message names it."""


def registry_of(kind: str) -> LazyRegistry:
    """The framework registry for *kind*.

    Raises:
        KeyError: If no registry has that kind.
    """
    module, attr = REGISTRY_HOMES[kind].split(":")
    return getattr(importlib.import_module(module), attr)


def load_topics(topics: Sequence[str]) -> None:
    """Import each topic and add its ``REGISTRATIONS`` to the registries.

    Raises:
        TopicError: If a topic cannot be imported, has no
            ``REGISTRATIONS``, registers under a kind no registry has, or
            maps a name that another topic or the framework maps elsewhere.
            The message names the topic.
    """
    for topic in topics:
        if topic in _loaded:
            continue
        try:
            module = importlib.import_module(topic)
        except ImportError as exc:
            raise TopicError(f"topic '{topic}' (in topics) cannot be imported: {exc}") from exc
        registrations = getattr(module, "REGISTRATIONS", None)
        if not isinstance(registrations, dict):
            raise TopicError(
                f"topic '{topic}' has no REGISTRATIONS mapping "
                f"(registry kind -> {{name: module}})"
            )
        unknown = sorted(set(registrations) - set(REGISTRY_HOMES))
        if unknown:
            raise TopicError(
                f"topic '{topic}' registers under {unknown}, which no registry knows "
                f"(registries: {sorted(REGISTRY_HOMES)})"
            )
        # All conflicts first, so a refused topic leaves no registry half-extended.
        for kind, entries in registrations.items():
            taken = registry_of(kind).conflicts(entries)
            if taken:
                raise TopicError(
                    f"topic '{topic}' registers {kind} names that are already "
                    f"mapped to other modules: {taken}"
                )
        for kind, entries in registrations.items():
            registry_of(kind).add_modules(entries, topic=topic)
        _loaded.append(topic)


def loaded_topics() -> tuple[str, ...]:
    """The topics loaded in this process, in load order."""
    return tuple(_loaded)
