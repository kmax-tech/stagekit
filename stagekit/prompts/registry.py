from pathlib import Path
from string import Template
from typing import List

from stagekit.prompts.base import PromptTemplate
from stagekit.utils.registry import LazyRegistry

#: The framework has no prompts of its own; a topic adds its entries
#: when loaded (stagekit/run/topics.py).
PROMPT_MODULE_MAP: dict[str, str] = {}

PROMPT_REGISTRY = LazyRegistry("prompt", PROMPT_MODULE_MAP)


def register_prompt(name: str):
    """Decorator that calls the factory, validates the result, and registers."""
    def decorator(factory):
        template = factory()
        if not isinstance(template, PromptTemplate):
            raise TypeError(
                f"Prompt '{name}' must return a PromptTemplate"
            )
        # Store the instance, not the factory
        PROMPT_REGISTRY.register(name)(template)
        return factory
    return decorator


def register_prompt_dir(directory: Path) -> list[str]:
    """Register every ``<name>.txt`` in *directory* as prompt ``<name>``.

    Prompts from outside the topic code — an experiment's own prompt
    directory. The required fields are the template's ``$placeholders``
    (``string.Template``, as ``PromptRenderer`` substitutes them).

    Returns:
        The registered names, sorted.

    Raises:
        FileNotFoundError: If *directory* does not exist.
        ValueError: If it holds no ``.txt`` file, a file of another kind
            (it would be skipped without a word), or a name the registry
            already knows. Nothing is registered then.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"prompt directory not found: {directory}")
    files = sorted(f for f in directory.iterdir() if not f.name.startswith("."))
    others = [f.name for f in files if f.suffix != ".txt" or not f.is_file()]
    if others:
        raise ValueError(f"prompt directory {directory} holds more than <name>.txt files: {others}")
    if not files:
        raise ValueError(f"prompt directory {directory} holds no <name>.txt file")
    taken = sorted(f.stem for f in files if f.stem in PROMPT_REGISTRY.available())
    if taken:
        raise ValueError(f"prompts in {directory} are already registered: {taken}")
    for f in files:
        text = f.read_text(encoding="utf-8")
        PROMPT_REGISTRY.register(f.stem)(PromptTemplate(
            name=f.stem,
            template_str=text,
            required_fields=sorted(set(Template(text).get_identifiers())),
        ))
    return [f.stem for f in files]


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------

def get_prompt(name: str) -> PromptTemplate:
    return PROMPT_REGISTRY.get(name)


def list_prompts() -> List[str]:
    return PROMPT_REGISTRY.available()
