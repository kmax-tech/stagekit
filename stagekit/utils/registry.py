import importlib
from typing import Any


class LazyRegistry:
    """Generic registry with lazy module loading.

    Stores items by name and lazily imports their modules on first access
    using a ``module_map`` (name -> module path). The map given here holds
    only the framework's own entries; a topic adds its entries with
    ``add_modules`` when it is loaded (``stagekit/run/topics.py``).
    """

    def __init__(self, kind: str, module_map: dict[str, str]):
        self._kind = kind
        self._module_map = dict(module_map)
        self._registry: dict[str, Any] = {}
        # name -> topic that added its map entry (framework entries absent)
        self._topic_of: dict[str, str] = {}

    def register(self, name: str):
        """Decorator that registers an item under *name*."""
        def decorator(item):
            if name in self._registry:
                raise ValueError(f"Duplicate {self._kind} registration: '{name}'")
            self._registry[name] = item
            return item
        return decorator

    def add_modules(self, module_map: dict[str, str], *, topic: str) -> None:
        """Add a topic's map entries (name -> module that registers it).

        A name already registered without a map entry — the framework's
        own direct registrations, or a topic module imported before its
        topic was loaded — is not checked here: the registered item stays,
        and the topic's module is not imported for that name. The topic is
        not told; its entry is dead for that name.

        Raises:
            ValueError: If a name already has a map entry pointing to
                another module — from the framework or another topic. Two
                owners of one name would make the item depend on which
                module happens to be imported first.
        """
        taken = self.conflicts(module_map)
        if taken:
            owners = {name: self._topic_of.get(name, "the framework") for name in taken}
            raise ValueError(
                f"topic '{topic}' registers {self._kind} names that are already "
                f"mapped elsewhere: {owners}"
            )
        self._module_map.update(module_map)
        self._topic_of.update({name: topic for name in module_map})

    def conflicts(self, module_map: dict[str, str]) -> list[str]:
        """Names in *module_map* whose existing map entry names another module."""
        return sorted(
            name for name, module in module_map.items()
            if self._module_map.get(name, module) != module
        )

    def get(self, name: str) -> Any:
        """Retrieve a registered item, lazily importing its module if needed."""
        if name not in self._registry and name in self._module_map:
            importlib.import_module(self._module_map[name])
        try:
            return self._registry[name]
        except KeyError as e:
            topics = sorted(set(self._topic_of.values())) or "none"
            raise ValueError(
                f"Unknown {self._kind}: '{name}' (topics that added {self._kind} "
                f"names: {topics}). Available: {self.available()}"
            ) from e

    def available(self) -> list[str]:
        """All known names (registered + lazy-loadable)."""
        return sorted(set(self._registry.keys()) | set(self._module_map.keys()))
