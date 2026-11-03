# stagekit/models/backends/factory.py

from stagekit.config.classes import ModelConfig
from stagekit.utils.registry import LazyRegistry

BACKEND_MODULE_MAP = {
    "openai": "stagekit.models.backends.openai_backend",
    "ollama": "stagekit.models.backends.ollama_backend",
    "ollama_completion": "stagekit.models.backends.ollama_completion_backend",
    "vllm": "stagekit.models.backends.vllm_backend",
    "claude_cli": "stagekit.models.backends.cli_backend",
    "codex_cli": "stagekit.models.backends.cli_backend",
    "labkit": "stagekit.models.backends.labkit_backend",
}

# ModelConfig fields whose identity-relevance depends on the backend, with
# the default for an unknown/legacy backend name. A field a backend does not
# pass to the model cannot affect output and must not enter the repro hash
# (see stagekit/analysis/repro_hash.py) or the LLM cache key.
#
# The defaults differ on purpose, and the polarity is the whole point:
# ``temperature`` predates this table, so stored configs carry it and must
# keep hashing as they always did (True). ``max_tokens`` and ``reasoning``
# were born after it — no stored config has them, so an unknown backend
# claiming them would only invent identity out of nothing (False).
CONDITIONAL_FIELD_DEFAULTS = {
    "temperature": True,
    "max_tokens": False,
    "reasoning": False,
    # Raw-completion knobs (born with ollama_completion, 2026-09-20): no
    # stored config has them, same polarity as max_tokens.
    "stop": False,
    "num_ctx": False,
    # Guard metadata, never identity: its effect on output travels
    # entirely through `temperature` (see repro_hash._model_canon, which
    # strips temperature when a model declares False). No backend lists
    # it in identity_fields, so it can never enter a hash.
    "supports_temperature": False,
}

#: The field names above, for callers that need the whole set.
BACKEND_CONDITIONAL_FIELDS = frozenset(CONDITIONAL_FIELD_DEFAULTS)

# Which of those fields each backend actually passes to the model. Every
# backend in BACKEND_MODULE_MAP needs an entry — enforced below, so a new
# backend cannot silently get wrong run identity.
BACKEND_CAPABILITIES = {
    "openai": {"identity_fields": frozenset({"temperature"})},
    "ollama": {"identity_fields": frozenset({"temperature"})},
    # A second ollama backend rather than more fields on `ollama`: a field a
    # backend consumes keys the cache and the repro hash of ALL its runs,
    # so widening `ollama` would re-key every stored ollama entry.
    "ollama_completion": {
        "identity_fields": frozenset({"temperature", "max_tokens", "stop", "num_ctx"}),
    },
    "vllm": {"identity_fields": frozenset({"temperature", "max_tokens", "reasoning"})},
    # CLI backends have no temperature flag, and no token/reasoning knobs.
    "claude_cli": {"identity_fields": frozenset()},
    "codex_cli": {"identity_fields": frozenset()},
    # labkit sends the temperature it is given; the rest of the call (token
    # cap, num_ctx, endpoints) is the models.toml alias, keyed in labkit's
    # cache by llm.identity — for ollama with the weights' digest.
    "labkit": {"identity_fields": frozenset({"temperature"})},
}

_missing = set(BACKEND_MODULE_MAP) - set(BACKEND_CAPABILITIES)
if _missing:
    raise RuntimeError(
        f"BACKEND_CAPABILITIES missing entries for: {sorted(_missing)} — "
        "declare output-affecting capabilities when registering a backend"
    )

_unknown = {
    name: sorted(caps["identity_fields"] - BACKEND_CONDITIONAL_FIELDS)
    for name, caps in BACKEND_CAPABILITIES.items()
    if caps["identity_fields"] - BACKEND_CONDITIONAL_FIELDS
}
if _unknown:
    raise RuntimeError(
        f"BACKEND_CAPABILITIES declares unknown identity fields: {_unknown} — "
        f"add them to CONDITIONAL_FIELD_DEFAULTS first"
    )


def backend_consumes(backend: str | None, field: str) -> bool:
    """Whether *backend* passes *field* through to the model.

    Args:
        backend: Backend name from ModelConfig, or None.
        field: One of BACKEND_CONDITIONAL_FIELDS.

    Returns:
        True if the field can affect this backend's output.

    Raises:
        KeyError: If *field* is not a conditional field — that is a typo at
            the call site, not a missing capability.
    """
    default = CONDITIONAL_FIELD_DEFAULTS[field]
    caps = BACKEND_CAPABILITIES.get(backend or "")
    return field in caps["identity_fields"] if caps else default


def backend_has_temperature(backend: str | None) -> bool:
    """Whether *backend* passes temperature through to the model."""
    return backend_consumes(backend, "temperature")


BACKEND_REGISTRY = LazyRegistry("backend", BACKEND_MODULE_MAP)


def register_backend(name: str):
    """Decorator that registers a backend class."""
    return BACKEND_REGISTRY.register(name)


def build_backend(model_cfg: ModelConfig):
    """Create execution backend from an ModelConfig."""
    backend_type = model_cfg.backend

    if not backend_type:
        raise ValueError(
            "ModelConfig.backend must be set (e.g. 'openai', 'ollama')"
        )

    backend_cls = BACKEND_REGISTRY.get(backend_type)
    # Backends declare which further ModelConfig fields they consume, so
    # adding one does not touch the backends that ignore it.
    extra = {
        field: getattr(model_cfg, field)
        for field in backend_cls.config_fields
    }
    return backend_cls(
        model_name=model_cfg.model_name,
        temperature=model_cfg.temperature,
        **extra,
    )
