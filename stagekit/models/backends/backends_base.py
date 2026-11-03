# stagekit/models/backends/backends_base.py
import logging
import os
from abc import ABC, abstractmethod
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Answer(NamedTuple):
    """One backend response, with the reasoning trace kept separate.

    A reasoning model emits a thinking trace alongside its answer. Whether
    the two arrive separated is a property of the SERVER, not of the model:
    ollama returns the trace in its own ``thinking`` field, while an
    OpenAI-compatible server does so only when started with
    ``--reasoning-parser``. Keeping them apart here means a parser never
    sees the trace — a trace merged into the answer flips labels silently
    instead of failing loudly (see NOTES.md 2026-09-01).

    Attributes:
        raw: The model's answer, as handed to the parser.
        thinking: The reasoning trace, or "" for a model that does not think.
    """

    raw: str
    thinking: str = ""


class BackendContractError(RuntimeError):
    """The server violated the contract its backend was configured for.

    Distinct from a transient failure: retrying cannot fix it, because the
    server itself is misconfigured (e.g. a reasoning model served without
    ``--reasoning-parser``). ``LLMBaseModel._classify_backend_error`` maps
    this to a fatal error type so the retry loop stops at the first item
    instead of hammering a broken endpoint once per item.
    """


class LLMBackend(ABC):
    """
    Backend adapter interface.
    Backends are responsible ONLY for prompt -> text generation.
    """

    name: str
    max_safe_workers: int | None = None

    #: ModelConfig fields this backend consumes; ``build_backend`` passes
    #: them as keyword arguments. Declaring a field here is what makes it
    #: reach the backend — whether it also counts towards run identity is
    #: declared separately in BACKEND_CAPABILITIES.
    config_fields: tuple[str, ...] = ()

    def __init__(self, model_name: str, temperature: float = 1):
        """Store the fields every backend carries; warn on a dead knob.

        The "this backend ignores temperature" warning derives from the
        SAME declaration that governs run identity (BACKEND_CAPABILITIES
        via ``backend_consumes``) — each backend restating it in its own
        constructor was a third copy of that knowledge.
        """
        self.model_name = model_name
        self.temperature = temperature

        from stagekit.models.backends.factory import backend_consumes
        if temperature != 1 and not backend_consumes(self.name, "temperature"):
            logger.warning(
                "%s backend ignores temperature (not in its identity "
                "fields); requested %s.", self.name, temperature,
            )

    def chat_payload(self, prompt: str) -> dict:
        """Single-user-message chat body of the OpenAI-protocol backends."""
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }

    @staticmethod
    def timeout_from_env(var: str, default: int) -> int:
        """Resolve a per-request timeout from the environment.

        One mechanism for every backend (audit B7): timeouts are node
        properties like the server URLs — they decide whether a call
        fails, never what the model answers — so they are env knobs,
        not ModelConfig fields, and never enter run identity.

        Args:
            var: Environment variable name (e.g. ``OLLAMA_TIMEOUT_S``).
            default: Seconds to use when the variable is unset.

        Raises:
            ValueError: If the variable is set but not an integer.
        """
        raw = os.getenv(var)
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError as e:
            raise ValueError(
                f"{var} must be an integer number of seconds, got {raw!r}"
            ) from e

    @staticmethod
    def require_env(var: str, what: str, *, example: str = "") -> str:
        """Read a required environment variable or fail loud.

        Args:
            var: Environment variable name.
            what: Human name of the thing it configures, for the error.
            example: Optional example value appended to the message.
        """
        value = os.getenv(var)
        if not value:
            suffix = f", e.g. {example}" if example else ""
            raise ValueError(f"Missing {what} (env {var}){suffix}.")
        return value

    @abstractmethod
    def generate(self, prompt: str) -> Answer:
        pass
