"""Fakes for driving ``LLMBaseModel``'s call path without a live backend.

Everything in ``stagekit/models/base.py`` is reachable only through a concrete
model that already has a backend attached, and building a real backend wants
a running server. So every test in this area injects the same three pieces:
a parser that can be scripted to fail, a backend that answers or raises on
command, and a model whose backend is replaced rather than built.

They live here so the retry-loop, call-path, batch and stats tests describe
one system instead of four slightly different ones.
"""

from contextlib import contextmanager
from threading import Lock
from typing import Any, Generator, Optional

import stagekit.models.base as base_module
from stagekit.config.classes import MetaConfig, ModelConfig
from stagekit.models.backends.backends_base import Answer
from stagekit.models.base import LLMBaseModel
from stagekit.models.llm_cache import LLMCache

#: Distinguishes "caller passed None on purpose" from "caller said nothing".
KEEP_DEFAULT: Any = object()

#: The slot coordinates every cache entry written by :class:`HarnessModel`
#: with the default configs lands under. Handy for ``get_entry`` lookups.
SLOT_DEFAULTS = dict(dataset_id="ds", parser_id="binary", prompt_version="v1")


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------
class FakeParser:
    """Echoes the raw answer as the parsed value, optionally failing first.

    Args:
        script: Per-call behaviours, consumed in order. ``"echo"`` parses,
            ``"none"`` returns no parse (a parse failure without an
            exception), an ``Exception`` instance is raised (a parse failure
            with an error string). Calls beyond the script echo.
        cache_id: Parser identity that enters the cache key.
    """

    cache_id = "binary"

    def __init__(
        self,
        script: Optional[list[Any]] = None,
        cache_id: Optional[str] = None,
    ) -> None:
        self.script = list(script or [])
        self.calls = 0
        self._lock = Lock()
        if cache_id is not None:
            self.cache_id = cache_id

    def prepare_fields(self, fields: dict) -> dict:
        """Pass the caller's fields through unchanged."""
        return fields

    def parse_with_logging(
        self,
        raw: str,
        fields: Optional[dict] = None,
        entry_meta: Optional[dict] = None,
        template_str: Optional[str] = None,
    ) -> Optional[dict]:
        """Return the parsed result per the script.

        Raises:
            Exception: Whatever the script holds for this call.
        """
        with self._lock:
            step = self.script[self.calls] if self.calls < len(self.script) else "echo"
            self.calls += 1
        if isinstance(step, Exception):
            raise step
        if step == "none":
            return None
        return {"text": raw}


# ------------------------------------------------------------------
# Backend
# ------------------------------------------------------------------
class FakeBackend:
    """Backend stand-in that answers or raises on command and counts calls.

    Args:
        answer: What ``generate`` returns once the script is exhausted.
        error: When set, every unscripted call raises it instead — the
            "server is down for the whole run" case.
        script: Per-call behaviours, consumed in order. An ``Exception``
            instance is raised, an ``Answer`` is returned.
        max_safe_workers: The concurrency cap the batch API must respect.
            ``None`` means "no cap declared", like a real backend that
            leaves the class default in place.
        name: Backend name; it enters cache keys and stats keys.
    """

    def __init__(
        self,
        answer: Optional[Answer] = None,
        error: Optional[Exception] = None,
        script: Optional[list[Any]] = None,
        max_safe_workers: Optional[int] = 1,
        name: str = "fake",
    ) -> None:
        self.answer = answer if answer is not None else Answer(raw="fine")
        self.error = error
        self.script = list(script or [])
        self.max_safe_workers = max_safe_workers
        self.name = name
        self.calls = 0
        self.prompts: list[str] = []
        self._lock = Lock()

    def generate(self, prompt: str) -> Answer:
        """Return the next scripted answer.

        Raises:
            Exception: Whatever the script holds for this call, or *error*
                once the script is exhausted.
        """
        with self._lock:
            step = self.script[self.calls] if self.calls < len(self.script) else None
            self.calls += 1
            self.prompts.append(prompt)
        if isinstance(step, Exception):
            raise step
        if step is not None:
            return step
        if self.error is not None:
            raise self.error
        return self.answer


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
@contextmanager
def _backend_never_built() -> Generator[None, None, None]:
    """Stub ``build_backend`` for the duration of one construction.

    ``LLMBaseModel.__init__`` builds a backend from the config, which for
    a server-backed backend means reaching for a URL that is not there.
    Tests inject their own backend afterwards, so the built one is dead
    weight either way.
    """
    original = base_module.build_backend
    base_module.build_backend = lambda cfg: FakeBackend()
    try:
        yield
    finally:
        base_module.build_backend = original


class HarnessModel(LLMBaseModel):
    """``LLMBaseModel`` with backend, parser, template and cache injected.

    Args:
        model_cfg: Model config; defaults to :func:`model_cfg`.
        meta_cfg: Meta config; defaults to :func:`meta_cfg`. Pass ``None``
            explicitly to exercise the "meta_cfg not initialized" guards.
        backend: Backend to use instead of the built one.
        parser: Parser for the stateful API; pass ``None`` to exercise the
            "parser not initialized" guards.
        template: Prompt template for the stateful API.
        cache_dir: When given, attaches a real ``LLMCache`` rooted there.
    """

    def __init__(
        self,
        model_cfg_: Optional[ModelConfig] = None,
        meta_cfg_: Any = KEEP_DEFAULT,
        *,
        backend: Optional[FakeBackend] = None,
        parser: Any = KEEP_DEFAULT,
        template: Any = None,
        cache_dir: Any = None,
    ) -> None:
        with _backend_never_built():
            super().__init__(
                model_cfg_ if model_cfg_ is not None else model_cfg(),
                None,
                meta_cfg() if meta_cfg_ is KEEP_DEFAULT else meta_cfg_,
            )
        self.backend = backend if backend is not None else FakeBackend()
        self.parser = FakeParser() if parser is KEEP_DEFAULT else parser
        self.template = template
        if cache_dir is not None:
            self.cache = LLMCache.get(cache_dir)


# ------------------------------------------------------------------
# Configs
# ------------------------------------------------------------------
def model_cfg(**overrides: Any) -> ModelConfig:
    """A minimal LLM ModelConfig, with *overrides* applied."""
    base = dict(
        model_type="llm",
        backend="claude_cli",
        model_name="m",
        temperature=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)


def meta_cfg(**overrides: Any) -> MetaConfig:
    """A MetaConfig with cache and diagnostics off, with *overrides* applied."""
    base = dict(enable_cache=False, store_parse_diagnostics=False)
    base.update(overrides)
    return MetaConfig(**base)
