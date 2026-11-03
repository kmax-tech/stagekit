import logging
import random
import re
import time
from abc import ABC
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, ClassVar, Dict, Optional

from tqdm import tqdm

from stagekit.config.classes import ModelConfig
from stagekit.models.backends.backends_base import BackendContractError
from stagekit.models.llm_cache import CacheCorruptionError, CacheMiss
from stagekit.models.backends.factory import (
    BACKEND_CONDITIONAL_FIELDS, BACKEND_REGISTRY, backend_consumes,
)
from stagekit.models.backends.factory import build_backend
from stagekit.parsers.base import BaseResponseParser
from stagekit.prompts.base import PromptTemplate, PromptRenderer
from stagekit.models.llm_cache import LLMCache
from stagekit.utils.parse_diagnostics import ParseDiagnostics

logger = logging.getLogger(__name__)

# Error types the retry loop must not retry: the condition is a property of
# the request or the server, so every attempt would fail the same way.
_FATAL_ERROR_TYPES = frozenset({"auth", "invalid_request", "backend_contract"})

# HTTP status codes in an error message, matched as whole tokens. A rate-limit
# body quotes token budgets ("Limit: 40000 / min"); read as substrings those
# digits become a 400, and a retryable error is abandoned as fatal.
_STATUS_CODE_RE = re.compile(r"\b\d{3}\b")

# Conditional fields that already have their own key in the cache payload.
# Everything else a backend consumes travels in `backend_params`.
_CACHE_KEY_LEGACY_FIELDS = frozenset({"temperature"})


# ============================================================
# Base model abstraction
# ============================================================
class BaseModel(ABC):

    model_type: ClassVar[Optional[str]] = None

    def __init__(self, model_cfg, llm_task_cfg=None, meta_cfg=None):
        self.model_cfg = model_cfg
        self.llm_task_cfg = llm_task_cfg
        self.meta_cfg = meta_cfg

    @classmethod
    def from_config(cls, *, model_cfg, llm_task_cfg=None, meta_cfg=None):
        return cls(model_cfg, llm_task_cfg, meta_cfg)

    # --------------------------------------------------
    # Optional persistence hooks
    # --------------------------------------------------
    def load(self, path):
        return None

    def save(self, path):
        return None


# ============================================================
# LLM run statistics
# ============================================================
@dataclass
class LLMRunStats:
    # Global counters
    total_calls: int = 0
    cache_hits: int = 0
    llm_calls: int = 0
    retries: int = 0
    parse_success: int = 0
    parse_failure: int = 0
    parse_diagnostics: Dict[str, list] = field(
        default_factory=lambda: defaultdict(list)
    )
    per_parser: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    errors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failed_items: Dict[str, list] = field(default_factory=lambda: defaultdict(list))
    retry_exhausted_items: Dict[str, list] = field(
        default_factory=lambda: defaultdict(list)
    )

    # internal lock (not part of dataclass state)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    # --------------------------------------------------
    # Public summary API
    # --------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_calls": self.total_calls,
                "cache_hits": self.cache_hits,
                "llm_calls": self.llm_calls,
                "retries": self.retries,
                "parse_success": self.parse_success,
                "parse_failure": self.parse_failure,
                "per_parser": {
                    parser: dict(stats) for parser, stats in self.per_parser.items()
                },
                "errors": dict(self.errors),
                "failed_items": dict(self.failed_items),
                "retry_exhausted_items": dict(self.retry_exhausted_items),
            }

    # -----------------------------
    # Recording helpers (THREAD-SAFE)
    # -----------------------------
    def record_cache_hit(self):
        with self._lock:
            self.total_calls += 1
            self.cache_hits += 1

    def record_llm_call(self):
        with self._lock:
            self.total_calls += 1
            self.llm_calls += 1

    def record_retry(self):
        with self._lock:
            self.retries += 1

    def record_parse_success(self, parser_id: str):
        with self._lock:
            self.parse_success += 1
            self.per_parser[parser_id]["success"] += 1

    def record_parse_failure(self, parser_id: str):
        with self._lock:
            self.parse_failure += 1
            self.per_parser[parser_id]["failure"] += 1

    def record_parse_diagnostic(self, parser_id: str, diag):
        with self._lock:
            self.parse_diagnostics[parser_id].append(diag)

    def record_error(self, error_type: str):
        with self._lock:
            self.errors[error_type] += 1

    def record_failed_item(self, backend: str, item_id: str):
        with self._lock:
            self.failed_items[backend].append(item_id)

    def record_retry_exhausted(self, backend: str, item_id: str):
        with self._lock:
            self.retry_exhausted_items[backend].append(item_id)

    def reset(self):
        with self._lock:
            self.total_calls = 0
            self.cache_hits = 0
            self.llm_calls = 0
            self.retries = 0
            self.parse_success = 0
            self.parse_failure = 0
            self.errors.clear()
            self.failed_items.clear()
            self.retry_exhausted_items.clear()
            self.per_parser.clear()
            self.parse_diagnostics.clear()


# ============================================================
# Structured return type for LLM calls
# ============================================================
@dataclass
class LLMCallResult:
    value: Optional[Any] = None
    # "cache" | "llm" | "replay" — the last: a miss in a replay for an item
    # the replayed run lists as failed; nothing was read, nothing called.
    source: str = "llm"
    success: bool = False
    attempts: int = 0
    error: Optional[str] = None
    error_type: Optional[str] = None
    raw_response: Optional[str] = None

    @staticmethod
    def summarize(results: list["LLMCallResult"]) -> Dict[str, Any]:
        """Aggregate stats from a batch of LLM call results."""
        total = len(results)
        successes = sum(1 for r in results if r.success)
        failures = total - successes
        cache_hits = sum(1 for r in results if r.source == "cache")

        error_counts: Dict[str, int] = defaultdict(int)
        for r in results:
            if not r.success and r.error_type:
                error_counts[r.error_type] += 1

        return {
            "total": total,
            "success": successes,
            "failed": failures,
            "cache_hits": cache_hits,
            "errors": dict(error_counts),
        }


# ============================================================
# Per-call context (avoids passing 10+ args to each helper)
# ============================================================
@dataclass
class _LLMCallContext:
    prompt: str
    parser_fields: Dict[str, Any]
    max_retries: int
    prompt_version: str
    item_id: Optional[str]
    template_str: Optional[str]
    run_metadata: Optional[Dict[str, Any]]
    parser_id: str
    backend_name: str
    requested_nbr: int
    cache_key: Optional[str] = None
    cache_dataset_id: Optional[str] = None
    parser: Optional[BaseResponseParser] = None  # Explicit parser override (stateless API)
    sample_slot: Optional[str] = None  # Per-task sampling slot (multi-draw at T>0)

    @property
    def cache_slot(self) -> str:
        """Cache slot for this call.

        Default is the run-global ``str(prompt_nbr)`` — unchanged, so
        existing cache entries stay resolvable. A task that requests
        per-call sampling gets ``"{prompt_nbr}:{sample_slot}"``: the
        prompt_nbr prefix keeps replication runs (prompt_nbrs axis)
        from replaying each other's draws, and the colon keeps sampled
        slots disjoint from every bare prompt_nbr slot.
        """
        if self.sample_slot is None:
            return str(self.requested_nbr)
        return f"{self.requested_nbr}:{self.sample_slot}"


@dataclass(slots=True)
class _AttemptResult:
    """Result of a single LLM attempt (used between _single_attempt and _execute_llm_call)."""
    parsed: Optional[Any] = None
    raw: str = ""
    error: Optional[str] = None
    error_type: Optional[str] = None
    source: str = "llm"  # "llm" or "cache" (when race lost)

    @property
    def success(self) -> bool:
        return self.parsed is not None


# ============================================================
# The backend's stand-in in a replay
# ============================================================
class _ReplayBackend:
    """Stands in for the backend when ``meta.cache_mode`` is ``replay``.

    Carries the backend's name, which keys the cache, and refuses every
    call. ``_execute_llm_call`` answers a replay before any attempt, so a
    call reaching this is a bug in that path, not a miss.
    """

    max_safe_workers = None

    def __init__(self, name: str) -> None:
        self.name = name

    def generate(self, prompt: str):
        raise CacheMiss(
            f"replay: backend {self.name!r} was asked for an answer — a "
            f"replay calls no backend"
        )


# ============================================================
# LLM base model
# ============================================================
#
# LLM API modes
#
# This model supports two calling styles:
#
# 1. Stateful convenience API
#    - Uses self.template / self.parser
#    - Intended for single-step stages configured via llm_task
#    - Entry points: run_llm_call(), run_llm_tasks()
#
# 2. Stateless explicit API
#    - Takes template / parser per call
#    - Intended for multi-step workflow stages that orchestrate
#      multiple prompts/parsers internally
#    - Entry points: _run_llm_call_with(), run_llm_tasks_with()
#
# Core execution must resolve one effective parser/template and then
# use that consistently for cache keys, parsing, diagnostics, and retries.
# Do not mix explicit parser/template inputs with self.parser/self.template
# within the same execution path.
#
class LLMBaseModel(BaseModel, ABC):

    model_cfg: ModelConfig
    parser: Optional[BaseResponseParser]
    template: Optional[PromptTemplate]
    cache: Optional[LLMCache]

    def __init__(self, model_cfg: ModelConfig, llm_task_cfg=None, meta_cfg=None):
        super().__init__(model_cfg, llm_task_cfg, meta_cfg)
        self.llm_stats = LLMRunStats()
        self.cache = None
        # A replay builds no backend: building one reaches for the server
        # (the ollama backend checks and pulls the model on creation).
        if meta_cfg is not None and meta_cfg.cache_mode == "replay":
            self.backend = _ReplayBackend(
                BACKEND_REGISTRY.get(self.model_cfg.backend).name,
            )
        else:
            self.backend = build_backend(self.model_cfg)

    # --------------------------------------------------
    # Core execution (single LLM call with cache + retries)
    # --------------------------------------------------
    def _execute_llm_call(
        self,
        *,
        prompt: str,
        parser_fields: Dict[str, Any],
        max_retries: int,
        prompt_version: str,
        dataset_id: Optional[str] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
        item_id: Optional[str] = None,
        template_str: Optional[str] = None,
        parser=None,
        sample_slot: Optional[str] = None,
    ):
        # Use explicitly provided parser, fall back to self.parser
        effective_parser = parser or self.parser
        if effective_parser is None:
            raise RuntimeError("Parser not initialized on model")
        if self.meta_cfg is None:
            raise RuntimeError("meta_cfg not initialized on model")

        parser_id = getattr(effective_parser, "cache_id", effective_parser.__class__.__name__)
        backend_name = self.backend.name

        # Build per-call context
        cache_key, cache_dataset_id = self._build_cache_key(
            backend_name, prompt, parser_id, prompt_version, dataset_id,
        )

        ctx = _LLMCallContext(
            prompt=prompt,
            parser_fields=parser_fields,
            max_retries=max_retries,
            prompt_version=prompt_version,
            item_id=item_id,
            template_str=template_str,
            run_metadata=run_metadata,
            parser_id=parser_id,
            backend_name=backend_name,
            requested_nbr=self.meta_cfg.prompt_nbr,
            cache_key=cache_key,
            cache_dataset_id=cache_dataset_id,
            parser=effective_parser,
            sample_slot=sample_slot,
        )

        replay = self.meta_cfg.cache_mode == "replay"
        if replay and self.cache is None:
            raise RuntimeError(
                "meta.cache_mode='replay' but the cache is off "
                "(meta.enable_cache) — a replay reads every answer from it"
            )

        # Check cache
        cached = self._check_cache(ctx)
        if cached is not None:
            return LLMCallResult(value=cached, source="cache", success=True, attempts=0)
        if replay:
            return self._replay_miss(ctx)

        # Retry loop
        last_raw = ""
        last_error = None
        last_error_type = None
        attempts_made = 0
        fatal = False
        for attempt in range(max_retries + 1):
            attempts_made = attempt + 1
            ar = self._single_attempt(ctx, attempt)
            last_raw = ar.raw or last_raw  # keep most recent non-empty raw
            # Message and type travel together: carrying them forward
            # separately reports one attempt's message under another
            # attempt's type.
            if ar.error is not None or ar.error_type is not None:
                last_error, last_error_type = ar.error, ar.error_type
            if ar.success:
                return LLMCallResult(
                    value=ar.parsed, source=ar.source, success=True,
                    attempts=attempt + 1, raw_response=last_raw,
                )
            # Don't retry fatal (non-transient) errors
            if ar.error_type in _FATAL_ERROR_TYPES:
                fatal = True
                break
            if attempt < max_retries:
                self.llm_stats.record_retry()
                # Parse failures (200 OK, wrong format) → retry immediately.
                # API/network errors → exponential backoff.
                if ar.error_type != "parse_failure":
                    delay = min(0.5 * (2 ** attempt), 30) + random.uniform(0, 0.5)
                    time.sleep(delay)

        # Out of attempts, or stopped early on a fatal error.
        self._record_final_failure(
            ctx, last_raw, attempts_made, retries_exhausted=not fatal,
        )
        return LLMCallResult(
            value=None, source="llm", success=False,
            attempts=attempts_made,
            error=last_error, error_type=last_error_type,
            raw_response=last_raw,
        )

    # --------------------------------------------------
    # Helpers for _execute_llm_call
    # --------------------------------------------------
    def _build_cache_key(
        self,
        backend_name: str,
        prompt: str,
        parser_id: str,
        prompt_version: str,
        dataset_id: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """Compute cache key and dataset_id. Returns (None, None) when cache is disabled."""
        if self.cache is None:
            return None, None

        if dataset_id is None:
            raise ValueError("dataset_id must be provided when cache is enabled")

        # temperature enters the key only when this backend consumes it —
        # the same backend_consumes() that decides run identity
        # (repro_hash._model_canon). Before this, a CLI backend keyed on
        # temperature it never sends: two runs with the SAME repro_hash
        # got disjoint cache keys and paid every call twice (audit B1).
        # Keys of consuming backends stay byte-identical; CLI keys re-key
        # once (their entries are few; legacy data is expendable).
        cache_key = self.cache.make_key(
            backend=backend_name,
            prompt=prompt,
            model=self.model_cfg.model_name,
            temperature=(
                self.model_cfg.temperature
                if backend_consumes(backend_name, "temperature")
                else None
            ),
            parser_id=parser_id,
            prompt_version=prompt_version,
            backend_params=self._backend_params(backend_name),
        )
        return cache_key, dataset_id

    def _backend_params(self, backend_name: str) -> Dict[str, Any]:
        """Output-affecting settings beyond those already in the key payload.

        Read from the same declaration as the repro hash
        (BACKEND_CAPABILITIES), so "different cache key" and "different run
        identity" cannot drift apart. Empty for every backend that consumes
        only the legacy fields — which keeps all existing keys resolvable.

        Args:
            backend_name: The backend this call runs on.

        Returns:
            Field name -> value, for the fields this backend consumes.
        """
        return {
            f: getattr(self.model_cfg, f)
            for f in sorted(BACKEND_CONDITIONAL_FIELDS - _CACHE_KEY_LEGACY_FIELDS)
            if backend_consumes(backend_name, f)
        }

    def _check_cache(self, ctx: _LLMCallContext) -> Optional[Any]:
        """Return the cached parsed value, or None.

        Deliberately asymmetric with a live call: a hit carries the parsed
        value only, not the raw answer and not the reasoning trace. Both
        live in the cache entry and are reachable via ``get_entry`` for
        auditing; fetching them here would put a second file read on the
        hot path for data no artifact consumes.
        """
        if self.cache is None or ctx.cache_key is None or ctx.cache_dataset_id is None:
            return None

        cached = self.cache.get_one(
            key=ctx.cache_key,
            slot=ctx.cache_slot,
            dataset_id=ctx.cache_dataset_id,
            model_id=self.model_cfg.model_name,
            parser_id=ctx.parser_id,
            prompt_version=ctx.prompt_version,
        )

        if cached is not None:
            self.llm_stats.record_cache_hit()
        return cached

    def _replay_miss(self, ctx: _LLMCallContext) -> LLMCallResult:
        """A miss in a replay: a listed item counts as failed, any other stops.

        Listed means in ``meta.replay_failed_items`` — items the replayed run
        itself gave up on, so it stored no answer for them. They are recorded
        as failed items, as that run recorded them; no attempt is made.

        Raises:
            CacheMiss: The item is not listed — naming item and key.
        """
        assert self.meta_cfg is not None and self.cache is not None  # _execute_llm_call
        item = None if ctx.item_id is None else str(ctx.item_id)
        if item is not None and item in self.meta_cfg.replay_failed_items:
            self.llm_stats.record_failed_item(ctx.backend_name, item)
            logger.warning(
                "REPLAY: no cache entry for item %s, listed as failed in the "
                "replayed run (meta.replay_failed_items) — counted as failed",
                item,
            )
            return LLMCallResult(
                value=None, source="replay", success=False, attempts=0,
                error="listed as failed in the replayed run "
                      "(meta.replay_failed_items)",
                error_type="replay_listed_failure",
            )
        raise CacheMiss(
            f"replay: no cache entry for item {item!r} — key {ctx.cache_key}, "
            f"slot {ctx.cache_slot}, dataset {ctx.cache_dataset_id}, parser "
            f"{ctx.parser_id}, prompt version {ctx.prompt_version}, backend "
            f"{ctx.backend_name}, in {self.cache.cache_dir}. A replay calls no "
            f"backend; list the item in meta.replay_failed_items only if the "
            f"replayed run itself failed it"
        )

    def _parse_and_diagnose(
        self, ctx: _LLMCallContext, attempt: int, raw: str,
        thinking: str = "",
    ) -> tuple[Optional[Any], Optional[str]]:
        """Parse raw LLM output and record diagnostics if enabled.

        Args:
            ctx: The call context (parser, ids, prompt version).
            attempt: Zero-based retry attempt, recorded in the diagnostic.
            raw: The model's answer. Only this is parsed.
            thinking: The reasoning trace, recorded but never parsed.

        Returns:
            (parsed_result, error_string).
        """
        parsed = None
        error = None

        assert ctx.parser is not None  # guaranteed by _execute_llm_call
        try:
            parsed = ctx.parser.parse_with_logging(
                raw.strip(),
                fields=ctx.parser_fields,
                entry_meta={
                    "backend": ctx.backend_name,
                    "item_id": ctx.item_id,
                    "dataset_id": ctx.cache_dataset_id,
                    "parser_id": ctx.parser_id,
                    "prompt_version": ctx.prompt_version,
                    "attempt": attempt,
                },
                template_str=ctx.template_str,
            )
        except Exception as e:
            error = str(e)

        assert self.meta_cfg is not None  # guaranteed by _execute_llm_call
        if self.meta_cfg.store_parse_diagnostics:
            self.llm_stats.record_parse_diagnostic(
                ctx.parser_id,
                ParseDiagnostics(
                    parser_id=ctx.parser_id,
                    backend=ctx.backend_name,
                    item_id=ctx.item_id,
                    attempt=attempt,
                    prompt_version=str(ctx.prompt_version),
                    raw_response=raw,
                    thinking=thinking,
                    parsed_output=parsed,
                    error=error,
                ),
            )

        return parsed, error

    def _write_cache(
        self, ctx: _LLMCallContext, parsed: Any, raw: str, thinking: str = "",
    ) -> Optional[Any]:
        """Write parsed result to cache with race-condition guard.

        Returns the existing cached value if another thread won the race,
        otherwise writes and returns None.

        Race safety: ``get_or_set()`` checks and writes under a single
        per-shard lock — no window for another thread to interleave.
        """
        if self.cache is None:
            raise RuntimeError("_write_cache called but self.cache is None")
        if ctx.cache_key is None:
            raise RuntimeError("_write_cache called but cache_key is None")
        if ctx.cache_dataset_id is None:
            raise RuntimeError("_write_cache called but cache_dataset_id is None")

        assert self.meta_cfg is not None  # guaranteed by _execute_llm_call
        result, was_cached = self.cache.get_or_set(
            key=ctx.cache_key,
            value=parsed,
            slot=ctx.cache_slot,
            prompt=ctx.prompt,
            dataset_id=ctx.cache_dataset_id,
            model_id=self.model_cfg.model_name,
            parser_id=ctx.parser_id,
            prompt_version=ctx.prompt_version,
            raw=raw if self.meta_cfg.save_raw else None,
            thinking=thinking if self.meta_cfg.save_raw else None,
            meta=ctx.run_metadata,
        )

        if was_cached:
            # Another thread wrote first — adopt its value for consistency
            self.llm_stats.record_cache_hit()
            return result

        return None

    @staticmethod
    def _classify_backend_error(exc: Exception) -> str:
        """Classify a backend exception into a retry/stats category.

        ``BackendContractError`` is matched by type first: it means the
        server is misconfigured, which no amount of retrying fixes, and it
        carries no marker the string matching below would recognise. The
        remaining categories are best-effort string matching, so different
        backends may format messages differently.

        Args:
            exc: The exception raised by the backend.

        Returns:
            An error type; those in ``_FATAL_ERROR_TYPES`` stop the retry
            loop, any other is retried.
        """
        if isinstance(exc, BackendContractError):
            return "backend_contract"

        msg = str(exc).lower()
        codes = set(_STATUS_CODE_RE.findall(msg))
        # Rate limits first: their bodies quote budgets ("Limit: 400 / min"),
        # and the phrase itself is unambiguous where a bare number is not.
        if "rate limit" in msg or "429" in codes:
            return "rate_limit"
        if codes & {"401", "403"} or "authentication" in msg:
            return "auth"
        if "400" in codes or ("invalid" in msg and "request" in msg):
            return "invalid_request"
        if "timeout" in msg:
            return "timeout"
        if "connection" in msg:
            return "connection"
        if "server error" in msg or "5xx" in msg or codes & {"500", "502", "503"}:
            return "server_error"
        return type(exc).__name__

    def _single_attempt(
        self, ctx: _LLMCallContext, attempt: int,
    ) -> _AttemptResult:
        """Execute one LLM call + parse + diagnostics + cache write."""
        try:
            self.llm_stats.record_llm_call()
            answer = self.backend.generate(ctx.prompt)
            raw = answer.raw or ""
            thinking = answer.thinking or ""

            parsed, error = self._parse_and_diagnose(
                ctx, attempt, raw, thinking,
            )

            if parsed is not None:
                self.llm_stats.record_parse_success(ctx.parser_id)
                if self.cache and ctx.cache_key:
                    existing = self._write_cache(ctx, parsed, raw, thinking)
                    if existing is not None:
                        return _AttemptResult(parsed=existing, raw=raw, source="cache")
                return _AttemptResult(parsed=parsed, raw=raw)

            self.llm_stats.record_parse_failure(ctx.parser_id)
            return _AttemptResult(raw=raw, error=error, error_type="parse_failure")

        except (CacheCorruptionError, CacheMiss):
            # Infrastructure failure, not an LLM failure: retrying cannot
            # help, and every item hitting the shard would fail the same
            # way. Propagate so the batch aborts fail-fast. CacheMiss: the
            # replay stand-in was called, which _execute_llm_call rules out.
            raise
        except Exception as e:
            error_type = self._classify_backend_error(e)
            self.llm_stats.record_error(f"{ctx.backend_name}:{error_type}")
            logger.error("%s error [%s]: %s", ctx.backend_name, error_type, e)
            return _AttemptResult(error=str(e), error_type=error_type)

    def _record_final_failure(
        self,
        ctx: _LLMCallContext,
        last_raw: str,
        attempts: int,
        *,
        retries_exhausted: bool,
    ) -> None:
        """Record stats and log when an item is given up on.

        Args:
            ctx: The call context of the failed item.
            last_raw: The most recent non-empty raw answer, for the log.
            attempts: How many attempts were actually made.
            retries_exhausted: True when the budget ran out, False when a
                fatal error stopped the loop early. Only the former is a
                retry statistic — filing a one-attempt fatal abort under
                ``retry_exhausted_items`` reads as a flaky endpoint when the
                cause was a misconfigured server or a rejected request.
        """
        if retries_exhausted:
            self.llm_stats.record_error(f"{ctx.backend_name}:retries_exhausted")

        if ctx.item_id is not None:
            if retries_exhausted:
                self.llm_stats.record_retry_exhausted(ctx.backend_name, str(ctx.item_id))
            self.llm_stats.record_failed_item(ctx.backend_name, str(ctx.item_id))

        logger.error(
            "FINAL FAILURE (backend=%s, item_id=%s, attempts=%d of %d allowed, "
            "reason=%s)\nRaw output:\n%s",
            ctx.backend_name, ctx.item_id, attempts, ctx.max_retries + 1,
            "retries exhausted" if retries_exhausted else "fatal error, not retried",
            last_raw,
        )

    # --------------------------------------------------
    # Public API: single task
    # --------------------------------------------------
    def run_llm_call(self, **fields):
        """Execute a single LLM call using self.parser / self.template."""
        if self.parser is None:
            raise RuntimeError("Parser not initialized on model")
        if self.template is None:
            raise RuntimeError("Prompt template not initialized on model")
        return self._run_llm_call_with(
            template=self.template, parser=self.parser, fields=fields,
        )

    def _run_llm_call_with(self, *, template, parser, fields: dict):
        """Execute a single LLM call with explicitly provided template/parser.

        This is the stateless core — no mutation of self.parser/self.template.

        Reserved (non-template) field keys: ``item_id``, ``dataset_id``,
        ``run_metadata``, and ``sample_slot`` — the latter requests an
        independent draw of the same prompt (multi-vote sampling at T>0)
        by moving the call to its own cache slot; without it, repeated
        identical prompts collapse onto one cached answer.
        """
        if self.meta_cfg is None:
            raise RuntimeError("meta_cfg not initialized on model")

        fields = parser.prepare_fields(fields)
        prompt = PromptRenderer.render(template, fields)

        return self._execute_llm_call(
            prompt=prompt,
            parser_fields=dict(fields),
            max_retries=self.meta_cfg.max_retries,
            prompt_version=template.version,
            item_id=fields.get("item_id"),
            dataset_id=fields.get("dataset_id"),
            run_metadata=fields.get("run_metadata"),
            template_str=template.template_str,
            parser=parser,
            sample_slot=fields.get("sample_slot"),
        )

    # --------------------------------------------------
    # Batch execution with threading
    # --------------------------------------------------
    def run_llm_tasks(self, *, tasks: list[dict], desc: str) -> list[LLMCallResult]:
        """Execute multiple LLM calls using self.parser / self.template."""
        if self.parser is None:
            raise RuntimeError("Parser not initialized on model")
        if self.template is None:
            raise RuntimeError("Prompt template not initialized on model")
        return self.run_llm_tasks_with(
            template=self.template, parser=self.parser, tasks=tasks, desc=desc,
        )

    def run_llm_tasks_with(
        self,
        *,
        template,
        parser,
        tasks: list[dict],
        desc: str,
    ) -> list[LLMCallResult]:
        """Execute multiple LLM calls with explicitly provided template/parser.

        Stateless batch API for workflow stages that manage their own
        prompts.  Does not read or mutate self.parser / self.template.
        """
        if self.meta_cfg is None:
            raise RuntimeError("meta_cfg not initialized on model")
        requested = self.meta_cfg.max_workers
        backend_limit = getattr(self.backend, "max_safe_workers", None)
        max_workers = (
            min(requested, backend_limit)
            if backend_limit is not None
            else requested
        )

        def _call(task):
            return self._run_llm_call_with(
                template=template, parser=parser, fields=task,
            )

        results: list[LLMCallResult] = [None] * len(tasks)  # type: ignore[list-item]
        pbar = tqdm(total=len(tasks), desc=desc)
        lock = Lock()

        try:
            if max_workers <= 1:
                for pos, task in enumerate(tasks):
                    results[pos] = _call(task)
                    pbar.update(1)
                return results

            logger.info("LLM execution: multithreaded (workers=%d)", max_workers)

            # Workers propagate exceptions intentionally (fail-fast).
            # A single unrecoverable failure aborts the entire batch
            # rather than silently producing partial results that
            # downstream stages would treat as complete.
            def _worker(pos, task):
                result = _call(task)
                with lock:
                    pbar.update(1)
                return pos, result

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_worker, pos, task)
                    for pos, task in enumerate(tasks)
                ]
                # No batch-wide timeout: every call is already bounded by
                # its backend's request timeout plus the retry budget, so
                # each worker terminates on its own. A wall-clock limit
                # over ALL futures would abort large batches whose total
                # runtime legitimately exceeds any per-call bound.
                for f in as_completed(futures):
                    pos, result = f.result()
                    results[pos] = result

            return results
        finally:
            pbar.close()
