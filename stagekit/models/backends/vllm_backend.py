"""vLLM backend: an OpenAI-compatible endpoint, with the reasoning trace kept apart.

Registered as its own backend rather than as ``openai`` with a base_url,
because the backend name is part of the LLM cache key: the same model served
by vLLM and by ollama is two instruments (different quantization, different
chat template), and their answers must not share cache entries.

The one thing this backend does that a plain HTTP client would not is refuse
a misconfigured server. Started without ``--reasoning-parser``, vLLM puts a
reasoning model's thinking trace inside ``content``, and nothing about the
response looks wrong.

What that costs is measured in tests/test_parser_trace_robustness.py rather
than assumed. Our parser scans bottom-up for the last JSON line, so a trace
followed by a complete answer still parses correctly — the danger is the
token budget. A deepseek-r1 trace runs ~1500 characters (~400 tokens); a
max_tokens sized for the answer alone is spent before the answer starts, and
what remains is a deliberation containing a draft verdict that the parser
reports as the answer. No error, no warning, wrong label. The same happens
when the model adds an afterthought after its answer.

Hence ``reasoning`` is declared per model and a violation raises instead of
returning.
"""

import logging
from typing import Any, Optional

import requests

from stagekit.models.backends.backends_base import (
    Answer, BackendContractError, LLMBackend,
)
from stagekit.models.backends.factory import register_backend

logger = logging.getLogger(__name__)

#: Seconds to wait for the metadata endpoints probed at construction time.
REACHABILITY_TIMEOUT_S = 15

#: Default seconds before a single generation request is abandoned.
#: Generous: a reasoning model can spend minutes on one answer. Override
#: per node via the VLLM_TIMEOUT_S env var (LLMBackend.timeout_from_env).
REQUEST_TIMEOUT_S = 1800

#: What a run records when the server reports no instrument digest. Explicit
#: rather than empty — the gap is a fact about the run, not a missing value.
UNAVAILABLE = "unavailable"


@register_backend("vllm")
class VllmBackend(LLMBackend):
    """Chat completions against a vLLM server or the swap proxy in front of it."""

    name = "vllm"

    #: Beyond temperature, these ModelConfig fields reach the model, so they
    #: belong in the cache key and the repro hash (BACKEND_CAPABILITIES).
    config_fields = ("max_tokens", "reasoning")

    # Deliberately low until measurement B (concurrency invariance at
    # temperature 0) has run: batch composition changes floating-point
    # reduction order, so a higher value is a claim about answers, not just
    # about throughput. Raise it once that measurement says which value holds.
    max_safe_workers = 8

    def __init__(
        self,
        model_name: str,
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        reasoning: bool = False,
        endpoint: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        """Resolve the endpoint, verify it serves this model, and pin what answers.

        Args:
            model_name: The model id as vLLM serves it.
            temperature: Sampling temperature.
            max_tokens: Token budget per answer. For a reasoning model this
                must cover trace AND answer — the trace burns it first.
            reasoning: Whether this model emits a thinking trace.
            endpoint: Base URL; falls back to the VLLM_ENDPOINT env var. The
                node changes with every SLURM job, so it is environment, not
                configuration — it stays out of ModelConfig and out of the
                run identity.
            timeout: Seconds per generation request.

        Raises:
            ValueError: If no endpoint is configured.
            RuntimeError: If the endpoint is unreachable or does not serve
                *model_name*. Fail fast: a stale URL never fixes itself, and
                retrying one costs a night for nothing.
        """
        super().__init__(model_name, temperature)
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        if timeout is None:
            timeout = self.timeout_from_env("VLLM_TIMEOUT_S", REQUEST_TIMEOUT_S)
        self.timeout = timeout

        raw_endpoint = endpoint or self.require_env(
            "VLLM_ENDPOINT", "vLLM endpoint",
            example="http://<host>:<port>",
        )
        self.endpoint = raw_endpoint.rstrip("/")

        self._check_serves_model()
        self.model_digest = self._read_digest()
        logger.info(
            "vLLM backend ready: %s on %s (digest=%s, reasoning=%s)",
            self.model_name, self.endpoint, self.model_digest, self.reasoning,
        )

    # --------------------------------------------------
    # Construction-time probes
    # --------------------------------------------------
    def _check_serves_model(self) -> None:
        """Fail now if the endpoint is wrong or the alias is not loaded.

        Raises:
            RuntimeError: On an unreachable endpoint or an unserved model,
                naming what the server does offer.
        """
        url = f"{self.endpoint}/v1/models"
        try:
            response = requests.get(url, timeout=REACHABILITY_TIMEOUT_S)
            response.raise_for_status()
            available = [entry["id"] for entry in response.json().get("data", [])]
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise RuntimeError(
                f"vLLM endpoint not reachable: {url} ({type(exc).__name__}: {exc}). "
                f"VLLM_ENDPOINT may still point at the node of a finished "
                f"SLURM job."
            ) from exc

        if self.model_name not in available:
            raise RuntimeError(
                f"Model {self.model_name!r} is not served by {self.endpoint}. "
                f"Available: {', '.join(sorted(available)) or '(none)'}"
            )

    def _read_digest(self) -> str:
        """Ask the swap proxy which instrument this alias currently is.

        vLLM exposes no weights digest, so the proxy hashes the served model
        id plus the launch command (quantization, chat template, reasoning
        parser) — for vLLM the launch configuration IS the instrument
        identity. A plain ``vllm serve`` has no such endpoint.

        Not cached across backend instances on purpose: the proxy re-reads
        its model table per request, so a remembered digest could go stale
        within a run, and a stale instrument id is worse than none.

        Returns:
            The short digest, or UNAVAILABLE. Best effort by design — a run
            must not die because a metadata endpoint hiccuped, but the gap
            is recorded rather than hidden.
        """
        try:
            response = requests.get(
                f"{self.endpoint}/health", timeout=REACHABILITY_TIMEOUT_S,
            )
            response.raise_for_status()
            instruments = response.json().get("instruments", {})
        except (requests.RequestException, ValueError):
            return UNAVAILABLE
        entry = instruments.get(self.model_name) or {}
        return entry.get("digest") or UNAVAILABLE

    # --------------------------------------------------
    # Backend API
    # --------------------------------------------------
    def generate(self, prompt: str) -> Answer:
        """Send one prompt and return answer and reasoning trace, separated.

        Args:
            prompt: The rendered prompt.

        Returns:
            The answer, with the trace in its own field.

        Raises:
            BackendContractError: If a model declared ``reasoning: true``
                answered without a separate trace field.
            RuntimeError: On a transport error or a non-2xx response; the
                server's own explanation travels with the message, because
                it says which limit was hit.
        """
        payload: dict[str, Any] = self.chat_payload(prompt)
        payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        try:
            response = requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload, timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"vLLM request to {self.endpoint} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if not response.ok:
            # requests' own message drops the body and leaves a bare
            # "400 Client Error"; vLLM's body names the actual limit, e.g.
            # the model's maximum context length against the prompt size.
            raise RuntimeError(
                f"vLLM returned HTTP {response.status_code} for "
                f"{self.model_name!r}: {response.text[:500]}"
            )

        return self._read_answer(response.json())

    def _read_answer(self, body: dict) -> Answer:
        """Split a chat-completions response into answer and trace.

        Args:
            body: The parsed JSON response.

        Returns:
            The answer with the trace in its own field.

        Raises:
            BackendContractError: If this model is declared as reasoning but
                the response has no trace field — the exact shape of a server
                started without ``--reasoning-parser``.
        """
        message = (body.get("choices") or [{}])[0].get("message") or {}
        content = message.get("content") or ""
        # vLLM 0.28 names it `reasoning`; earlier versions `reasoning_content`.
        thinking = (
            message.get("reasoning_content") or message.get("reasoning") or ""
        )

        if self.reasoning and content.strip() and not thinking.strip():
            raise BackendContractError(
                f"Model {self.model_name!r} is declared reasoning: true, but "
                f"{self.endpoint} returned content with no reasoning_content "
                f"field. That is what a server started WITHOUT --reasoning-parser "
                f"looks like: the trace is mixed into the answer, where it "
                f"spends the token budget before the answer starts and can "
                f"leave a draft verdict as the last JSON object in the "
                f"response — reported as the answer, with no parse error "
                f"(see NOTES.md 2026-09-01 and "
                f"tests/test_parser_trace_robustness.py).\n"
                f"Restart vllm serve with --reasoning-parser for this model, "
                f"or set reasoning: false if it truly does not think."
            )

        return Answer(raw=content, thinking=thinking)
