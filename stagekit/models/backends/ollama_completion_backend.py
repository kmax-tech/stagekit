# stagekit/models/backends/ollama_completion_backend.py
"""ollama raw-completion backend: the prompt reaches the model untemplated.

The ``ollama`` backend calls ``/api/generate`` without ``raw``, so the
server wraps every prompt in the model's chat template. That is right for
a chat prompt and wrong for a prompt that carries its own instruction
format — CASA's ``### Instruction / ### Input / ### Response`` (Liu et
al., NAACL 2024) would be templated twice. This backend sends
``raw=True`` and, because a raw completion has no natural end, pins the
answer budget (``num_predict``), the stop sequences and the context
window (``num_ctx``; ollama's default depends on the server version and
a too-long prompt is truncated from the front, silently).

A second backend rather than three more fields on ``ollama``: every
field a backend consumes keys the cache and the repro hash of ALL its
runs (``base._backend_params``, ``repro_hash._model_canon``), so widening
``ollama`` would re-key every stored ollama entry. The same model id on
two backends is two instruments (NOTES.md 2026-09-15).

Instrument identity is read at construction and recorded in run
metadata, never in the config (an observation, not an input): the digest
of the served weights (``/api/tags`` — an ollama tag is mutable) and the
server version (``/api/version`` — the quantised runtime and its
defaults are version properties). Servers that disagree on the digest
are two instruments behind one alias and refuse to start.
"""
import logging

import ollama
import requests

from stagekit.models.backends.backends_base import Answer
from stagekit.models.backends.factory import register_backend
from stagekit.models.backends.ollama_backend import OllamaBackend

logger = logging.getLogger(__name__)

#: Recorded when a metadata probe fails: the gap is written down, not hidden.
UNAVAILABLE = "unavailable"

#: Budget for the metadata probes; they answer from memory, not from a model.
PROBE_TIMEOUT_S = 10

#: Length of the recorded digest prefix (sha256 is 64 hex chars; 12 identify).
DIGEST_CHARS = 12


@register_backend("ollama_completion")
class OllamaCompletionBackend(OllamaBackend):
    """ollama ``/api/generate`` with ``raw=True`` and pinned generation knobs.

    Inherits server discovery, model pulling and the round-robin client
    from ``OllamaBackend``; differs in what one call sends and in the
    instrument identity it records.

    Config fields (``ModelConfig``): ``max_tokens`` -> ``num_predict``,
    ``stop`` -> stop sequences, ``num_ctx`` -> context window. All three
    are identity fields of this backend (BACKEND_CAPABILITIES).
    """

    name = "ollama_completion"
    config_fields = ("max_tokens", "stop", "num_ctx")

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        num_ctx: int | None = None,
        server_urls: list[str] | None = None,
        timeout: int | None = None,
    ):
        """Resolve servers, pull the model, then read the instrument identity.

        Args:
            model_name: ollama model tag, e.g. ``llama2:7b-chat``.
            temperature: Sampling temperature (sent as an option).
            max_tokens: Answer budget in tokens (``num_predict``); None
                leaves the server default, which for a raw completion
                means "until the context is full".
            stop: Sequences at which generation stops; None sends none.
            num_ctx: Context window in tokens; None leaves the server default.
            server_urls: Explicit server list; default from ``OLLAMA_SERVER``.
            timeout: Per-request timeout; default from ``OLLAMA_TIMEOUT_S``.

        Raises:
            RuntimeError: If the servers disagree on the model's digest.
        """
        super().__init__(model_name, temperature, server_urls=server_urls, timeout=timeout)
        self.max_tokens = max_tokens
        self.stop = list(stop) if stop else None
        self.num_ctx = num_ctx
        self.model_digest = self._read_model_digest()
        self.server_version = self._read_server_version()
        logger.info(
            "ollama_completion ready: %s (digest=%s, server=%s, num_predict=%s, "
            "stop=%s, num_ctx=%s)",
            self.model_name, self.model_digest, self.server_version,
            self.max_tokens, self.stop, self.num_ctx,
        )

    # --------------------------------------------------
    # Construction-time probes
    # --------------------------------------------------
    def _read_model_digest(self) -> str:
        """Digest of the served weights, agreed across all servers.

        The pull check has just confirmed the model on every server, so a
        missing entry here is a server-side inconsistency, not a race.

        Returns:
            The first ``DIGEST_CHARS`` of the sha256 digest.

        Raises:
            RuntimeError: If a server lists no digest for the model, or
                two servers list different ones (two instruments behind
                one alias; a run must not mix them).
        """
        digests: dict[str, str] = {}
        for url in self.server_urls:
            entry = self._find_listed(ollama.Client(host=url, timeout=self.timeout))
            digest = getattr(entry, "digest", None) if entry is not None else None
            if not digest:
                raise RuntimeError(
                    f"{url} lists no digest for {self.model_name!r} although the "
                    f"pull check passed — cannot record which weights answer."
                )
            digests[url] = str(digest)[:DIGEST_CHARS]
        if len(set(digests.values())) > 1:
            raise RuntimeError(
                f"Servers disagree on the weights of {self.model_name!r}: {digests}. "
                f"One alias, two instruments — pull the same digest everywhere."
            )
        return next(iter(digests.values()))

    def _read_server_version(self) -> str:
        """ollama server version, agreed across all servers; best effort.

        The python client exposes no version call, so this is one HTTP
        GET per server. A failed probe records UNAVAILABLE rather than
        killing the run: the version is metadata, the digest above is
        the identity that must not be missing.
        """
        versions: set[str] = set()
        for url in self.server_urls:
            try:
                response = requests.get(
                    f"{url.rstrip('/')}/api/version", timeout=PROBE_TIMEOUT_S,
                )
                response.raise_for_status()
                versions.add(str(response.json().get("version") or UNAVAILABLE))
            except (requests.RequestException, ValueError):
                versions.add(UNAVAILABLE)
        if len(versions) > 1:
            logger.warning(
                "ollama servers report different versions: %s", sorted(versions),
            )
            return "|".join(sorted(versions))
        return versions.pop()

    # --------------------------------------------------
    # Backend API
    # --------------------------------------------------
    def generate(self, prompt: str) -> Answer:
        """One untemplated completion on the next server in the round-robin.

        Only knobs that are set are sent, so a None leaves the server
        default and the payload says exactly what the config said.
        """
        options: dict[str, object] = {"temperature": self.temperature}
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens
        if self.stop:
            options["stop"] = self.stop
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        client = self._next_client()
        resp = client.generate(
            model=self.model_name,
            prompt=prompt,
            raw=True,
            options=options,
        )
        return Answer(
            raw=resp.get("response", "") or "",
            thinking=resp.get("thinking", "") or "",
        )
