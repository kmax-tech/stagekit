# stagekit/models/backends/ollama_backend.py
import logging

import ollama
from threading import Lock

from stagekit.models.backends.backends_base import Answer, LLMBackend
from stagekit.models.backends.factory import register_backend

logger = logging.getLogger(__name__)

#: Budget for pulling model weights (multi-GB downloads are legitimate);
#: distinct from the per-request timeout. The library default of None
#: would hang forever.
PULL_TIMEOUT_S = 3600

#: Default per-request timeout; override per node via OLLAMA_TIMEOUT_S
#: (LLMBackend.timeout_from_env — a reasoning model can need far more).
DEFAULT_REQUEST_TIMEOUT_S = 120


@register_backend("ollama")
class OllamaBackend(LLMBackend):
    """
    Ollama backend adapter.

    Responsibilities:
    - Server discovery
    - Model availability checks
    - Model pulling
    - Thread-safe round-robin client selection
    """

    name = "ollama"

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        server_urls: list[str] | None = None,
        timeout: int | None = None,
    ):
        super().__init__(model_name, temperature)
        # build_backend passes no timeout; the env var is the only knob
        # that reaches production calls (see DEFAULT_REQUEST_TIMEOUT_S).
        if timeout is None:
            timeout = self.timeout_from_env(
                "OLLAMA_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S,
            )

        # --------------------------------------------------
        # Resolve server URLs
        # --------------------------------------------------
        if server_urls is None:
            raw = self.require_env("OLLAMA_SERVER", "Ollama server address")
            server_urls = [s.strip() for s in raw.split(",") if s.strip()]

        if not server_urls:
            raise ValueError("No valid Ollama server URLs found.")

        self.server_urls = server_urls
        self.timeout = timeout

        # --------------------------------------------------
        # Thread-safe round-robin state
        # --------------------------------------------------
        self._idx = 0
        self._lock = Lock()

        # --------------------------------------------------
        # Ensure model is available on all servers
        # --------------------------------------------------
        self._ensure_model_pulled()

        # --------------------------------------------------
        # Backend concurrency hint
        # --------------------------------------------------
        self.max_safe_workers = len(self.server_urls)

    # --------------------------------------------------
    # Client management
    # --------------------------------------------------
    def _next_client(self) -> ollama.Client:
        """Return a fresh client for the next server URL (round-robin).

        A new client is created per call intentionally — ollama.Client
        is not thread-safe for concurrent requests, and the pipeline
        runs up to max_workers threads in parallel.  On localhost this
        is cheap; connection exhaustion is not a concern.
        """
        with self._lock:
            url = self.server_urls[self._idx]
            self._idx = (self._idx + 1) % len(self.server_urls)
        return ollama.Client(host=url, timeout=self.timeout)

    # --------------------------------------------------
    # Model availability
    # --------------------------------------------------
    def _find_listed(self, client: ollama.Client):
        """The server's entry for the configured model, or None.

        Ollama lists models with an explicit tag (``llama3.2:latest``)
        while configs may name them tag-less (``llama3.2``) — the two are
        the same model. An exact-only comparison never matched tag-less
        names, so every backend construction re-pulled (and failed hard
        on offline nodes with the weights already present). One home for
        that matching: the pull check and the digest probe both use it.
        """
        wanted = self.model_name
        accepted = {wanted} if ":" in wanted else {wanted, f"{wanted}:latest"}
        for entry in client.list().get("models", []):
            if entry.model in accepted:
                return entry
        return None

    def _is_model_pulled(self, client: ollama.Client) -> bool:
        """Whether the server already has the configured model."""
        return self._find_listed(client) is not None

    def _ensure_model_pulled(self):
        for url in self.server_urls:
            # The availability check gets the normal request timeout; the
            # default client has NONE and would hang forever on a dead
            # server. The pull gets its own generous budget — multi-GB
            # downloads are legitimate, an unbounded hang is not.
            if self._is_model_pulled(ollama.Client(host=url, timeout=self.timeout)):
                continue

            logger.info("Pulling model '%s' on %s...", self.model_name, url)
            ollama.Client(host=url, timeout=PULL_TIMEOUT_S).pull(self.model_name)
            logger.info("Model '%s' pulled on %s", self.model_name, url)

        logger.info("Model '%s' verified on all servers.", self.model_name)

    # --------------------------------------------------
    # Backend API
    # --------------------------------------------------
    def generate(self, prompt: str) -> Answer:
        """Send one prompt to the next server in the round-robin.

        Returns:
            The answer plus the reasoning trace. ollama keeps the trace in
            its own ``thinking`` field, so there is nothing to separate
            here; the field is absent for models that do not think.
        """
        client = self._next_client()
        resp = client.generate(
            model=self.model_name,
            prompt=prompt,
            options={"temperature": self.temperature},
        )
        return Answer(
            raw=resp.get("response", "") or "",
            thinking=resp.get("thinking", "") or "",
        )
