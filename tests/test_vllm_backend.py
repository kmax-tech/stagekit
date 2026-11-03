"""Tests for the vLLM backend, above all its refusal of a misconfigured server.

The guard is the reason this backend exists as its own class. A vLLM server
started without --reasoning-parser returns a reasoning model's thinking trace
inside `content`. That produces no parse error and no warning — just answers
derived from the deliberation instead of the conclusion. So the failure has
to be made loud here, and it has to be fatal rather than retried.
"""

import pytest
import requests

import stagekit.models.backends.vllm_backend as vllm_module
from stagekit.config.classes import ModelConfig
from stagekit.models.backends.backends_base import BackendContractError
from stagekit.models.backends.factory import build_backend
from stagekit.models.backends.vllm_backend import UNAVAILABLE, VllmBackend

ENDPOINT = "http://node:11447"


class _Resp:
    """Minimal stand-in for requests.Response."""

    def __init__(self, payload=None, status=200, text=""):
        self._payload = payload
        self.status_code = status
        self.text = text or str(payload)

    @property
    def ok(self):
        return self.status_code < 400

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _chat(content, thinking=None, field="reasoning_content"):
    message = {"content": content}
    if thinking is not None:
        message[field] = thinking
    return {"choices": [{"message": message}]}


@pytest.fixture
def server(monkeypatch):
    """A fake vLLM server whose two probe responses and answer are settable."""
    state = {
        "models": _Resp({"data": [{"id": "m"}]}),
        "health": _Resp({"instruments": {"m": {"digest": "abc123def456"}}}),
        "answer": _Resp(_chat("the answer")),
        "posts": [],
    }

    def fake_get(url, **kwargs):
        return state["health"] if url.endswith("/health") else state["models"]

    def fake_post(url, **kwargs):
        state["posts"].append(kwargs.get("json"))
        return state["answer"]

    monkeypatch.setattr(vllm_module.requests, "get", fake_get)
    monkeypatch.setattr(vllm_module.requests, "post", fake_post)
    monkeypatch.setenv("VLLM_ENDPOINT", ENDPOINT)
    return state


class TestConstruction:

    def test_missing_endpoint_fails_loud(self, monkeypatch):
        monkeypatch.delenv("VLLM_ENDPOINT", raising=False)
        with pytest.raises(ValueError, match="VLLM_ENDPOINT"):
            VllmBackend(model_name="m")

    def test_unreachable_endpoint_fails_at_construction(self, server, monkeypatch):
        """A stale node URL never fixes itself — do not carry it into the run."""
        def boom(url, **kwargs):
            raise requests.ConnectionError("no route to host")
        monkeypatch.setattr(vllm_module.requests, "get", boom)

        with pytest.raises(RuntimeError, match="not reachable"):
            VllmBackend(model_name="m")

    def test_unserved_model_names_the_alternatives(self, server):
        server["models"] = _Resp({"data": [{"id": "other"}]})
        with pytest.raises(RuntimeError, match="Available: other"):
            VllmBackend(model_name="m")

    def test_digest_pins_the_instrument(self, server):
        assert VllmBackend(model_name="m").model_digest == "abc123def456"

    def test_digest_gap_is_recorded_not_fatal(self, server):
        """A plain `vllm serve` has no /health payload — record the gap."""
        server["health"] = _Resp(status=404)
        assert VllmBackend(model_name="m").model_digest == UNAVAILABLE


class TestReasoningGuard:

    def test_separated_trace_is_kept_apart(self, server):
        server["answer"] = _Resp(_chat("Argument", thinking="Okay, so I need to"))
        out = VllmBackend(model_name="m", reasoning=True).generate("p")

        assert out.raw == "Argument"
        assert out.thinking == "Okay, so I need to"

    def test_newer_vllm_field_name_accepted(self, server):
        """vLLM 0.28 renamed reasoning_content to reasoning."""
        server["answer"] = _Resp(_chat("Argument", thinking="trace", field="reasoning"))
        out = VllmBackend(model_name="m", reasoning=True).generate("p")

        assert out.thinking == "trace"

    def test_inline_trace_refused(self, server):
        """The whole point: an unconfigured server must not answer quietly."""
        server["answer"] = _Resp(_chat("Okay, so I need to figure out... Argument"))
        backend = VllmBackend(model_name="m", reasoning=True)

        with pytest.raises(BackendContractError, match="--reasoning-parser"):
            backend.generate("p")

    def test_non_reasoning_model_is_not_policed(self, server):
        """A model that does not think legitimately returns content only."""
        out = VllmBackend(model_name="m", reasoning=False).generate("p")

        assert out.raw == "the answer"
        assert out.thinking == ""

    def test_empty_content_does_not_trip_the_guard(self, server):
        """An empty answer is a different failure; do not mislabel it."""
        server["answer"] = _Resp(_chat(""))
        out = VllmBackend(model_name="m", reasoning=True).generate("p")

        assert out.raw == ""


class TestRequestShape:

    def test_max_tokens_sent_only_when_set(self, server):
        VllmBackend(model_name="m", max_tokens=4096).generate("p")
        VllmBackend(model_name="m").generate("p")

        assert server["posts"][0]["max_tokens"] == 4096
        assert "max_tokens" not in server["posts"][1]

    def test_http_error_carries_the_servers_explanation(self, server):
        """vLLM says which limit was hit; requests' own message drops it."""
        server["answer"] = _Resp(
            status=400,
            text="maximum context length is 8192 tokens. However, your prompt...",
        )
        backend = VllmBackend(model_name="m")

        with pytest.raises(RuntimeError, match="maximum context length"):
            backend.generate("p")


class TestFactory:

    def test_build_backend_passes_declared_fields(self, server):
        backend = build_backend(ModelConfig(
            model_type="llm", backend="vllm", model_name="m",
            temperature=0.0, max_tokens=2048, reasoning=True,
        ))

        assert backend.name == "vllm"
        assert backend.max_tokens == 2048
        assert backend.reasoning is True

    def test_identity_fields_declared(self):
        """A knob that reaches the model must also key the cache and hash."""
        from stagekit.models.backends.factory import BACKEND_CAPABILITIES
        declared = BACKEND_CAPABILITIES["vllm"]["identity_fields"]
        assert set(VllmBackend.config_fields) <= declared


class TestTimeoutEnvKnob:
    """VLLM_TIMEOUT_S is the per-node timeout knob (audit B7) — the same
    timeout_from_env mechanism as OLLAMA_TIMEOUT_S/CLI_TIMEOUT_S."""

    def test_env_var_overrides_default(self, server, monkeypatch):
        monkeypatch.setenv("VLLM_TIMEOUT_S", "90")
        backend = VllmBackend(model_name="m")
        assert backend.timeout == 90

    def test_default_without_env(self, server, monkeypatch):
        monkeypatch.delenv("VLLM_TIMEOUT_S", raising=False)
        backend = VllmBackend(model_name="m")
        assert backend.timeout == vllm_module.REQUEST_TIMEOUT_S

    def test_bad_value_fails_loud(self, server, monkeypatch):
        monkeypatch.setenv("VLLM_TIMEOUT_S", "soon")
        with pytest.raises(ValueError, match="VLLM_TIMEOUT_S"):
            VllmBackend(model_name="m")
