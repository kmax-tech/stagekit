"""Tests for the ollama backend, focused on the trace it already separates.

ollama returns the reasoning trace in its own `thinking` field, so this
backend needs no guard — but it must actually pass the field on, or the one
backend that gets separation for free would be the one that drops it.
"""

import pytest

import stagekit.models.backends.ollama_backend as ollama_module
from stagekit.models.backends.ollama_backend import OllamaBackend

URL = "http://node:11434"


class _FakeClient:
    """Stands in for ollama.Client; records what it was asked."""

    last_kwargs: dict = {}

    def __init__(self, response, host=None, timeout=None):
        self._response = response
        self.host = host

    def list(self):
        return {"models": [type("M", (), {"model": "m"})()]}

    def generate(self, **kwargs):
        _FakeClient.last_kwargs = kwargs
        return self._response


@pytest.fixture
def backend(monkeypatch):
    def make(response):
        monkeypatch.setattr(
            ollama_module.ollama, "Client",
            lambda host=None, timeout=None: _FakeClient(response, host, timeout),
        )
        return OllamaBackend(model_name="m", temperature=0.3, server_urls=[URL])
    return make


class TestAnswerShape:

    def test_trace_passed_through_when_present(self, backend):
        out = backend({"response": "the answer", "thinking": "the trace"}).generate("p")

        assert out.raw == "the answer"
        assert out.thinking == "the trace"

    def test_absent_trace_becomes_empty_string(self, backend):
        """Most models have no thinking field at all — that is not an error."""
        out = backend({"response": "the answer"}).generate("p")

        assert out.raw == "the answer"
        assert out.thinking == ""

    def test_null_fields_do_not_become_none(self, backend):
        """A None must not reach the parser or the cache as a value."""
        out = backend({"response": None, "thinking": None}).generate("p")

        assert out.raw == ""
        assert out.thinking == ""

    def test_temperature_is_sent(self, backend):
        backend({"response": "x"}).generate("p")

        assert _FakeClient.last_kwargs["options"] == {"temperature": 0.3}


class TestConfiguration:

    def test_missing_server_fails_loud(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_SERVER", raising=False)
        with pytest.raises(ValueError, match="OLLAMA_SERVER"):
            OllamaBackend(model_name="m")

    def test_worker_cap_follows_server_count(self, monkeypatch):
        monkeypatch.setattr(
            ollama_module.ollama, "Client",
            lambda host=None, timeout=None: _FakeClient({"response": "x"}, host),
        )
        backend = OllamaBackend(model_name="m", server_urls=[URL, "http://other:11434"])

        assert backend.max_safe_workers == 2

    def test_declares_no_extra_config_fields(self):
        """max_tokens/reasoning are not sent, so they must not key the cache."""
        assert OllamaBackend.config_fields == ()

    def test_timeout_env_var_reaches_the_client(self, monkeypatch):
        """OLLAMA_TIMEOUT_S is the only timeout knob that reaches
        production calls — build_backend passes no timeout (audit B7)."""
        monkeypatch.setenv("OLLAMA_TIMEOUT_S", "900")
        monkeypatch.setattr(
            ollama_module.ollama, "Client",
            lambda host=None, timeout=None: _FakeClient({"response": "x"}, host),
        )
        backend = OllamaBackend(model_name="m", server_urls=[URL])
        assert backend.timeout == 900

    def test_bad_timeout_env_var_fails_loud(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_TIMEOUT_S", "fast")
        with pytest.raises(ValueError, match="OLLAMA_TIMEOUT_S"):
            OllamaBackend(model_name="m", server_urls=[URL])


class TestModelPulledCheck:
    """Tag-less config names must match the ':latest' tag ollama lists.

    Audit A7: exact-only matching never matched 'llama3.2' against the
    listed 'llama3.2:latest', so every construction re-pulled — and
    failed hard on offline nodes with the weights already present.
    """

    def _client_listing(self, *names):
        class _C:
            def list(self):
                return {"models": [type("M", (), {"model": n})() for n in names]}
        return _C()

    def _backend(self, monkeypatch, model_name):
        # Bypass __init__ — only _is_model_pulled is under test.
        b = object.__new__(OllamaBackend)
        b.model_name = model_name
        return b

    def test_tagless_name_matches_latest(self, monkeypatch):
        b = self._backend(monkeypatch, "llama3.2")
        assert b._is_model_pulled(self._client_listing("llama3.2:latest"))

    def test_tagless_name_matches_exact(self, monkeypatch):
        b = self._backend(monkeypatch, "llama3.2")
        assert b._is_model_pulled(self._client_listing("llama3.2"))

    def test_tagged_name_requires_exact_tag(self, monkeypatch):
        b = self._backend(monkeypatch, "deepseek-r1:8b")
        assert b._is_model_pulled(self._client_listing("deepseek-r1:8b"))
        assert not b._is_model_pulled(self._client_listing("deepseek-r1:latest"))

    def test_tagless_does_not_match_other_tags(self, monkeypatch):
        """'llama3.2' means ':latest' — a node holding only ':8b' must pull."""
        b = self._backend(monkeypatch, "llama3.2")
        assert not b._is_model_pulled(self._client_listing("llama3.2:8b"))
