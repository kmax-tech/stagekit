"""Tests for the hosted OpenAI backend.

Thin by design — the SDK is trusted. What is checked is the contract this
repo depends on: the Answer shape, the temperature quirk of the gpt-5 models,
and that no output-affecting field reaches the model undeclared.
"""

import pytest

from stagekit.models.backends.backends_base import Answer
from stagekit.models.backends.openai_backend import OpenAIBackend


class _FakeCompletions:
    def __init__(self, content, seen, reasoning_content=None):
        self._content = content
        self._seen = seen
        self._reasoning_content = reasoning_content

    def create(self, **kwargs):
        self._seen.update(kwargs)
        if self._content is None:
            return type("R", (), {"choices": []})()
        attrs = {"content": self._content}
        if self._reasoning_content is not None:
            attrs["reasoning_content"] = self._reasoning_content
        message = type("M", (), attrs)()
        choice = type("C", (), {"message": message})()
        return type("R", (), {"choices": [choice]})()


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    seen: dict = {}

    def make(model_name="gpt-4o", temperature=1, content="the answer",
             supports_temperature=True, reasoning_content=None):
        b = OpenAIBackend(model_name=model_name, temperature=temperature,
                          supports_temperature=supports_temperature)
        completions = _FakeCompletions(content, seen, reasoning_content)
        b.client = type("Client", (), {
            "chat": type("Chat", (), {"completions": completions})()
        })()
        return b, seen
    return make


class TestAnswerShape:

    def test_returns_answer_with_empty_trace(self, backend):
        b, _ = backend()
        out = b.generate("p")

        assert isinstance(out, Answer)
        assert out.raw == "the answer"
        assert out.thinking == ""

    def test_empty_content_stays_empty_string(self, backend):
        b, _ = backend(content="")
        assert b.generate("p").raw == ""

    def test_empty_choices_raises(self, backend):
        b, _ = backend(content=None)
        with pytest.raises(ValueError, match="empty choices"):
            b.generate("p")


class TestReasoningTrace:
    """OpenAI-protocol reasoning endpoints return message.reasoning_content
    (deepseek-reasoner runs over this backend). It must land in
    Answer.thinking, never be dropped (audit C4)."""

    def test_reasoning_content_lands_in_thinking(self, backend):
        b, _ = backend(model_name="deepseek-reasoner",
                       reasoning_content="chain of thought")
        out = b.generate("p")

        assert out == Answer(raw="the answer", thinking="chain of thought")

    def test_absent_field_means_empty_thinking(self, backend):
        b, _ = backend()  # hosted OpenAI: message has no reasoning_content
        assert b.generate("p").thinking == ""

    def test_null_reasoning_content_stays_empty_string(self, backend):
        """The DeepSeek API sends reasoning_content: null on some calls —
        an explicit None must normalize to ''."""
        b, _ = backend(reasoning_content="ignored")
        b.client.chat.completions._reasoning_content = "x"

        class _NullMessage:
            content = "the answer"
            reasoning_content = None

        completions = b.client.chat.completions
        completions.create = lambda **kwargs: type("R", (), {
            "choices": [type("C", (), {"message": _NullMessage()})()]
        })()
        assert b.generate("p").thinking == ""


class TestTemperature:

    def test_sent_when_it_deviates(self, backend):
        b, seen = backend(temperature=0.2)
        b.generate("p")

        assert seen["temperature"] == 0.2

    def test_omitted_at_the_default(self, backend):
        """The API default is 1; sending it explicitly is noise."""
        b, seen = backend(temperature=1)
        b.generate("p")

        assert "temperature" not in seen

    def test_rejecting_model_with_override_fails_loud(self, backend):
        """supports_temperature comes from the model YAML now (audit B2).

        The old backend silently forced temperature=1 for a hardcoded ID
        list — a new snapshot escaped the guard, and the config lied
        about the run. Now the combination raises.
        """
        with pytest.raises(ValueError, match="supports_temperature"):
            backend(temperature=0.2, supports_temperature=False)

    def test_rejecting_model_at_default_temperature_is_fine(self, backend):
        b, seen = backend(temperature=1, supports_temperature=False)
        b.generate("p")

        assert "temperature" not in seen


class TestConfiguration:

    def test_missing_key_fails_loud(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIBackend(model_name="gpt-4o")

    def test_declares_supports_temperature_only(self):
        """max_tokens/reasoning are not sent, so they must not key the cache;
        supports_temperature is construction input, not a cache-key field
        (it is not in any backend's identity_fields)."""
        assert OpenAIBackend.config_fields == ("supports_temperature",)
