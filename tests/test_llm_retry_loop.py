"""Tests for the retry loop in LLMBaseModel: what is retried, and how it waits.

The distinction matters operationally. A transient error (timeout, 5xx) is
worth another attempt; a contract violation is not, because the server is
misconfigured and every item would fail the same way. Without a type-based
rule, a BackendContractError falls through the message-based classification
and gets retried once per item against a server that cannot answer.

The waiting is the second half of the same decision. A parse failure is a
200 OK with the wrong shape — nothing to wait for. Every other retryable
error is an endpoint under stress, so those back off exponentially. Getting
that branch backwards costs either a hammered server or an hour of sleeping
through parse failures on a large batch.
"""

import pytest

import stagekit.models.base as base_module
from stagekit.models.backends.backends_base import Answer, BackendContractError
from stagekit.models.base import LLMBaseModel
from tests.llm_harness import FakeBackend, FakeParser, HarnessModel

#: Message -> error type, for errors the loop must give up on.
FATAL_MESSAGES = [
    ("401 Unauthorized", "auth"),
    ("403 Forbidden", "auth"),
    ("Authentication failed for the configured key", "auth"),
    ("400 Bad Request", "invalid_request"),
    ("The request body is invalid", "invalid_request"),
]

#: Message -> error type, for errors another attempt may well clear.
TRANSIENT_MESSAGES = [
    ("Error code: 429 - too many requests, rate limit exceeded", "rate_limit"),
    ("Read timeout after 30s", "timeout"),
    ("Connection refused", "connection"),
    ("500 Internal Server Error", "server_error"),
    ("502 Bad Gateway", "server_error"),
    ("503 Service Unavailable", "server_error"),
    ("upstream server error while generating", "server_error"),
]


@pytest.fixture(autouse=True)
def sleeps(monkeypatch) -> list[float]:
    """Record backoff delays instead of serving them.

    Returns:
        The list of delays passed to ``time.sleep``, in order.
    """
    recorded: list[float] = []
    monkeypatch.setattr(base_module.time, "sleep", recorded.append)
    return recorded


def _call(model: HarnessModel, max_retries: int):
    return model._execute_llm_call(
        prompt="p",
        parser_fields={},
        max_retries=max_retries,
        prompt_version="v1",
        item_id="i1",
    )


class TestErrorClassification:
    def test_contract_error_classified_by_type(self):
        """No message marker exists — the type is the only signal."""
        exc = BackendContractError("server started without --reasoning-parser")
        assert LLMBaseModel._classify_backend_error(exc) == "backend_contract"

    def test_unknown_error_falls_back_to_type_name(self):
        assert LLMBaseModel._classify_backend_error(ValueError("boom")) == "ValueError"

    def test_contract_error_is_fatal(self):
        assert "backend_contract" in base_module._FATAL_ERROR_TYPES

    @pytest.mark.parametrize("message,expected", FATAL_MESSAGES)
    def test_fatal_messages_stop_the_loop(self, message: str, expected: str):
        """These describe the request or the key — retrying cannot change them."""
        assert LLMBaseModel._classify_backend_error(RuntimeError(message)) == expected
        assert expected in base_module._FATAL_ERROR_TYPES

    @pytest.mark.parametrize("message,expected", TRANSIENT_MESSAGES)
    def test_transient_messages_stay_retryable(self, message: str, expected: str):
        """The load-shedding cases. Classifying any of them fatal would turn a
        busy endpoint into a run-wide data loss."""
        assert LLMBaseModel._classify_backend_error(RuntimeError(message)) == expected
        assert expected not in base_module._FATAL_ERROR_TYPES

    def test_invalid_alone_does_not_mean_invalid_request(self):
        """The ``and`` in that guard is load-bearing: 'invalid' shows up in
        plenty of transient wordings, and only 'invalid ... request' is the
        fatal one."""
        exc = RuntimeError("invalid json in response body")
        assert LLMBaseModel._classify_backend_error(exc) == "RuntimeError"

    def test_rate_limit_quoting_a_token_budget_stays_retryable(self):
        """A real OpenAI 429 body carries numbers that a substring test for
        '400' reads as a bad request — fatal, for an error the next attempt
        would have cleared. Status codes are matched as whole tokens."""
        exc = RuntimeError(
            "Rate limit reached for gpt-4 in organization org-x on tokens per "
            "min. Limit: 40000 / min. Current: 41000 / min."
        )
        assert LLMBaseModel._classify_backend_error(exc) == "rate_limit"


class TestRetryBehaviour:
    def test_contract_error_stops_after_one_attempt(self):
        """The point of the type-based rule: no retry storm on a broken server."""
        backend = FakeBackend(error=BackendContractError("misconfigured"))
        model = HarnessModel(backend=backend)

        result = _call(model, max_retries=5)

        assert backend.calls == 1
        assert result.success is False
        assert result.error_type == "backend_contract"
        # The count must reflect what happened, not the budget: reporting
        # attempts=6 after one call is a failure statistic that overstates
        # itself, and the fatal path is where that became routine.
        assert result.attempts == 1

    def test_transient_error_exhausts_retries(self):
        """Contrast case — proves the fatal rule is what stops the loop."""
        backend = FakeBackend(error=ValueError("transient"))
        model = HarnessModel(backend=backend)

        result = _call(model, max_retries=3)

        assert backend.calls == 4  # initial attempt + 3 retries
        assert result.attempts == 4
        assert result.success is False

    def test_successful_call_returns_parsed_value(self):
        model = HarnessModel(backend=FakeBackend())

        result = _call(model, max_retries=2)

        assert result.success is True
        assert result.value == {"text": "fine"}

    def test_retry_counter_counts_retries_not_attempts(self):
        """``retries`` lands in the per-stage stats file; off by one there is
        an off-by-one in every reported retry rate."""
        model = HarnessModel(backend=FakeBackend(error=ValueError("transient")))

        _call(model, max_retries=3)

        assert model.llm_stats.retries == 3

    def test_raw_of_the_last_failing_attempt_is_reported(self):
        """The final-failure log is the only place a malformed answer is
        visible; it must show what the model actually last said."""
        model = HarnessModel(
            backend=FakeBackend(script=[Answer(raw="first"), Answer(raw="second")]),
            parser=FakeParser(script=["none", "none"]),
        )

        result = _call(model, max_retries=1)

        assert result.success is False
        assert result.raw_response == "second"

    def test_fatal_abort_is_not_filed_as_retry_exhaustion(self):
        """The item did fail — but not from exhausting retries, and that is
        the difference between 'the endpoint is flaky' and 'the server is
        misconfigured' when reading a stage's stats afterwards."""
        model = HarnessModel(backend=FakeBackend(error=BackendContractError("bad")))

        _call(model, max_retries=5)

        assert model.llm_stats.retry_exhausted_items == {}
        assert "fake:retries_exhausted" not in model.llm_stats.errors
        # It still counts as a failed item — only the reason is wrong.
        assert model.llm_stats.failed_items == {"fake": ["i1"]}

    def test_error_message_and_type_describe_the_same_attempt(self):
        """Carried forward separately, a parse failure that follows a backend
        error reports the backend error's message under the parse failure's
        type — two attempts fused into one bogus report."""
        model = HarnessModel(
            backend=FakeBackend(script=[ValueError("boom")], answer=Answer(raw="junk")),
            parser=FakeParser(script=["none"]),
        )

        result = _call(model, max_retries=1)

        assert result.error_type == "parse_failure"
        assert result.error != "boom"


class TestBackoffSemantics:
    def test_parse_failure_retries_without_waiting(self, sleeps: list[float]):
        """A 200 OK in the wrong shape has nothing to wait out. Backing off
        here would add minutes per item to a batch that reparses fine."""
        model = HarnessModel(
            backend=FakeBackend(answer=Answer(raw="junk")),
            parser=FakeParser(script=["none", "none", "none"]),
        )

        result = _call(model, max_retries=2)

        assert result.error_type == "parse_failure"
        assert result.attempts == 3
        assert sleeps == []

    def test_transient_error_waits_between_attempts_only(self, sleeps: list[float]):
        """Exponential from 0.5 s with up to 0.5 s of jitter, and no wait after
        the last attempt — sleeping once more would delay the whole batch for
        an attempt that is never made."""
        model = HarnessModel(backend=FakeBackend(error=ValueError("transient")))

        _call(model, max_retries=3)

        assert len(sleeps) == 3
        for attempt, delay in enumerate(sleeps):
            floor = 0.5 * 2**attempt
            assert floor <= delay < floor + 0.5

    def test_backoff_is_capped(self, sleeps: list[float]):
        """Without the cap, the default of 10 retries ends in waits of hours."""
        model = HarnessModel(backend=FakeBackend(error=ValueError("transient")))

        _call(model, max_retries=8)

        assert 30.0 <= sleeps[-1] < 30.5
        assert max(sleeps) < 30.5

    def test_fatal_error_never_waits(self, sleeps: list[float]):
        model = HarnessModel(backend=FakeBackend(error=BackendContractError("bad")))

        _call(model, max_retries=5)

        assert sleeps == []

    def test_waiting_stops_once_an_attempt_succeeds(self, sleeps: list[float]):
        model = HarnessModel(
            backend=FakeBackend(script=[ValueError("transient")], answer=Answer("ok")),
        )

        result = _call(model, max_retries=5)

        assert result.success is True
        assert result.attempts == 2
        assert len(sleeps) == 1
