"""Tests for the CLI subprocess backends (claude_cli / codex_cli).

The load-bearing contract, per design: the CLIs run as BARE models —
empty sandbox cwd, no tools, no MCP servers, no system prompt, no
session persistence. Tests pin those flags so a convenience edit can't
quietly turn the backend into an agent.
"""

import json
import subprocess

import pytest

from stagekit.config.classes import ModelConfig
from stagekit.models.backends.cli_backend import (
    _SANDBOX,
    ClaudeCliBackend,
    CodexCliBackend,
)
from stagekit.models.backends.factory import build_backend


def _ok(stdout: str):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _claude_payload(result="the answer", subtype="success", is_error=False):
    return json.dumps({"is_error": is_error, "subtype": subtype, "result": result})


class TestClaudeCli:

    def test_sandboxed_bare_model_invocation(self, monkeypatch):
        """Empty cwd + no tools/MCP/system prompt — the contract."""
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["kwargs"] = kwargs
            return _ok(_claude_payload())

        monkeypatch.setattr(subprocess, "run", fake_run)
        out = ClaudeCliBackend(model_name="claude-opus-4-6").generate("hi")

        assert out.raw == "the answer"
        assert out.thinking == ""
        cmd = seen["cmd"]
        assert cmd[:2] == ["claude", "-p"]
        assert ["--tools", ""] == cmd[cmd.index("--tools"):cmd.index("--tools") + 2]
        assert '{"mcpServers":{}}' in cmd
        assert "--no-session-persistence" in cmd
        assert ["--system-prompt", ""] == cmd[cmd.index("--system-prompt"):cmd.index("--system-prompt") + 2]
        # empty sandbox dir, prompt via stdin
        assert seen["kwargs"]["cwd"] == str(_SANDBOX)
        assert seen["kwargs"]["input"] == "hi"

    def test_error_payload_raises(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run",
                            lambda *a, **k: _ok(_claude_payload(is_error=True, subtype="error")))
        with pytest.raises(RuntimeError, match="error payload"):
            ClaudeCliBackend(model_name="m").generate("hi")

    def test_unparsable_wrapper_raises(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _ok("not json"))
        with pytest.raises(RuntimeError, match="unparsable"):
            ClaudeCliBackend(model_name="m").generate("hi")

    def test_nonzero_exit_raises(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: subprocess.CompletedProcess([], 1, stdout="boom", stderr=""),
        )
        with pytest.raises(RuntimeError, match="exit code 1"):
            ClaudeCliBackend(model_name="m").generate("hi")

    def test_timeout_raises(self, monkeypatch):
        def raise_timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd="claude", timeout=300)
        monkeypatch.setattr(subprocess, "run", raise_timeout)
        with pytest.raises(RuntimeError, match="timed out"):
            ClaudeCliBackend(model_name="m").generate("hi")

    def test_missing_binary_raises_helpfully(self, monkeypatch):
        def raise_oserror(*a, **k):
            raise FileNotFoundError("No such file: claude")
        monkeypatch.setattr(subprocess, "run", raise_oserror)
        with pytest.raises(RuntimeError, match="on PATH"):
            ClaudeCliBackend(model_name="m").generate("hi")


class TestCodexCli:

    def test_readonly_ephemeral_invocation(self, monkeypatch):
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["kwargs"] = kwargs
            return _ok("  raw text out  ")

        monkeypatch.setattr(subprocess, "run", fake_run)
        out = CodexCliBackend(model_name="gpt-5.4").generate("hi")

        assert out.raw == "raw text out"
        assert out.thinking == ""
        cmd = seen["cmd"]
        assert cmd[:2] == ["codex", "exec"]
        assert ["-s", "read-only"] == cmd[cmd.index("-s"):cmd.index("-s") + 2]
        assert "--ephemeral" in cmd
        assert seen["kwargs"]["cwd"] == str(_SANDBOX)


class TestFactory:

    def test_build_backend_resolves_cli_backends(self):
        claude = build_backend(ModelConfig(
            model_type="llm", backend="claude_cli", model_name="claude-opus-4-6",
        ))
        codex = build_backend(ModelConfig(
            model_type="llm", backend="codex_cli", model_name="gpt-5.4",
        ))
        assert claude.name == "claude_cli"
        assert codex.name == "codex_cli"
        # subprocess concurrency is deliberately capped
        assert claude.max_safe_workers == 8


class TestCliTimeoutEnvKnob:
    """CLI_TIMEOUT_S is the per-node timeout knob (audit B7)."""

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("CLI_TIMEOUT_S", "600")
        backend = ClaudeCliBackend(model_name="claude-opus-4-6")
        assert backend.timeout == 600

    def test_default_without_env(self, monkeypatch):
        monkeypatch.delenv("CLI_TIMEOUT_S", raising=False)
        backend = ClaudeCliBackend(model_name="claude-opus-4-6")
        from stagekit.models.backends.cli_backend import DEFAULT_CLI_TIMEOUT_S
        assert backend.timeout == DEFAULT_CLI_TIMEOUT_S
