"""CLI subprocess backends: drive `claude -p` and `codex exec` as LLM backends.

Same adapter contract as the API backends — prompt in, text out, raise
on failure (retry/error classification live centrally in stagekit/models/base.py,
caching above that). The invocation mirrors the battle-tested flags
from the fallacies-shared-task orchestrator: sandboxed (no MCP, no
tools, no session persistence), prompt on stdin, and for claude the
``--output-format json`` wrapper so errors are structural, not
heuristic.

Subprocesses run in an empty sandbox directory so config auto-discovery
(CLAUDE.md, codex config walking up from cwd) finds nothing — the CLI
must behave like a bare model, not like an agent in this repo.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from stagekit.models.backends.backends_base import Answer, LLMBackend
from stagekit.models.backends.factory import register_backend

logger = logging.getLogger(__name__)

#: Default seconds before a single CLI call is killed (long prompts +
#: reasoning); override per node via the CLI_TIMEOUT_S env var
#: (LLMBackend.timeout_from_env).
DEFAULT_CLI_TIMEOUT_S = 300

_SANDBOX = Path(tempfile.gettempdir()) / "fallacy-detection-cli-sandbox"


class _CliBackend(LLMBackend):
    """Shared subprocess plumbing for CLI-driven models."""

    # CLI subprocesses are heavyweight (node startup per call) — keep
    # concurrency far below the API backends.
    max_safe_workers = 8

    def __init__(self, model_name: str, temperature: float = 1):
        # Base init warns about the ignored temperature (derived from
        # BACKEND_CAPABILITIES — the CLI backends have no identity fields).
        super().__init__(model_name, temperature)
        self.timeout = self.timeout_from_env(
            "CLI_TIMEOUT_S", DEFAULT_CLI_TIMEOUT_S,
        )
        _SANDBOX.mkdir(parents=True, exist_ok=True)

    def _command(self) -> list[str]:
        raise NotImplementedError

    def _extract(self, stdout: str) -> str:
        """Turn raw stdout into the model's text. Raises on error payloads."""
        raise NotImplementedError

    def generate(self, prompt: str) -> Answer:
        """Run the CLI once and return its text.

        Returns:
            The answer with an empty ``thinking`` — the CLIs expose no
            separate reasoning channel.

        Raises:
            RuntimeError: On timeout, launch failure, or non-zero exit.
        """
        cmd = self._command()
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(_SANDBOX),
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"{self.name}: subprocess timed out after {self.timeout}s"
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                f"{self.name}: failed to launch '{cmd[0]}' — is the CLI "
                f"installed and on PATH? ({exc})"
            )

        # Both CLIs write errors to stdout with a non-zero exit code.
        if result.returncode != 0:
            raise RuntimeError(
                f"{self.name}: exit code {result.returncode}; "
                f"stdout: {result.stdout[:300]!r}; stderr: {result.stderr[:300]!r}"
            )
        return Answer(raw=self._extract(result.stdout))


@register_backend("claude_cli")
class ClaudeCliBackend(_CliBackend):
    """`claude -p` in print mode with the structured JSON wrapper."""

    name = "claude_cli"

    def _command(self) -> list[str]:
        return [
            "claude",
            "-p",                          # non-interactive, prompt on stdin
            "--model", self.model_name,
            "--strict-mcp-config",         # only MCP servers from --mcp-config
            "--mcp-config", '{"mcpServers":{}}',  # empty -> none load
            "--tools", "",                 # no built-in tools
            "--system-prompt", "",         # no default system prompt
            "--no-session-persistence",
            "--output-format", "json",     # wrapper: is_error/subtype/result
        ]

    def _extract(self, stdout: str) -> str:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"{self.name}: unparsable JSON wrapper: {stdout[:300]!r}"
            )
        if payload.get("is_error") or payload.get("subtype") != "success":
            raise RuntimeError(
                f"{self.name}: error payload (subtype="
                f"{payload.get('subtype')!r}, error={str(payload.get('error'))[:200]!r})"
            )
        result_text = payload.get("result")
        if not isinstance(result_text, str):
            raise RuntimeError(f"{self.name}: payload missing string .result")
        return result_text.strip()


@register_backend("codex_cli")
class CodexCliBackend(_CliBackend):
    """`codex exec` read-only; raw text on stdout."""

    name = "codex_cli"

    def _command(self) -> list[str]:
        return [
            "codex", "exec",
            "-m", self.model_name,
            "-s", "read-only",             # sandbox: read files, no writes
            "--ephemeral",                 # no persisted session files
        ]

    def _extract(self, stdout: str) -> str:
        return stdout.strip()
