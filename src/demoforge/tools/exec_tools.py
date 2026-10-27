"""Sandboxed command execution tool."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from demoforge.workspace import WorkspaceError

if TYPE_CHECKING:
    from demoforge.tools.registry import ToolContext

_OUTPUT_TRUNCATE = 8000  # chars returned to the model
_FULL_OUTPUT_TRUNCATE = 100_000  # chars stored in audit (via logger)


async def handle_run_command(
    *, ctx: "ToolContext", cmd: list[str], cwd: str = "."
) -> str:
    """Run *cmd* inside the workspace demo directory (sandboxed).

    Validates the first token against the allowlist, resolves *cwd* inside the
    workspace, and captures stdout+stderr. Returns truncated output.
    """
    if not cmd:
        return "ERROR: cmd must be a non-empty list"

    cfg = ctx.config
    allowed = cfg.execution.allowed_commands if cfg else ["python", "pip", "pytest", "ruff", "ls", "cat"]
    timeout = cfg.limits.command_timeout_seconds if cfg else 300

    executable = cmd[0]
    if executable not in allowed:
        return (
            f"ERROR: Command '{executable}' is not in the allowed list: {allowed}. "
            "No command was run."
        )

    # Resolve working directory
    try:
        work_dir = ctx.workspace.guard_path(cwd, allow_run_root=True)
    except WorkspaceError as exc:
        return f"ERROR: {exc}"
    if not work_dir.is_dir():
        work_dir = ctx.workspace.demo_dir

    # Build a clean environment — never expose the API key
    env = _scrubbed_env()

    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(work_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ),
            timeout=timeout,
        )
        stdout_bytes, _ = await proc.communicate()
        returncode = proc.returncode
    except asyncio.TimeoutError:
        return f"ERROR: Command timed out after {timeout}s"
    except Exception as exc:
        return f"ERROR: Failed to run command: {exc}"

    output = stdout_bytes.decode("utf-8", errors="replace")
    full_output = output[:_FULL_OUTPUT_TRUNCATE]
    truncated_output = output[:_OUTPUT_TRUNCATE]
    if len(output) > _OUTPUT_TRUNCATE:
        truncated_output += f"\n... [output truncated at {_OUTPUT_TRUNCATE} chars]"

    status = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
    return f"{status}\n{truncated_output}"


def _scrubbed_env() -> dict[str, str]:
    """Return os.environ minus sensitive variables."""
    _STRIP = {
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    }
    return {k: v for k, v in os.environ.items() if k not in _STRIP}
