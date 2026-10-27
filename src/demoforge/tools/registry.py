"""Tool registry: definitions, permissions, and dispatch."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from demoforge.workspace import Workspace

# Type alias for async tool handlers
Handler = Callable[..., Awaitable[str]]


@dataclass
class ToolDef:
    """Declaration of one agent tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Handler
    allowed_agents: list[str]  # agent names that may use this tool

    def to_api_dict(self) -> dict[str, Any]:
        """Return the tool definition dict expected by the Messages API."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """Registry of all tools; builds per-agent allowlists and dispatches calls."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        """Add a tool definition."""
        self._tools[tool.name] = tool

    def tools_for(self, agent: str) -> list[dict[str, Any]]:
        """Return the API tool-definition list for *agent*."""
        return [t.to_api_dict() for t in self._tools.values() if agent in t.allowed_agents]

    async def dispatch(
        self,
        name: str,
        inputs: dict[str, Any],
        ctx: "ToolContext",
    ) -> str:
        """Invoke *name* with *inputs* in *ctx*; return result string."""
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name!r}")
        tool = self._tools[name]
        return await tool.handler(ctx=ctx, **inputs)


@dataclass
class ToolContext:
    """Execution context passed to every tool handler."""

    workspace: Workspace
    agent: str
    stage: str
    run_id: str
    # Injected by the orchestrator for ask_user support
    ask_user_callback: Callable[[str], Awaitable[str]] | None = None
    # Injected for submit_plan / submit_manifest validation results
    structured_output: dict[str, Any] = field(default_factory=dict)
    config: Any = None  # Config object


def build_registry(workspace: Workspace, config: Any) -> ToolRegistry:
    """Create and return a fully populated ToolRegistry."""
    from demoforge.tools.fs_tools import (
        handle_read_file,
        handle_write_file,
        handle_list_files,
        handle_edit_file,
    )
    from demoforge.tools.exec_tools import handle_run_command
    from demoforge.tools.control_tools import (
        handle_submit_plan,
        handle_submit_manifest,
        handle_finish_stage,
        handle_ask_user,
    )

    reg = ToolRegistry()

    _ALL_EXCEPT_PLANNER = [
        "architect", "builder", "reviewer", "runbook", "simulator", "documentation"
    ]
    _WRITE_AGENTS = ["architect", "builder", "reviewer", "runbook", "simulator", "documentation"]
    _EDIT_AGENTS = ["builder", "reviewer"]
    _CMD_AGENTS = ["builder", "reviewer", "runbook"]
    _ALL = _ALL_EXCEPT_PLANNER + ["planner"]

    reg.register(ToolDef(
        name="read_file",
        description="Read the contents of a file inside the run workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path within the workspace"}
            },
            "required": ["path"],
        },
        handler=handle_read_file,
        allowed_agents=_ALL_EXCEPT_PLANNER,
    ))

    reg.register(ToolDef(
        name="list_files",
        description="List files and directories inside the run workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "subdir": {
                    "type": "string",
                    "description": "Subdirectory to list (relative, default '.')",
                    "default": ".",
                }
            },
        },
        handler=handle_list_files,
        allowed_agents=_ALL_EXCEPT_PLANNER,
    ))

    reg.register(ToolDef(
        name="write_file",
        description=(
            "Create or overwrite a file inside the run workspace. "
            "The path must be relative to the demo directory."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to write"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
        handler=handle_write_file,
        allowed_agents=_WRITE_AGENTS,
    ))

    reg.register(ToolDef(
        name="edit_file",
        description="Replace an exact string in a file inside the run workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string", "description": "Exact string to find"},
                "new": {"type": "string", "description": "Replacement string"},
            },
            "required": ["path", "old", "new"],
        },
        handler=handle_edit_file,
        allowed_agents=_EDIT_AGENTS,
    ))

    reg.register(ToolDef(
        name="run_command",
        description=(
            "Run an allowlisted shell command inside the demo project directory. "
            "Only commands in the allowed list may be used."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command as a list of tokens, e.g. ['python', 'script.py']",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory relative to demo root (default '.')",
                    "default": ".",
                },
            },
            "required": ["cmd"],
        },
        handler=handle_run_command,
        allowed_agents=_CMD_AGENTS,
    ))

    reg.register(ToolDef(
        name="submit_plan",
        description=(
            "Submit the structured plan produced by Stage 0. "
            "The plan must conform to plan.schema.json."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "plan": {"type": "object", "description": "The plan object"}
            },
            "required": ["plan"],
        },
        handler=handle_submit_plan,
        allowed_agents=["planner"],
    ))

    reg.register(ToolDef(
        name="submit_manifest",
        description=(
            "Submit the structured file manifest produced by Stage 1. "
            "The manifest must conform to manifest.schema.json."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "manifest": {"type": "object", "description": "The manifest object"}
            },
            "required": ["manifest"],
        },
        handler=handle_submit_manifest,
        allowed_agents=["architect"],
    ))

    reg.register(ToolDef(
        name="ask_user",
        description=(
            "Pause the run to ask the user a clarifying question. "
            "Use sparingly — at most 3 times per run."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to ask"}
            },
            "required": ["question"],
        },
        handler=handle_ask_user,
        allowed_agents=["planner"],
    ))

    reg.register(ToolDef(
        name="finish_stage",
        description=(
            "Signal that this stage is complete. "
            "Provide a summary of at most 120 words describing what was produced."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Stage completion summary (max 120 words)"}
            },
            "required": ["summary"],
        },
        handler=handle_finish_stage,
        allowed_agents=_ALL,
    ))

    return reg
