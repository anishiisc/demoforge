"""Tests for the generic agent loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from demoforge.config import Config
from demoforge.audit.logger import AuditLogger
from demoforge.llm.agent_loop import AgentResult, AgentStalled, run_agent_loop
from demoforge.tools.registry import ToolContext, ToolRegistry, ToolDef, build_registry
from demoforge.workspace import Workspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(content: list[dict[str, Any]], stop_reason: str = "end_turn") -> MagicMock:
    """Build a mock Message object."""
    msg = MagicMock()
    msg.stop_reason = stop_reason
    msg.content = [_make_block(b) for b in content]
    msg.usage = MagicMock(input_tokens=10, output_tokens=5)
    return msg


def _make_block(data: dict[str, Any]) -> MagicMock:
    block = MagicMock()
    block.type = data["type"]
    if data["type"] == "text":
        block.text = data.get("text", "")
    elif data["type"] == "tool_use":
        block.id = data.get("id", "toolu_1")
        block.name = data["name"]
        block.input = data.get("input", {})
    return block


def _finish_stage_msg() -> MagicMock:
    return _make_message(
        [{"type": "tool_use", "id": "toolu_f", "name": "finish_stage", "input": {"summary": "done"}}],
        stop_reason="tool_use",
    )


@pytest.fixture()
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace("20240101-000000-test", runs_root=tmp_path)
    ws.create()
    return ws


@pytest.fixture()
def audit(tmp_path: Path) -> AuditLogger:
    return AuditLogger(
        log_path=tmp_path / "log.jsonl",
        payloads_dir=tmp_path / "payloads",
        run_id="test",
    )


@pytest.fixture()
def config() -> Config:
    return Config()


@pytest.fixture()
def registry(workspace: Workspace, config: Config) -> ToolRegistry:
    return build_registry(workspace, config)


@pytest.fixture()
def ctx(workspace: Workspace, config: Config) -> ToolContext:
    return ToolContext(
        workspace=workspace,
        agent="builder",
        stage="builder",
        run_id="test",
        config=config,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_turn_stops_loop(registry: ToolRegistry, ctx: ToolContext, audit: AuditLogger):
    """Loop exits cleanly on end_turn without tool calls."""
    client = AsyncMock()
    client.create.return_value = _make_message(
        [{"type": "text", "text": "All done."}], stop_reason="end_turn"
    )
    result = await run_agent_loop(
        agent="builder",
        stage="builder",
        model="m",
        system="sys",
        initial_message="hello",
        client=client,
        registry=registry,
        ctx=ctx,
        audit=audit,
        max_turns=5,
    )
    assert result.turns == 1
    assert result.final_text == "All done."


@pytest.mark.asyncio
async def test_finish_stage_stops_loop(registry: ToolRegistry, ctx: ToolContext, audit: AuditLogger):
    """Loop exits when finish_stage tool is called."""
    client = AsyncMock()
    client.create.return_value = _finish_stage_msg()

    result = await run_agent_loop(
        agent="builder",
        stage="builder",
        model="m",
        system="sys",
        initial_message="build",
        client=client,
        registry=registry,
        ctx=ctx,
        audit=audit,
        max_turns=5,
    )
    assert result.stage_summary == "done"


@pytest.mark.asyncio
async def test_tool_error_does_not_crash_loop(
    registry: ToolRegistry, ctx: ToolContext, audit: AuditLogger
):
    """A tool that raises returns is_error=True; loop continues."""
    # First call: call a bad tool; second call: finish_stage
    bad_tool_msg = _make_message(
        [{"type": "tool_use", "id": "toolu_bad", "name": "read_file", "input": {"path": "../../etc/passwd"}}],
        stop_reason="tool_use",
    )
    client = AsyncMock()
    client.create.side_effect = [bad_tool_msg, _finish_stage_msg()]

    result = await run_agent_loop(
        agent="builder",
        stage="builder",
        model="m",
        system="sys",
        initial_message="do bad thing",
        client=client,
        registry=registry,
        ctx=ctx,
        audit=audit,
        max_turns=5,
    )
    # Should have completed (finish_stage on second turn)
    assert result.turns == 2


@pytest.mark.asyncio
async def test_max_turn_stall_raises(
    registry: ToolRegistry, ctx: ToolContext, audit: AuditLogger
):
    """Exceeding max_turns raises AgentStalled."""
    # Always return a tool_use that doesn't finish
    read_msg = _make_message(
        [{"type": "tool_use", "id": "toolu_1", "name": "list_files", "input": {}}],
        stop_reason="tool_use",
    )
    client = AsyncMock()
    client.create.return_value = read_msg

    with pytest.raises(AgentStalled):
        await run_agent_loop(
            agent="builder",
            stage="builder",
            model="m",
            system="sys",
            initial_message="loop",
            client=client,
            registry=registry,
            ctx=ctx,
            audit=audit,
            max_turns=3,
        )


@pytest.mark.asyncio
async def test_multi_tool_turn(registry: ToolRegistry, ctx: ToolContext, audit: AuditLogger, tmp_path: Path):
    """Multiple tool calls in one turn all get results appended in one user message."""
    # Write two files first so read_file doesn't fail
    (ctx.workspace.demo_dir / "a.txt").write_text("hello")
    (ctx.workspace.demo_dir / "b.txt").write_text("world")

    multi_tool_msg = _make_message(
        [
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.txt"}},
            {"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "b.txt"}},
        ],
        stop_reason="tool_use",
    )
    client = AsyncMock()
    client.create.side_effect = [multi_tool_msg, _finish_stage_msg()]

    result = await run_agent_loop(
        agent="builder",
        stage="builder",
        model="m",
        system="sys",
        initial_message="read both",
        client=client,
        registry=registry,
        ctx=ctx,
        audit=audit,
        max_turns=5,
    )
    # The loop dispatched 2 read_file calls + 1 finish_stage = 3 tool calls total
    assert len(result.tool_calls) == 3
    assert result.tool_calls[0]["tool"] == "read_file"
    assert result.tool_calls[1]["tool"] == "read_file"
    assert result.tool_calls[2]["tool"] == "finish_stage"

    # The second client.create call receives messages that include the 2 tool_results.
    # Because the messages list is shared/mutated, inspect the final list via
    # the last-but-two entry (before finish_stage assistant and its result were appended).
    # Final messages: [initial, asst(2_tools), user(2_results), asst(finish), user(finish_result)]
    final_messages = client.create.call_args_list[1][1]["messages"]
    # Find the user message that contains tool_results for t1 and t2
    two_result_msg = final_messages[-3]  # third from last in the final list
    assert two_result_msg["role"] == "user"
    content = two_result_msg["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    assert all(b["type"] == "tool_result" for b in content)
