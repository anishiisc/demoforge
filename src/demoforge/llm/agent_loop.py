"""Generic tool-use agent loop for DemoForge agents."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import anthropic

from demoforge.tools.control_tools import FINISH_STAGE_SENTINEL

if TYPE_CHECKING:
    from demoforge.llm.client import LLMClient
    from demoforge.llm.dry_run import DryRunClient
    from demoforge.tools.registry import ToolContext, ToolRegistry
    from demoforge.audit.logger import AuditLogger


@dataclass
class AgentResult:
    """Result returned after a completed agent loop."""

    final_text: str
    tool_calls: list[dict[str, Any]]
    turns: int
    input_tokens: int
    output_tokens: int
    stage_summary: str = ""


class AgentStalled(Exception):
    """Raised when the agent exceeds max_turns without completing."""

    def __init__(self, agent: str, turns: int) -> None:
        super().__init__(f"Agent '{agent}' stalled after {turns} turns without calling finish_stage")
        self.agent = agent
        self.turns = turns


async def run_agent_loop(
    *,
    agent: str,
    stage: str,
    model: str,
    system: list[dict[str, Any]] | str,
    initial_message: str,
    client: "LLMClient | DryRunClient",
    registry: "ToolRegistry",
    ctx: "ToolContext",
    audit: "AuditLogger",
    max_turns: int = 40,
    max_tokens: int = 16000,
    tool_choice: dict[str, Any] | None = None,
    context_compaction_chars: int = 200_000,
    compaction_keep_turns: int = 6,
    compaction_model: str | None = None,
) -> AgentResult:
    """Run a tool-use agent loop until finish_stage is called or max_turns is reached.

    Parameters
    ----------
    agent:
        Agent name (used for tool filtering and logging).
    stage:
        Pipeline stage label.
    model:
        Anthropic model ID to use.
    system:
        System prompt string or list of content blocks.
    initial_message:
        User message that starts the conversation.
    client:
        LLMClient or DryRunClient instance for this stage.
    registry:
        ToolRegistry to look up and dispatch tools.
    ctx:
        ToolContext providing workspace and config.
    audit:
        AuditLogger for this run.
    max_turns:
        Maximum number of message round-trips before raising AgentStalled.
    max_tokens:
        Max tokens per response.
    tool_choice:
        Optional forced tool_choice dict (e.g. for submit_plan).
    context_compaction_chars:
        Total character length of messages before compaction triggers.
    compaction_keep_turns:
        Number of recent turns (user+assistant pairs) to retain after compaction.
    compaction_model:
        Model to use for compaction summary; falls back to haiku.
    """
    tools = registry.tools_for(agent)
    messages: list[dict[str, Any]] = [{"role": "user", "content": initial_message}]
    all_tool_calls: list[dict[str, Any]] = []
    total_input = 0
    total_output = 0

    for turn_idx in range(max_turns):
        # Compact context if it's grown too large
        if _messages_chars(messages) > context_compaction_chars:
            messages = await _compact(
                messages=messages,
                keep_turns=compaction_keep_turns,
                client=client,
                agent=agent,
                stage=stage,
                model=compaction_model or model,
                system=system,
                audit=audit,
            )

        resp = await client.create(
            agent=agent,
            model=model,
            system=system,
            messages=messages,
            tools=tools or None,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
        )

        total_input += getattr(resp.usage, "input_tokens", 0)
        total_output += getattr(resp.usage, "output_tokens", 0)

        # Append assistant response
        assistant_content = _content_to_dicts(resp.content)
        messages.append({"role": "assistant", "content": assistant_content})

        stop_reason = resp.stop_reason

        if stop_reason == "end_turn":
            break

        # Always check for tool_use blocks first — the model can include fully-formed
        # tool_use blocks even when stop_reason is "max_tokens" (e.g. it hit the limit
        # after generating the tool call but before finishing a subsequent text block).
        tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]

        if tool_use_blocks:
            tool_result_blocks: list[dict[str, Any]] = []
            finished = False

            for block in tool_use_blocks:
                tool_name = block.name
                tool_input = block.input if isinstance(block.input, dict) else {}
                t0 = time.monotonic()
                is_error = False
                output = ""

                try:
                    output = await registry.dispatch(tool_name, tool_input, ctx)
                except Exception as exc:
                    is_error = True
                    output = f"Tool error: {exc}"

                duration_ms = (time.monotonic() - t0) * 1000

                audit.tool_call(
                    stage=stage,
                    agent=agent,
                    tool=tool_name,
                    input_data=tool_input,
                    is_error=is_error,
                    output=output,
                    duration_ms=duration_ms,
                )

                all_tool_calls.append({"tool": tool_name, "input": tool_input, "output": output})

                if output == FINISH_STAGE_SENTINEL:
                    finished = True
                    output = "Stage marked complete."

                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                })

            messages.append({"role": "user", "content": tool_result_blocks})

            if finished:
                break

        elif stop_reason == "max_tokens":
            # No tool calls and response was truncated — ask the model to continue.
            messages.append({"role": "user", "content": "continue"})

        # Clear tool_choice after first turn (it's only forced on turn 0)
        tool_choice = None

    else:
        raise AgentStalled(agent, max_turns)

    # Extract final text from last assistant message
    final_text = _extract_text(messages)
    stage_summary = ctx.structured_output.get("stage_summary", "")

    return AgentResult(
        final_text=final_text,
        tool_calls=all_tool_calls,
        turns=turn_idx + 1,
        input_tokens=total_input,
        output_tokens=total_output,
        stage_summary=stage_summary,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content_to_dicts(content: list[Any]) -> list[dict[str, Any]]:
    """Convert Anthropic SDK content block objects to plain dicts.

    Only includes the fields the API accepts when the dict is sent back as
    message history.  In particular, SDK v2 ToolUseBlock.model_dump() includes
    extra vendor fields (``caller``, ``toolset_name``) that the API rejects, so
    we map each known block type explicitly instead of calling model_dump().
    """
    result = []
    for block in content:
        if isinstance(block, dict):
            result.append(block)
            continue
        btype = getattr(block, "type", None)
        if btype == "text":
            result.append({"type": "text", "text": block.text})
        elif btype == "tool_use":
            result.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
        elif btype == "thinking":
            # thinking field is required by the API even when empty; always include it.
            d: dict[str, Any] = {
                "type": "thinking",
                "thinking": getattr(block, "thinking", ""),
            }
            sig = getattr(block, "signature", None)
            if sig is not None:
                d["signature"] = sig
            result.append(d)
        elif btype == "redacted_thinking":
            d = {"type": "redacted_thinking"}
            data = getattr(block, "data", None)
            if data is not None:
                d["data"] = data
            result.append(d)
        elif btype is not None:
            # Unknown future block type: include type plus any non-null scalar attrs
            d = {"type": btype}
            for attr in ("id", "name", "input", "text", "thinking", "signature", "data"):
                val = getattr(block, attr, None)
                if val is not None:
                    d[attr] = val
            result.append(d)
    return result


def _extract_text(messages: list[dict[str, Any]]) -> str:
    """Return the last assistant text content from messages."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", [])
            texts = []
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
            elif isinstance(content, str):
                texts.append(content)
            if texts:
                return "\n".join(texts)
    return ""


def _messages_chars(messages: list[dict[str, Any]]) -> int:
    """Rough total character count of all messages."""
    import json
    return len(json.dumps(messages))


async def _compact(
    *,
    messages: list[dict[str, Any]],
    keep_turns: int,
    client: Any,
    agent: str,
    stage: str,
    model: str,
    system: Any,
    audit: "AuditLogger",
) -> list[dict[str, Any]]:
    """Compact a long conversation, returning a shortened message list.

    Keeps the first user message and the most recent *keep_turns* turns.
    Inserts a summary of dropped content produced by a Haiku call.
    """
    if len(messages) <= keep_turns * 2 + 1:
        return messages

    first = messages[:1]
    # Pair messages into turns (user, assistant)
    rest = messages[1:]
    pairs = []
    i = 0
    while i + 1 < len(rest):
        pairs.append((rest[i], rest[i + 1]))
        i += 2
    if i < len(rest):
        pairs.append((rest[i],))

    drop_pairs = pairs[:-keep_turns]
    keep_pairs = pairs[-keep_turns:]
    dropped_turns = len(drop_pairs)

    # Summarise dropped content
    dropped_text = " ".join(
        str(p) for pair in drop_pairs for p in pair
    )[:4000]
    summary_prompt = (
        f"Summarise the following conversation history in 2-3 sentences for context:\n\n{dropped_text}"
    )
    try:
        summary_resp = await client.create(
            agent=agent,
            model=model,
            system="You are a concise summariser.",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=300,
        )
        summary_text = _extract_text([{"role": "assistant", "content": _content_to_dicts(summary_resp.content)}])
        summary_tokens = getattr(summary_resp.usage, "output_tokens", 0)
    except Exception:
        summary_text = "[compaction summary unavailable]"
        summary_tokens = 0

    audit.compaction(agent=agent, dropped_turns=dropped_turns, summary_tokens=summary_tokens)

    summary_msg = {
        "role": "user",
        "content": f"[Context compacted — {dropped_turns} turns omitted. Summary: {summary_text}]",
    }
    compacted = first + [summary_msg]
    for pair in keep_pairs:
        compacted.extend(pair)
    return compacted
