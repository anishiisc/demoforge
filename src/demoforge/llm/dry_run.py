"""Dry-run LLM client that returns canned fixture responses instead of calling Anthropic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic

from demoforge.audit.logger import AuditLogger
from demoforge.config import Config
from demoforge.llm.cost import CostTracker, UsageRecord

_FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "dry_run_responses"


class DryRunClient:
    """Drop-in replacement for :class:`LLMClient` that serves fixtures.

    Fixtures live at ``tests/fixtures/dry_run_responses/<stage>/<turn>.json``.
    Each file must be a JSON object that looks like an Anthropic Message response.
    Tool calls in the fixtures are executed for real against the workspace.
    """

    def __init__(
        self,
        config: Config,
        cost_tracker: CostTracker,
        audit: AuditLogger,
        stage: str,
        fixtures_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._cost_tracker = cost_tracker
        self._audit = audit
        self._stage = stage
        self._fixtures_dir = fixtures_dir or _FIXTURES_DIR
        self._turn: int = 0

    async def create(
        self,
        *,
        agent: str,
        model: str,
        system: list[dict[str, Any]] | str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> anthropic.types.Message:
        """Return the next fixture response for the current stage and turn."""
        self._turn += 1
        fixture = self._load_fixture(agent, self._turn)
        msg = _dict_to_message(fixture)

        # Log as if it were a real call
        self._audit.llm_call(
            stage=self._stage,
            agent=agent,
            model=model,
            turn=self._turn,
            request_message_count=len(messages),
            stop_reason=fixture.get("stop_reason", "end_turn"),
            input_tokens=fixture.get("usage", {}).get("input_tokens", 0),
            output_tokens=fixture.get("usage", {}).get("output_tokens", 0),
            cache_read_tokens=0,
            cache_creation_tokens=0,
            latency_ms=0.0,
            cost_usd=0.0,
            request_id=fixture.get("id", f"dry-run-{self._stage}-{self._turn}"),
        )
        return msg

    def _load_fixture(self, agent: str, turn: int) -> dict[str, Any]:
        """Load fixture JSON for this stage/turn, falling back to a generic end_turn."""
        stage_dir = self._fixtures_dir / self._stage
        candidates = [
            stage_dir / f"{turn:02d}_{agent}.json",
            stage_dir / f"{turn:02d}.json",
            stage_dir / "default.json",
        ]
        for path in candidates:
            if path.exists():
                return json.loads(path.read_text())
        # Synthetic end_turn response when no fixture exists
        return _end_turn_fixture(agent, turn)

    def for_stage(self, stage: str) -> "DryRunClient":
        """Return a shallow copy of this client with a different stage label."""
        clone = object.__new__(DryRunClient)
        clone._config = self._config
        clone._cost_tracker = self._cost_tracker
        clone._audit = self._audit
        clone._stage = stage
        clone._fixtures_dir = self._fixtures_dir
        clone._turn = 0
        return clone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _end_turn_fixture(agent: str, turn: int) -> dict[str, Any]:
    """Return a minimal end_turn fixture."""
    return {
        "id": f"dry-run-{agent}-{turn}",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": f"toolu_dry_{turn}",
                "name": "finish_stage",
                "input": {"summary": f"Dry-run stage complete (turn {turn})."},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _dict_to_message(data: dict[str, Any]) -> anthropic.types.Message:
    """Convert a fixture dict to an :class:`anthropic.types.Message`-like object."""
    content_blocks = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            content_blocks.append(anthropic.types.TextBlock(type="text", text=block.get("text", "")))
        elif block.get("type") == "tool_use":
            content_blocks.append(
                anthropic.types.ToolUseBlock(
                    type="tool_use",
                    id=block.get("id", "toolu_dry"),
                    name=block.get("name", ""),
                    input=block.get("input", {}),
                )
            )

    usage_data = data.get("usage", {})
    usage = anthropic.types.Usage(
        input_tokens=usage_data.get("input_tokens", 0),
        output_tokens=usage_data.get("output_tokens", 0),
    )

    return anthropic.types.Message(
        id=data.get("id", "dry-run"),
        type="message",
        role="assistant",
        content=content_blocks,
        model=data.get("model", "dry-run"),
        stop_reason=data.get("stop_reason", "end_turn"),
        stop_sequence=data.get("stop_sequence"),
        usage=usage,
    )
