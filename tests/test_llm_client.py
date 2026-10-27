"""Unit tests for the LLM cost tracker and dry-run client."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from demoforge.config import Config, LimitsConfig
from demoforge.llm.cost import BudgetExceeded, CostTracker, UsageRecord, _compute_cost
from demoforge.llm.dry_run import DryRunClient, _end_turn_fixture, _dict_to_message
from demoforge.audit.logger import AuditLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_workspace(tmp_path: Path):
    return tmp_path


@pytest.fixture()
def config():
    return Config()


@pytest.fixture()
def audit(tmp_path: Path):
    return AuditLogger(
        log_path=tmp_path / "orchestration_log.jsonl",
        payloads_dir=tmp_path / "llm_payloads",
        run_id="test-run",
    )


@pytest.fixture()
def tracker(config):
    return CostTracker(config)


@pytest.fixture()
def dry_client(config, tracker, audit, tmp_path: Path):
    fixtures_dir = tmp_path / "dry_run_responses"
    fixtures_dir.mkdir()
    return DryRunClient(
        config=config,
        cost_tracker=tracker,
        audit=audit,
        stage="planner",
        fixtures_dir=fixtures_dir,
    )


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------


def test_cost_zero_pricing(config, tracker):
    """With zero pricing, all calls cost zero."""
    usage = UsageRecord(input_tokens=1000, output_tokens=500)
    cost = _compute_cost("claude-sonnet-5-5", usage, config)
    assert cost == 0.0


def test_cost_accumulates(config, tracker):
    usage = UsageRecord(input_tokens=0, output_tokens=0)
    tracker.record("claude-sonnet-5-5", usage)
    tracker.record("claude-sonnet-5-5", usage)
    assert tracker.total_usd == 0.0


def test_budget_exceeded():
    """BudgetExceeded is raised when cost crosses the limit."""
    from demoforge.config import Config, LimitsConfig
    import pydantic

    cfg = Config.model_validate({
        "limits": {"run_budget_usd": 0.001},
        "pricing_usd_per_mtok": {
            "my-model": {"input": 100.0, "output": 100.0},
        },
    })
    tracker = CostTracker(cfg)
    usage = UsageRecord(input_tokens=100_000, output_tokens=100_000)
    with pytest.raises(BudgetExceeded) as exc_info:
        tracker.record("my-model", usage)
    assert exc_info.value.limit == 0.001


def test_tracker_summary(tracker):
    usage = UsageRecord(input_tokens=100, output_tokens=50)
    tracker.record("claude-sonnet-5-5", usage)
    s = tracker.summary
    assert s["total_input_tokens"] == 100
    assert s["total_output_tokens"] == 50


# ---------------------------------------------------------------------------
# Dry-run client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_end_turn_fallback(dry_client):
    """Without fixtures, dry-run client returns a finish_stage tool_use response."""
    msg = await dry_client.create(
        agent="planner",
        model="claude-opus-5-5",
        system="system prompt",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert msg.stop_reason == "tool_use"
    assert msg.content[0].name == "finish_stage"


@pytest.mark.asyncio
async def test_dry_run_loads_fixture(dry_client, tmp_path: Path):
    """Dry-run client loads fixture JSON from the fixtures directory."""
    stage_dir = dry_client._fixtures_dir / "planner"
    stage_dir.mkdir(parents=True)
    fixture = {
        "id": "msg_fixture_01",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "fixture response"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    (stage_dir / "01.json").write_text(json.dumps(fixture))

    msg = await dry_client.create(
        agent="planner",
        model="claude-opus-5-5",
        system="system",
        messages=[{"role": "user", "content": "test"}],
    )
    assert msg.stop_reason == "end_turn"
    assert msg.content[0].text == "fixture response"


@pytest.mark.asyncio
async def test_dry_run_for_stage(dry_client):
    """for_stage returns a new client with the given stage, turn reset to 0."""
    cloned = dry_client.for_stage("builder")
    assert cloned._stage == "builder"
    assert cloned._turn == 0


def test_dict_to_message_text():
    data = {
        "id": "m1",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    msg = _dict_to_message(data)
    assert msg.stop_reason == "end_turn"
    assert msg.content[0].text == "hello"


def test_dict_to_message_tool_use():
    data = {
        "id": "m2",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "write_file",
                "input": {"path": "foo.py", "content": "print('hi')"},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 5, "output_tokens": 10},
    }
    msg = _dict_to_message(data)
    assert msg.stop_reason == "tool_use"
    assert msg.content[0].name == "write_file"
    assert msg.content[0].input["path"] == "foo.py"
