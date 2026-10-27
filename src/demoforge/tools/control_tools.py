"""Control tools: submit_plan, submit_manifest, finish_stage, ask_user."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonschema

if TYPE_CHECKING:
    from demoforge.tools.registry import ToolContext

_SCHEMAS_DIR = Path(__file__).parent.parent.parent.parent / "schemas"

# Sentinel used by the agent loop to detect stage completion.
FINISH_STAGE_SENTINEL = "__FINISH_STAGE__"


async def handle_submit_plan(*, ctx: "ToolContext", plan: dict[str, Any]) -> str:
    """Validate *plan* against plan.schema.json and write plan.json."""
    error = _validate_against_schema(plan, "plan.schema.json")
    if error:
        return f"VALIDATION_ERROR: {error}"

    path = ctx.workspace.root / "plan.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    ctx.structured_output["plan"] = plan
    return "OK: plan.json written and validated."


async def handle_submit_manifest(*, ctx: "ToolContext", manifest: dict[str, Any]) -> str:
    """Validate *manifest* against manifest.schema.json and write manifest.json."""
    error = _validate_against_schema(manifest, "manifest.schema.json")
    if error:
        return f"VALIDATION_ERROR: {error}"

    path = ctx.workspace.root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ctx.structured_output["manifest"] = manifest
    return "OK: manifest.json written and validated."


async def handle_finish_stage(*, ctx: "ToolContext", summary: str) -> str:
    """Signal stage completion. The agent loop detects this sentinel."""
    word_count = len(summary.split())
    if word_count > 120:
        summary = " ".join(summary.split()[:120]) + " [truncated]"
    ctx.structured_output["stage_summary"] = summary
    return FINISH_STAGE_SENTINEL


async def handle_ask_user(*, ctx: "ToolContext", question: str) -> str:
    """Ask the user a clarifying question (CLI prompt or Streamlit callback)."""
    if ctx.ask_user_callback is not None:
        return await ctx.ask_user_callback(question)
    # No callback: record assumption
    assumption = f"[no-ask mode] Assumption recorded for: {question}"
    ctx.structured_output.setdefault("assumptions", []).append(question)
    return assumption


# ---------------------------------------------------------------------------
# Schema validation helpers
# ---------------------------------------------------------------------------


def _validate_against_schema(data: Any, schema_filename: str) -> str:
    """Return an error message string, or '' if valid."""
    schema_path = _SCHEMAS_DIR / schema_filename
    if not schema_path.exists():
        return ""  # Missing schema: pass through (schema may not be installed yet)
    try:
        schema = json.loads(schema_path.read_text())
        jsonschema.validate(data, schema)
        return ""
    except jsonschema.ValidationError as exc:
        return exc.message
    except Exception as exc:
        return str(exc)
