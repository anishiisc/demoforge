"""Architect agent — Stage 1: blueprint, production spec, and manifest."""

from __future__ import annotations

import json
from typing import Any

from demoforge.agents.base import BaseAgent
from demoforge.llm.agent_loop import AgentResult


class ArchitectAgent(BaseAgent):
    """Stage 1: produce demo_blueprint.md, production_spec.md, and manifest.json."""

    agent_name = "architect"
    prompt_template = "architect"

    def build_initial_message(self, *, plan: dict[str, Any], **kwargs: Any) -> str:
        return (
            "Please produce the demo blueprint, production spec, and file manifest "
            "for the following plan. Start with demo_blueprint.md.\n\n"
            f"Plan title: {plan.get('title', '(untitled)')}"
        )

    async def run(self, *, plan: dict[str, Any], **kwargs: Any) -> AgentResult:  # type: ignore[override]
        """Run the Architect with the validated plan from Stage 0."""
        return await super().run(
            plan=plan,
            plan_json=json.dumps(plan, indent=2),
            **kwargs,
        )
