"""Runbook agent — Stage 4: produce a verified RUN.md."""

from __future__ import annotations

from typing import Any

from demoforge.agents.base import BaseAgent
from demoforge.llm.agent_loop import AgentResult


class RunbookAgent(BaseAgent):
    """Stage 4: write and verify RUN.md."""

    agent_name = "runbook"
    prompt_template = "runbook"

    def build_initial_message(
        self,
        *,
        blueprint: str,
        review_report: str,
        file_tree: str,
        **kwargs: Any,
    ) -> str:
        return (
            "Please write RUN.md with full setup and demo instructions. "
            "Verify each command with run_command before including it."
        )

    async def run(  # type: ignore[override]
        self,
        *,
        blueprint: str,
        review_report: str,
        file_tree: str,
        **kwargs: Any,
    ) -> AgentResult:
        return await super().run(
            blueprint=blueprint,
            review_report=review_report,
            file_tree=file_tree,
        )
