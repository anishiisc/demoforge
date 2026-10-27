"""Documentation agent — Stage 5: tech_note.md and MANIFEST.md."""

from __future__ import annotations

from typing import Any

from demoforge.agents.base import BaseAgent
from demoforge.llm.agent_loop import AgentResult


class DocumentationAgent(BaseAgent):
    """Stage 5: write tech_note.md and MANIFEST.md."""

    agent_name = "documentation"
    prompt_template = "documentation"

    def build_initial_message(
        self,
        *,
        artefacts_summary: str,
        log_summary: str,
        **kwargs: Any,
    ) -> str:
        return (
            "Please write tech_note.md and MANIFEST.md as the final documentation "
            "for this demo project. Use a formal register with no em dashes."
        )

    async def run(  # type: ignore[override]
        self,
        *,
        artefacts_summary: str,
        log_summary: str,
        **kwargs: Any,
    ) -> AgentResult:
        return await super().run(
            artefacts_summary=artefacts_summary,
            log_summary=log_summary,
        )
