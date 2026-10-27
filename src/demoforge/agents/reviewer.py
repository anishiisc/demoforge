"""Reviewer agent — Stage 3: static/dynamic review with fix cycles."""

from __future__ import annotations

import json
from typing import Any

from demoforge.agents.base import BaseAgent
from demoforge.llm.agent_loop import AgentResult, run_agent_loop


class ReviewerAgent(BaseAgent):
    """Stage 3: check the demo against the blueprint; fix and re-run up to max_cycles."""

    agent_name = "reviewer"
    prompt_template = "reviewer"

    def build_initial_message(
        self,
        *,
        blueprint: str,
        manifest: dict[str, Any],
        cycle: int = 1,
        **kwargs: Any,
    ) -> str:
        return (
            f"This is review cycle {cycle}. "
            "Please inspect the demo codebase, run the tests, and fix any issues you find. "
            "Write review_report.md when done."
        )

    async def run(  # type: ignore[override]
        self,
        *,
        blueprint: str,
        manifest: dict[str, Any],
        **kwargs: Any,
    ) -> AgentResult:
        """Run the reviewer for up to max_cycles fix cycles."""
        max_cycles = self._config.limits.reviewer_max_fix_cycles
        last_result: AgentResult | None = None

        for cycle in range(1, max_cycles + 1):
            system = self.render_system_prompt(
                blueprint=blueprint,
                manifest_json=json.dumps(manifest, indent=2),
                cycle=cycle,
                max_cycles=max_cycles,
                allow_pip_install=self._config.execution.allow_pip_install,
            )
            initial_message = self.build_initial_message(
                blueprint=blueprint,
                manifest=manifest,
                cycle=cycle,
            )
            ctx = self.make_context()

            self._audit.stage_start(stage=self._stage, agent=self.agent_name)
            try:
                result = await run_agent_loop(
                    agent=self.agent_name,
                    stage=self._stage,
                    model=self.model,
                    system=system,
                    initial_message=initial_message,
                    client=self._client,
                    registry=self._registry,
                    ctx=ctx,
                    audit=self._audit,
                    max_turns=self._config.limits.max_turns_per_agent,
                    max_tokens=self._config.limits.max_tokens_per_response,
                    context_compaction_chars=self._config.limits.context_compaction_chars,
                    compaction_keep_turns=self._config.limits.compaction_keep_turns,
                    compaction_model=self._config.models.runbook,
                )
                last_result = result
            except Exception:
                self._audit.stage_end(
                    stage=self._stage, agent=self.agent_name, status="failed", summary=""
                )
                raise
            self._audit.stage_end(
                stage=self._stage,
                agent=self.agent_name,
                status="done",
                summary=result.stage_summary,
            )

            # Check if the reviewer declared all tests pass in the summary
            summary_lower = result.stage_summary.lower()
            if "all pass" in summary_lower or "no issues" in summary_lower:
                break

        return last_result  # type: ignore[return-value]
