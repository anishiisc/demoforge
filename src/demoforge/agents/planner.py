"""Planner agent — Stage 0: parse the request and produce plan.json."""

from __future__ import annotations

from typing import Any

from demoforge.agents.base import BaseAgent
from demoforge.llm.agent_loop import AgentResult


class PlannerAgent(BaseAgent):
    """Stage 0: analyse the user request and produce a validated plan.json."""

    agent_name = "planner"
    prompt_template = "planner"

    def build_initial_message(self, *, request: str, no_ask: bool = False, **kwargs: Any) -> str:
        """Return the initial user message for the Planner."""
        return (
            f"Please analyse the following request and produce a plan:\n\n{request}"
        )

    async def run(  # type: ignore[override]
        self,
        *,
        request: str,
        no_ask: bool = False,
        ask_user_cb: Any = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run the planner with *request* as input.

        In no-ask mode the ask_user callback is replaced with one that records
        assumptions instead of prompting.
        """
        from demoforge.llm.agent_loop import run_agent_loop
        from demoforge.tools.registry import ToolContext

        system = self.render_system_prompt(request=request, no_ask=no_ask)
        initial_message = self.build_initial_message(request=request, no_ask=no_ask)
        ctx = self.make_context(ask_user_cb=ask_user_cb if not no_ask else _no_ask_cb)

        # Force submit_plan on an early turn to ensure structured output
        # (we do NOT force on turn 0 since the model may need to think first)
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
        return result


async def _no_ask_cb(question: str) -> str:
    """Default callback for no-ask mode: return a placeholder."""
    return f"[no-ask mode] Assumption recorded: {question}"
