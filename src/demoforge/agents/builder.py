"""Builder agent — Stage 2: manifest-driven grouped file generation."""

from __future__ import annotations

import json
from typing import Any

from demoforge.agents.base import BaseAgent
from demoforge.llm.agent_loop import AgentResult, run_agent_loop
from demoforge.tools.registry import ToolContext


class BuilderAgent(BaseAgent):
    """Stage 2: create the demo project files described in the manifest.

    The orchestrator calls :meth:`run_group` once per file group rather than
    calling :meth:`run` directly. This keeps each loop context focused.
    """

    agent_name = "builder"
    prompt_template = "builder"

    def build_initial_message(
        self,
        *,
        target_files: list[dict[str, Any]],
        files_written: list[str],
        **kwargs: Any,
    ) -> str:
        targets = ", ".join(f["path"] for f in target_files)
        return (
            f"Please implement the following file(s) for the demo project: {targets}. "
            "Use write_file for each. Follow the blueprint exactly."
        )

    async def run_group(  # type: ignore[override]
        self,
        *,
        blueprint: str,
        manifest: dict[str, Any],
        target_files: list[dict[str, Any]],
        files_written: list[str],
        **kwargs: Any,
    ) -> AgentResult:
        """Run one Builder loop for the given *target_files* group."""
        system = self.render_system_prompt(
            blueprint=blueprint,
            manifest_json=json.dumps(manifest, indent=2),
            target_files=target_files,
            files_written=files_written,
        )
        initial_message = self.build_initial_message(
            target_files=target_files,
            files_written=files_written,
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
