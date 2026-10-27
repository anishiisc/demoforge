"""Simulator agent — parallel stage: produce demo_simulation.html."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from demoforge.agents.base import BaseAgent
from demoforge.llm.agent_loop import AgentResult


class SimulatorAgent(BaseAgent):
    """Parallel stage: create a self-contained HTML simulation of the workflow."""

    agent_name = "simulator"
    prompt_template = "simulator"

    def build_initial_message(
        self,
        *,
        blueprint: str,
        plan: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        return (
            "Please write demo_simulation.html — a single self-contained file with "
            "inline CSS and JS that steps through the workflow. Embed representative synthetic data."
        )

    async def run(  # type: ignore[override]
        self,
        *,
        blueprint: str,
        plan: dict[str, Any],
        **kwargs: Any,
    ) -> AgentResult:
        result = await super().run(
            blueprint=blueprint,
            plan=plan,
            plan_json=json.dumps(plan, indent=2),
        )
        # Post-write validation
        html_path = self._workspace.simulation_path
        if html_path.exists():
            _validate_simulation_html(html_path)
        return result


def _validate_simulation_html(path: Path) -> None:
    """Raise ValueError if the HTML contains external src/href URLs."""
    content = path.read_text(encoding="utf-8")
    # Check for external src= or href= attributes pointing to http/https
    external_pattern = re.compile(
        r"""(?:src|href)\s*=\s*["']https?://""",
        re.IGNORECASE,
    )
    matches = external_pattern.findall(content)
    if matches:
        raise ValueError(
            f"demo_simulation.html contains {len(matches)} external URL reference(s). "
            "The simulation must be fully self-contained."
        )
    # Basic HTML parse check: must contain <html and </html>
    if "<html" not in content.lower():
        raise ValueError("demo_simulation.html does not appear to be valid HTML.")
