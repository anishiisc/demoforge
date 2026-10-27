"""BaseAgent: shared prompt rendering and loop invocation for all DemoForge agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from demoforge.llm.agent_loop import AgentResult, run_agent_loop
from demoforge.tools.registry import ToolContext

if TYPE_CHECKING:
    from demoforge.config import Config
    from demoforge.audit.logger import AuditLogger
    from demoforge.llm.client import LLMClient
    from demoforge.llm.dry_run import DryRunClient
    from demoforge.tools.registry import ToolRegistry
    from demoforge.workspace import Workspace

_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"
_SHARED_RULES_PATH = _PROMPTS_DIR / "shared_rules.md"


class BaseAgent:
    """Base class for all DemoForge pipeline agents.

    Subclasses set :attr:`agent_name` and override :meth:`build_initial_message`.
    """

    #: Override in subclass to match the agent name used in config and the tool registry.
    agent_name: str = ""
    #: Override to match the prompt template filename (without .j2 extension).
    prompt_template: str = ""

    def __init__(
        self,
        config: "Config",
        client: "LLMClient | DryRunClient",
        registry: "ToolRegistry",
        audit: "AuditLogger",
        workspace: "Workspace",
        stage: str,
    ) -> None:
        self._config = config
        self._client = client
        self._registry = registry
        self._audit = audit
        self._workspace = workspace
        self._stage = stage
        self._jinja = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            undefined=StrictUndefined,
            autoescape=False,
        )

    @property
    def model(self) -> str:
        """Return the configured model for this agent."""
        return self._config.models.for_agent(self.agent_name)

    def render_system_prompt(self, **kwargs: Any) -> str:
        """Render the Jinja2 system prompt template with *kwargs* plus shared_rules."""
        shared_rules = _SHARED_RULES_PATH.read_text(encoding="utf-8")
        template_name = f"{self.prompt_template}.md.j2"
        template = self._jinja.get_template(template_name)
        return template.render(shared_rules=shared_rules, **kwargs)

    def build_initial_message(self, **kwargs: Any) -> str:
        """Build the first user message for this agent. Override in subclass."""
        raise NotImplementedError

    def make_context(self, ask_user_cb: Any = None) -> ToolContext:
        """Create a ToolContext for this agent."""
        return ToolContext(
            workspace=self._workspace,
            agent=self.agent_name,
            stage=self._stage,
            run_id=self._workspace.run_id,
            ask_user_callback=ask_user_cb,
            config=self._config,
        )

    async def run(self, ask_user_cb: Any = None, **kwargs: Any) -> AgentResult:
        """Render prompts, run the agent loop, and return the result."""
        system = self.render_system_prompt(**kwargs)
        initial_message = self.build_initial_message(**kwargs)
        ctx = self.make_context(ask_user_cb)

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
                compaction_model=self._config.models.runbook,  # use haiku for compaction
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
