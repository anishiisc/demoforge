"""Pipeline orchestrator: sequences agents, manages parallel Simulator, handles resume."""

from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Any, Callable, Awaitable

from demoforge.audit.logger import AuditLogger
from demoforge.config import Config
from demoforge.llm.cost import BudgetExceeded, CostTracker
from demoforge.llm.agent_loop import AgentStalled
from demoforge.state import RunState, StageStatus
from demoforge.workspace import Workspace

# Type for progress callbacks: (event_type, stage, detail_dict)
ProgressCallback = Callable[[str, str, dict[str, Any]], Awaitable[None] | None]


class Orchestrator:
    """Sequence and execute the DemoForge pipeline stages.

    Parameters
    ----------
    config:
        Loaded configuration object.
    workspace:
        The run workspace for this execution.
    state:
        Run state (loaded for resume; fresh for new run).
    client:
        LLMClient or DryRunClient instance.
    audit:
        AuditLogger for this run.
    cost_tracker:
        Shared cost tracker.
    progress_cb:
        Optional async/sync callback receiving (event, stage, detail) progress events.
    ask_user_cb:
        Async callback for the Planner's ask_user tool.
    no_ask:
        If True, Planner records assumptions instead of asking.
    skip_sim:
        If True, skip the Simulator even if html_sim.feasible is True.
    """

    STAGE_NAMES = ["planner", "architect", "builder", "reviewer", "runbook", "documentation"]
    SIM_STAGE = "simulator"

    def __init__(
        self,
        config: Config,
        workspace: Workspace,
        state: RunState,
        client: Any,
        audit: AuditLogger,
        cost_tracker: CostTracker,
        progress_cb: ProgressCallback | None = None,
        ask_user_cb: Any = None,
        no_ask: bool = False,
        skip_sim: bool = False,
    ) -> None:
        self._config = config
        self._workspace = workspace
        self._state = state
        self._client = client
        self._audit = audit
        self._cost_tracker = cost_tracker
        self._progress_cb = progress_cb
        self._ask_user_cb = ask_user_cb
        self._no_ask = no_ask
        self._skip_sim = skip_sim

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def run(self, request: str) -> RunState:
        """Execute the full pipeline from the first incomplete stage."""
        self._state.request = request
        self._state.save(self._workspace.state_path)
        self._workspace.root.joinpath("request.txt").write_text(request, encoding="utf-8")

        all_stages = self.STAGE_NAMES
        start_from = self._state.first_incomplete(all_stages) or "done"

        sim_task: asyncio.Task | None = None

        try:
            for stage_name in all_stages:
                s = self._state.stages.get(stage_name)
                if s and s.status in (StageStatus.done, StageStatus.skipped):
                    continue
                if stage_name == start_from or (
                    self._state.stages.get(stage_name, None) is None
                ):
                    pass  # run this stage

                await self._emit("stage_started", stage_name, {})
                self._state.mark_running(stage_name)
                self._state.save(self._workspace.state_path)

                try:
                    artefacts = await self._run_stage(stage_name, sim_task=sim_task)
                    self._state.mark_done(stage_name, artefacts=artefacts)
                    self._state.cost_usd = self._cost_tracker.total_usd
                    self._state.save(self._workspace.state_path)
                    await self._emit("stage_done", stage_name, {"artefacts": artefacts})

                    # Start Simulator after architect if feasible
                    if stage_name == "architect" and not self._skip_sim:
                        plan = self._state.plan
                        if plan.get("html_sim", {}).get("feasible"):
                            sim_task = asyncio.create_task(
                                self._run_simulator(), name="simulator"
                            )

                except (BudgetExceeded, AgentStalled) as exc:
                    self._state.mark_failed(stage_name, error=str(exc))
                    self._state.save(self._workspace.state_path)
                    await self._emit("stage_failed", stage_name, {"error": str(exc)})
                    raise
                except Exception as exc:
                    tb = traceback.format_exc()
                    self._state.mark_failed(stage_name, error=str(exc))
                    self._state.save(self._workspace.state_path)
                    self._audit.error(
                        stage=stage_name,
                        agent=stage_name,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        traceback=tb,
                    )
                    await self._emit("stage_failed", stage_name, {"error": str(exc)})
                    raise

            # Wait for Simulator before Stage 5 (documentation already done, but guard)
            if sim_task is not None:
                try:
                    await sim_task
                    self._state.mark_done(self.SIM_STAGE)
                except Exception as exc:
                    # Simulator failure is non-fatal
                    self._state.mark_failed(self.SIM_STAGE, error=str(exc))
                    self._audit.error(
                        stage="simulator",
                        agent="simulator",
                        error_type=type(exc).__name__,
                        message=str(exc),
                        traceback=traceback.format_exc(),
                    )

        finally:
            self._state.cost_usd = self._cost_tracker.total_usd
            self._state.save(self._workspace.state_path)

        return self._state

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    async def _run_stage(self, stage_name: str, *, sim_task: Any = None) -> list[str]:
        """Dispatch to the appropriate stage handler."""
        registry = self._build_registry()
        client = self._client.for_stage(stage_name)

        if stage_name == "planner":
            return await self._run_planner(client, registry)
        elif stage_name == "architect":
            return await self._run_architect(client, registry)
        elif stage_name == "builder":
            return await self._run_builder(client, registry)
        elif stage_name == "reviewer":
            return await self._run_reviewer(client, registry)
        elif stage_name == "runbook":
            return await self._run_runbook(client, registry)
        elif stage_name == "documentation":
            # Wait for simulator before documentation
            if sim_task is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(sim_task), timeout=None)
                except Exception:
                    pass
            return await self._run_documentation(client, registry)
        else:
            raise ValueError(f"Unknown stage: {stage_name}")

    async def _run_planner(self, client: Any, registry: Any) -> list[str]:
        from demoforge.agents.planner import PlannerAgent
        agent = PlannerAgent(
            config=self._config,
            client=client,
            registry=registry,
            audit=self._audit,
            workspace=self._workspace,
            stage="planner",
        )
        result = await agent.run(
            request=self._state.request,
            no_ask=self._no_ask,
            ask_user_cb=self._ask_user_cb,
        )
        # Load the plan from disk
        plan_path = self._workspace.plan_path
        if plan_path.exists():
            self._state.plan = json.loads(plan_path.read_text())
        self._audit.handoff(
            from_stage="planner", to_stage="architect", artefacts=["plan.json"]
        )
        return ["plan.json"]

    async def _run_architect(self, client: Any, registry: Any) -> list[str]:
        from demoforge.agents.architect import ArchitectAgent
        agent = ArchitectAgent(
            config=self._config,
            client=client,
            registry=registry,
            audit=self._audit,
            workspace=self._workspace,
            stage="architect",
        )
        result = await agent.run(plan=self._state.plan)
        # Load manifest
        manifest_path = self._workspace.manifest_path
        if manifest_path.exists():
            self._state.manifest = json.loads(manifest_path.read_text())
        artefacts = ["demo_blueprint.md", "production_spec.md", "manifest.json"]
        self._audit.handoff(from_stage="architect", to_stage="builder", artefacts=artefacts)
        return artefacts

    async def _run_builder(self, client: Any, registry: Any) -> list[str]:
        from demoforge.agents.builder import BuilderAgent

        blueprint_path = self._workspace.blueprint_path
        blueprint = blueprint_path.read_text(encoding="utf-8") if blueprint_path.exists() else ""
        manifest = self._state.manifest
        files = manifest.get("files", [])

        # Group files in dependency order (group size 1 by default)
        import os
        group_size = int(os.environ.get("BUILDER_GROUP_SIZE", "1"))

        agent = BuilderAgent(
            config=self._config,
            client=client,
            registry=registry,
            audit=self._audit,
            workspace=self._workspace,
            stage="builder",
        )

        files_written: list[str] = []
        for i in range(0, len(files), group_size):
            group = files[i : i + group_size]
            await agent.run_group(
                blueprint=blueprint,
                manifest=manifest,
                target_files=group,
                files_written=files_written,
            )
            files_written.extend(f["path"] for f in group)
            await self._emit("files_written", "builder", {"files": [f["path"] for f in group]})

        self._audit.handoff(
            from_stage="builder", to_stage="reviewer", artefacts=files_written
        )
        return files_written

    async def _run_reviewer(self, client: Any, registry: Any) -> list[str]:
        from demoforge.agents.reviewer import ReviewerAgent

        blueprint = (
            self._workspace.blueprint_path.read_text(encoding="utf-8")
            if self._workspace.blueprint_path.exists()
            else ""
        )
        agent = ReviewerAgent(
            config=self._config,
            client=client,
            registry=registry,
            audit=self._audit,
            workspace=self._workspace,
            stage="reviewer",
        )
        await agent.run(blueprint=blueprint, manifest=self._state.manifest)
        self._audit.handoff(
            from_stage="reviewer", to_stage="runbook", artefacts=["review_report.md"]
        )
        return ["review_report.md"]

    async def _run_runbook(self, client: Any, registry: Any) -> list[str]:
        from demoforge.agents.runbook import RunbookAgent

        blueprint = (
            self._workspace.blueprint_path.read_text(encoding="utf-8")
            if self._workspace.blueprint_path.exists()
            else ""
        )
        review_report = (
            self._workspace.review_report_path.read_text(encoding="utf-8")
            if self._workspace.review_report_path.exists()
            else ""
        )
        # Generate file tree
        file_tree = await _build_file_tree(self._workspace)

        agent = RunbookAgent(
            config=self._config,
            client=client,
            registry=registry,
            audit=self._audit,
            workspace=self._workspace,
            stage="runbook",
        )
        await agent.run(blueprint=blueprint, review_report=review_report, file_tree=file_tree)
        self._audit.handoff(
            from_stage="runbook", to_stage="documentation", artefacts=["RUN.md"]
        )
        return ["RUN.md"]

    async def _run_documentation(self, client: Any, registry: Any) -> list[str]:
        from demoforge.agents.documentation import DocumentationAgent

        artefacts_summary = _build_artefacts_summary(self._state)
        log_summary = _build_log_summary(self._state)

        agent = DocumentationAgent(
            config=self._config,
            client=client,
            registry=registry,
            audit=self._audit,
            workspace=self._workspace,
            stage="documentation",
        )
        await agent.run(artefacts_summary=artefacts_summary, log_summary=log_summary)
        return ["tech_note.md", "MANIFEST.md"]

    async def _run_simulator(self) -> None:
        """Run the Simulator stage as an asyncio task."""
        from demoforge.agents.simulator import SimulatorAgent
        client = self._client.for_stage("simulator")
        registry = self._build_registry()
        blueprint = (
            self._workspace.blueprint_path.read_text(encoding="utf-8")
            if self._workspace.blueprint_path.exists()
            else ""
        )
        agent = SimulatorAgent(
            config=self._config,
            client=client,
            registry=registry,
            audit=self._audit,
            workspace=self._workspace,
            stage="simulator",
        )
        await agent.run(blueprint=blueprint, plan=self._state.plan)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_registry(self) -> Any:
        from demoforge.tools.registry import build_registry
        return build_registry(self._workspace, self._config)

    async def _emit(self, event: str, stage: str, detail: dict[str, Any]) -> None:
        if self._progress_cb is None:
            return
        result = self._progress_cb(event, stage, detail)
        if asyncio.iscoroutine(result):
            await result


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_orchestrator(
    *,
    request: str,
    config: Config,
    runs_root: Path | None = None,
    dry_run: bool = False,
    no_ask: bool = False,
    skip_sim: bool = False,
    budget: float | None = None,
    progress_cb: ProgressCallback | None = None,
    ask_user_cb: Any = None,
) -> tuple["Orchestrator", RunState]:
    """Create an Orchestrator and a fresh RunState for a new run.

    Returns the orchestrator and the initial state.
    """
    from demoforge.workspace import make_run_id, Workspace
    from demoforge.audit.logger import AuditLogger
    from demoforge.llm.cost import CostTracker

    if budget is not None:
        config = config.model_copy(
            update={"limits": config.limits.model_copy(update={"run_budget_usd": budget})}
        )

    # Derive a run_id from the first words of the request
    title_slug = request.strip().split()[0] if request.strip() else "demo"
    run_id = make_run_id(title_slug)

    workspace = Workspace(run_id, runs_root=runs_root)
    workspace.create()

    state = RunState.new(run_id, request=request)

    audit = AuditLogger(
        log_path=workspace.log_path,
        payloads_dir=workspace.payloads_dir,
        run_id=run_id,
    )
    cost_tracker = CostTracker(config)

    if dry_run:
        from demoforge.llm.dry_run import DryRunClient
        client: Any = DryRunClient(
            config=config,
            cost_tracker=cost_tracker,
            audit=audit,
            stage="planner",
        )
    else:
        from demoforge.llm.client import LLMClient
        client = LLMClient(
            config=config,
            cost_tracker=cost_tracker,
            audit=audit,
            stage="planner",
        )

    orchestrator = Orchestrator(
        config=config,
        workspace=workspace,
        state=state,
        client=client,
        audit=audit,
        cost_tracker=cost_tracker,
        progress_cb=progress_cb,
        ask_user_cb=ask_user_cb,
        no_ask=no_ask,
        skip_sim=skip_sim,
    )
    return orchestrator, state


def resume_orchestrator(
    run_id: str,
    *,
    config: Config,
    runs_root: Path | None = None,
    dry_run: bool = False,
    progress_cb: ProgressCallback | None = None,
    ask_user_cb: Any = None,
) -> tuple["Orchestrator", RunState]:
    """Load an existing run state and create an orchestrator to resume from it."""
    from demoforge.workspace import Workspace
    from demoforge.audit.logger import AuditLogger
    from demoforge.llm.cost import CostTracker

    workspace = Workspace(run_id, runs_root=runs_root)
    state = RunState.load(workspace.state_path)

    audit = AuditLogger(
        log_path=workspace.log_path,
        payloads_dir=workspace.payloads_dir,
        run_id=run_id,
    )
    cost_tracker = CostTracker(config)

    if dry_run:
        from demoforge.llm.dry_run import DryRunClient
        client: Any = DryRunClient(
            config=config,
            cost_tracker=cost_tracker,
            audit=audit,
            stage="planner",
        )
    else:
        from demoforge.llm.client import LLMClient
        client = LLMClient(
            config=config,
            cost_tracker=cost_tracker,
            audit=audit,
            stage="planner",
        )

    orchestrator = Orchestrator(
        config=config,
        workspace=workspace,
        state=state,
        client=client,
        audit=audit,
        cost_tracker=cost_tracker,
        progress_cb=progress_cb,
        ask_user_cb=ask_user_cb,
    )
    return orchestrator, state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _build_file_tree(workspace: Workspace) -> str:
    """Return a simple text tree of the demo directory."""
    lines: list[str] = []
    demo = workspace.demo_dir
    if demo.exists():
        for p in sorted(demo.rglob("*")):
            rel = p.relative_to(demo)
            depth = len(rel.parts) - 1
            prefix = "  " * depth
            lines.append(f"{prefix}{p.name}{'/' if p.is_dir() else ''}")
    return "\n".join(lines) or "(empty)"


def _build_artefacts_summary(state: RunState) -> str:
    """Build a markdown table of artefacts from the run state."""
    rows = ["| Stage | Artefact |", "|-------|----------|"]
    for stage_name, s in state.stages.items():
        for art in s.artefacts:
            rows.append(f"| {stage_name} | {art} |")
    return "\n".join(rows)


def _build_log_summary(state: RunState) -> str:
    """Return a brief summary of run cost and stage statuses."""
    lines = [f"Total cost: ${state.cost_usd:.4f}"]
    for name, s in state.stages.items():
        lines.append(f"- {name}: {s.status.value}")
    if state.errors:
        lines.append("\nErrors:")
        lines.extend(f"  - {e}" for e in state.errors)
    return "\n".join(lines)
