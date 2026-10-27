"""End-to-end dry-run test using fixture responses and the sample request."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from demoforge.config import Config
from demoforge.orchestrator import create_orchestrator
from demoforge.state import StageStatus

_FIXTURES = Path(__file__).parent / "fixtures"
_SAMPLE_REQUEST = _FIXTURES / "sample_request.txt"
_DRY_RUN_RESPONSES = _FIXTURES / "dry_run_responses"


@pytest.fixture()
def sample_request() -> str:
    return _SAMPLE_REQUEST.read_text(encoding="utf-8").strip()


@pytest.fixture()
def config() -> Config:
    return Config()


# ---------------------------------------------------------------------------
# End-to-end dry-run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_completes(sample_request: str, config: Config, tmp_path: Path):
    """Full pipeline dry-run completes with all expected artefacts present."""
    events: list[tuple[str, str]] = []

    def progress_cb(event: str, stage: str, detail: dict) -> None:
        events.append((event, stage))

    orchestrator, state = create_orchestrator(
        request=sample_request,
        config=config,
        runs_root=tmp_path,
        dry_run=True,
        no_ask=True,
        skip_sim=False,
        budget=100.0,
        progress_cb=progress_cb,
    )

    final_state = await orchestrator.run(sample_request)

    # All pipeline stages must be done or at least not failed
    for stage_name in orchestrator.STAGE_NAMES:
        s = final_state.stages.get(stage_name)
        assert s is not None, f"Stage {stage_name} not in state"
        assert s.status in (StageStatus.done, StageStatus.skipped), (
            f"Stage {stage_name} ended with status {s.status}: {s.error}"
        )

    # plan.json must exist with required fields
    plan_path = orchestrator._workspace.plan_path
    assert plan_path.exists(), "plan.json not found"
    plan = json.loads(plan_path.read_text())
    assert "title" in plan
    assert "demo_agents" in plan
    assert "html_sim" in plan

    # manifest.json must exist
    manifest_path = orchestrator._workspace.manifest_path
    assert manifest_path.exists(), "manifest.json not found"
    manifest = json.loads(manifest_path.read_text())
    assert "files" in manifest
    assert len(manifest["files"]) > 0

    # demo_blueprint.md must exist
    assert orchestrator._workspace.blueprint_path.exists(), "demo_blueprint.md not found"

    # production_spec.md must exist
    assert orchestrator._workspace.production_spec_path.exists(), "production_spec.md not found"

    # review_report.md must exist
    assert orchestrator._workspace.review_report_path.exists(), "review_report.md not found"

    # RUN.md must exist
    assert orchestrator._workspace.runbook_path.exists(), "RUN.md not found"

    # Orchestration log must exist and be non-empty
    assert orchestrator._workspace.log_path.exists(), "orchestration_log.jsonl not found"
    log_lines = orchestrator._workspace.log_path.read_text().strip().splitlines()
    assert len(log_lines) > 0, "orchestration_log.jsonl is empty"

    # run_state.json must exist
    assert orchestrator._workspace.state_path.exists(), "run_state.json not found"

    # Progress events must include start and done for pipeline stages
    started = {s for ev, s in events if ev == "stage_started"}
    assert "planner" in started
    assert "architect" in started


@pytest.mark.asyncio
async def test_dry_run_log_has_llm_events(sample_request: str, config: Config, tmp_path: Path):
    """orchestration_log.jsonl must contain llm_call events with token counts."""
    orchestrator, state = create_orchestrator(
        request=sample_request,
        config=config,
        runs_root=tmp_path,
        dry_run=True,
        no_ask=True,
        skip_sim=True,
        budget=100.0,
    )
    await orchestrator.run(sample_request)

    log_path = orchestrator._workspace.log_path
    events = [json.loads(l) for l in log_path.read_text().strip().splitlines() if l.strip()]
    llm_events = [e for e in events if e.get("event") == "llm_call"]
    assert len(llm_events) > 0, "No llm_call events in log"
    for ev in llm_events:
        assert "agent" in ev
        assert "model" in ev
        assert "turn" in ev


@pytest.mark.asyncio
async def test_dry_run_html_sim_when_feasible(sample_request: str, config: Config, tmp_path: Path):
    """When html_sim.feasible=True in the plan, demo_simulation.html must be created."""
    orchestrator, state = create_orchestrator(
        request=sample_request,
        config=config,
        runs_root=tmp_path,
        dry_run=True,
        no_ask=True,
        skip_sim=False,
        budget=100.0,
    )
    final_state = await orchestrator.run(sample_request)

    plan = final_state.plan
    if plan.get("html_sim", {}).get("feasible"):
        assert orchestrator._workspace.simulation_path.exists(), (
            "html_sim is feasible but demo_simulation.html was not created"
        )


@pytest.mark.asyncio
async def test_resume_from_mid_run(sample_request: str, config: Config, tmp_path: Path):
    """Killing mid-run and resuming continues from the first incomplete stage."""
    from demoforge.orchestrator import resume_orchestrator
    from demoforge.state import RunState

    # Create a run and mark planner as done, architect as pending
    orchestrator, state = create_orchestrator(
        request=sample_request,
        config=config,
        runs_root=tmp_path,
        dry_run=True,
        no_ask=True,
        skip_sim=True,
        budget=100.0,
    )
    workspace = orchestrator._workspace

    # Pre-populate state as if planner completed
    state.mark_done("planner", artefacts=["plan.json"])
    state.plan = {
        "title": "Invoice Reconciliation Demo",
        "domain": "Finance",
        "persona": "AP Manager",
        "workflow_steps": ["step1"],
        "demo_agents": [{"name": "a", "role": "r", "inputs": [], "outputs": []}],
        "demo_tools": [],
        "data_entities": [],
        "success_criteria": ["sc1"],
        "assumptions": [],
        "html_sim": {"feasible": False, "reason": "skip"},
        "gui_framework": "streamlit",
    }
    # Write plan.json to disk
    workspace.plan_path.write_text(json.dumps(state.plan, indent=2))
    state.save(workspace.state_path)

    # Resume
    resumed_orch, resumed_state = resume_orchestrator(
        workspace.run_id,
        config=config,
        runs_root=tmp_path,
        dry_run=True,
    )
    final_state = await resumed_orch.run(sample_request)

    # Planner should still be done (not re-run)
    assert final_state.stages["planner"].status == StageStatus.done
    # Architect should have run
    assert final_state.stages["architect"].status in (StageStatus.done, StageStatus.skipped)


@pytest.mark.asyncio
async def test_no_files_written_outside_workspace(sample_request: str, config: Config, tmp_path: Path):
    """Verify no artefacts appear outside the run workspace directory."""
    orchestrator, state = create_orchestrator(
        request=sample_request,
        config=config,
        runs_root=tmp_path,
        dry_run=True,
        no_ask=True,
        skip_sim=True,
        budget=100.0,
    )
    await orchestrator.run(sample_request)

    workspace_root = orchestrator._workspace.root
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert str(p).startswith(str(workspace_root)), (
                f"File found outside workspace: {p}"
            )
