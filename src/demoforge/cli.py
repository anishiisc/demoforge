"""DemoForge CLI — typer entrypoint."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="demoforge",
    help="Generate complete mock AI demos from natural-language descriptions.",
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("new")
def cmd_new(
    request: Optional[str] = typer.Argument(None, help="Free-text use case description"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Read request from file"),
    no_ask: bool = typer.Option(False, "--no-ask", help="Skip clarifying questions; record assumptions"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Use fixture responses instead of calling Anthropic"),
    skip_sim: bool = typer.Option(False, "--skip-sim", help="Skip the HTML Simulator stage"),
    budget: float = typer.Option(25.0, "--budget", help="Cost limit in USD (default 25)"),
) -> None:
    """Start a new DemoForge run."""
    if file:
        request_text = file.read_text(encoding="utf-8").strip()
    elif request:
        request_text = request.strip()
    else:
        console.print("[red]Provide a request string or --file path.[/red]")
        raise typer.Exit(1)

    if not request_text:
        console.print("[red]Request is empty.[/red]")
        raise typer.Exit(1)

    from demoforge.config import get_config
    from demoforge.orchestrator import create_orchestrator

    config = get_config()

    async def _ask_user(question: str) -> str:
        console.print(f"\n[bold yellow]Planner asks:[/bold yellow] {question}")
        return typer.prompt("Your answer")

    ask_cb = None if no_ask else _ask_user

    def progress_cb(event: str, stage: str, detail: dict) -> None:
        if event == "stage_started":
            console.print(f"[cyan]>>>[/cyan] Starting stage: [bold]{stage}[/bold]")
        elif event == "stage_done":
            arts = detail.get("artefacts", [])
            console.print(f"[green]✓[/green] Stage [bold]{stage}[/bold] done. Artefacts: {', '.join(arts) or '(none)'}")
        elif event == "stage_failed":
            console.print(f"[red]✗[/red] Stage [bold]{stage}[/bold] FAILED: {detail.get('error', '')}")
        elif event == "files_written":
            for f in detail.get("files", []):
                console.print(f"    [dim]wrote[/dim] {f}")

    orchestrator, state = create_orchestrator(
        request=request_text,
        config=config,
        dry_run=dry_run,
        no_ask=no_ask,
        skip_sim=skip_sim,
        budget=budget,
        progress_cb=progress_cb,
        ask_user_cb=ask_cb,
    )

    console.print(f"\n[bold]DemoForge run:[/bold] {state.run_id}\n")

    try:
        final_state = asyncio.run(orchestrator.run(request_text))
        console.print(f"\n[bold green]Run complete.[/bold green] Cost: ${final_state.cost_usd:.4f}")
        console.print(f"Run directory: runs/{final_state.run_id}/")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Use 'demoforge resume' to continue.[/yellow]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"\n[red]Run failed:[/red] {exc}")
        raise typer.Exit(1)


@app.command("resume")
def cmd_resume(
    run_id: str = typer.Argument(..., help="Run ID to resume"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Resume a failed or interrupted run."""
    from demoforge.config import get_config
    from demoforge.orchestrator import resume_orchestrator
    from demoforge.state import RunState
    from demoforge.workspace import Workspace

    config = get_config()
    workspace = Workspace(run_id)
    if not workspace.state_path.exists():
        console.print(f"[red]No run state found for {run_id}[/red]")
        raise typer.Exit(1)

    state = RunState.load(workspace.state_path)
    console.print(f"[bold]Resuming run:[/bold] {run_id}")

    def progress_cb(event: str, stage: str, detail: dict) -> None:
        if event == "stage_started":
            console.print(f"[cyan]>>>[/cyan] Starting stage: [bold]{stage}[/bold]")
        elif event == "stage_done":
            console.print(f"[green]✓[/green] Stage [bold]{stage}[/bold] done.")
        elif event == "stage_failed":
            console.print(f"[red]✗[/red] Stage [bold]{stage}[/bold] FAILED: {detail.get('error', '')}")

    orchestrator, state = resume_orchestrator(
        run_id,
        config=config,
        dry_run=dry_run,
        progress_cb=progress_cb,
    )

    try:
        final_state = asyncio.run(orchestrator.run(state.request))
        console.print(f"\n[bold green]Run complete.[/bold green] Cost: ${final_state.cost_usd:.4f}")
    except Exception as exc:
        console.print(f"\n[red]Run failed:[/red] {exc}")
        raise typer.Exit(1)


@app.command("status")
def cmd_status(
    run_id: str = typer.Argument(..., help="Run ID to inspect"),
) -> None:
    """Show the status of a run."""
    from demoforge.workspace import Workspace
    from demoforge.state import RunState

    workspace = Workspace(run_id)
    if not workspace.state_path.exists():
        console.print(f"[red]No run state found for {run_id}[/red]")
        raise typer.Exit(1)

    state = RunState.load(workspace.state_path)

    table = Table(title=f"Run: {run_id}", show_header=True)
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Summary")

    status_styles = {
        "done": "green",
        "running": "yellow",
        "failed": "red",
        "pending": "dim",
        "skipped": "cyan",
    }

    for name, s in state.stages.items():
        style = status_styles.get(s.status.value, "")
        table.add_row(name, f"[{style}]{s.status.value}[/{style}]", s.summary[:80])

    console.print(table)
    console.print(f"[dim]Total cost:[/dim] ${state.cost_usd:.4f}")
    if state.errors:
        console.print("[red]Errors:[/red]")
        for e in state.errors:
            console.print(f"  {e}")


@app.command("logs")
def cmd_logs(
    run_id: str = typer.Argument(..., help="Run ID"),
    stage: Optional[str] = typer.Option(None, "--stage", "-s", help="Filter by stage"),
    event_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by event type"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max lines to show"),
) -> None:
    """View the orchestration log for a run."""
    from demoforge.workspace import Workspace

    workspace = Workspace(run_id)
    if not workspace.log_path.exists():
        console.print(f"[red]No log found for {run_id}[/red]")
        raise typer.Exit(1)

    shown = 0
    with workspace.log_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if stage and record.get("stage") != stage:
                continue
            if event_type and record.get("event") != event_type:
                continue
            console.print_json(line)
            shown += 1
            if shown >= limit:
                console.print(f"[dim]... (limit {limit} reached)[/dim]")
                break


if __name__ == "__main__":
    app()
