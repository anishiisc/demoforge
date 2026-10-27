"""RunState model and persistence to run_state.json."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class StageStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"
    skipped = "skipped"


class StageState(BaseModel):
    """State for a single pipeline stage."""

    name: str
    status: StageStatus = StageStatus.pending
    summary: str = ""
    artefacts: list[str] = Field(default_factory=list)
    error: str = ""


class RunState(BaseModel):
    """Persisted state for one DemoForge run."""

    run_id: str
    request: str = ""
    stages: dict[str, StageState] = Field(default_factory=dict)
    cost_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)
    plan: dict[str, Any] = Field(default_factory=dict)
    manifest: dict[str, Any] = Field(default_factory=dict)

    def stage(self, name: str) -> StageState:
        """Return (or create) the StageState for *name*."""
        if name not in self.stages:
            self.stages[name] = StageState(name=name)
        return self.stages[name]

    def mark_running(self, name: str) -> None:
        self.stage(name).status = StageStatus.running

    def mark_done(self, name: str, *, summary: str = "", artefacts: list[str] | None = None) -> None:
        s = self.stage(name)
        s.status = StageStatus.done
        s.summary = summary
        if artefacts:
            s.artefacts = artefacts

    def mark_failed(self, name: str, *, error: str = "") -> None:
        s = self.stage(name)
        s.status = StageStatus.failed
        s.error = error
        if error:
            self.errors.append(f"[{name}] {error}")

    def mark_skipped(self, name: str) -> None:
        self.stage(name).status = StageStatus.skipped

    def first_incomplete(self, stage_names: list[str]) -> str | None:
        """Return the first stage that is not done or skipped."""
        for name in stage_names:
            s = self.stages.get(name)
            if s is None or s.status not in (StageStatus.done, StageStatus.skipped):
                return name
        return None

    def save(self, path: Path) -> None:
        """Persist state to *path* as JSON."""
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RunState":
        """Load state from *path*."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    @classmethod
    def new(cls, run_id: str, request: str = "") -> "RunState":
        """Create a fresh RunState."""
        return cls(run_id=run_id, request=request)
