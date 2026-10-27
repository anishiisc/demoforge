"""Run workspace creation and path-guard utilities."""

from __future__ import annotations

import re
import time
from pathlib import Path

_RUNS_DIR = Path(__file__).parent.parent.parent / "runs"

_SLUG_RE = re.compile(r"[^a-z0-9]")


class WorkspaceError(Exception):
    """Raised when a path would escape the run workspace."""


def make_run_id(title: str) -> str:
    """Generate a run_id from *title* using format YYYYMMDD-HHMMSS-<6-char slug>."""
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    slug = _SLUG_RE.sub("", title.lower())[:6]
    if not slug:
        slug = "noname"
    return f"{ts}-{slug}"


class Workspace:
    """Manages the directory tree for a single DemoForge run."""

    def __init__(self, run_id: str, runs_root: Path | None = None) -> None:
        self.run_id = run_id
        self.root: Path = (runs_root or _RUNS_DIR) / run_id
        self.demo_dir: Path = self.root / "demo"
        self.payloads_dir: Path = self.root / "llm_payloads"

    def create(self) -> None:
        """Create all required subdirectories."""
        for subdir in [
            self.root,
            self.payloads_dir,
            self.demo_dir / "data",
            self.demo_dir / "agents",
            self.demo_dir / "tools",
            self.demo_dir / "audit",
            self.demo_dir / "gui",
            self.demo_dir / "logs",
            self.demo_dir / "tests",
        ]:
            subdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path resolution helpers
    # ------------------------------------------------------------------

    def guard_path(self, path: str | Path, *, allow_run_root: bool = False) -> Path:
        """Resolve *path* and verify it sits inside the workspace.

        Raises :class:`WorkspaceError` for any attempt to escape via ``..``,
        absolute paths outside the workspace, or symlinks that exit the tree.

        Parameters
        ----------
        path:
            Relative or absolute path to validate.
        allow_run_root:
            If True, allow the run root itself (not just demo/).
            Documentation-type files (plan.json, review_report.md, etc.) live
            in the run root; demo source files live under demo/.
        """
        # Resolve relative to demo dir unless it's explicitly under run root
        raw = Path(path)
        if raw.is_absolute():
            candidate = raw.resolve()
        else:
            # Try demo dir first
            candidate = (self.demo_dir / raw).resolve()

        # For run-root docs we accept paths relative to run root too.
        if allow_run_root:
            candidate_from_root = (self.root / raw).resolve()
            if _is_inside(candidate_from_root, self.root) and not _escapes(
                candidate_from_root, self.root
            ):
                return candidate_from_root

        if not _is_inside(candidate, self.root):
            raise WorkspaceError(
                f"Path '{path}' resolves outside the run workspace '{self.root}'"
            )
        if _escapes(candidate, self.root):
            raise WorkspaceError(
                f"Path '{path}' contains a symlink that escapes the run workspace"
            )
        return candidate

    def guard_doc_path(self, path: str | Path) -> Path:
        """Like guard_path but for artefacts in the run root (not demo/)."""
        raw = Path(path)
        if raw.is_absolute():
            candidate = raw.resolve()
        else:
            candidate = (self.root / raw).resolve()
        if not _is_inside(candidate, self.root):
            raise WorkspaceError(
                f"Path '{path}' resolves outside the run root '{self.root}'"
            )
        if _escapes(candidate, self.root):
            raise WorkspaceError(
                f"Path '{path}' contains a symlink that escapes the run root"
            )
        return candidate

    # Convenience path properties
    @property
    def log_path(self) -> Path:
        return self.root / "orchestration_log.jsonl"

    @property
    def state_path(self) -> Path:
        return self.root / "run_state.json"

    @property
    def plan_path(self) -> Path:
        return self.root / "plan.json"

    @property
    def blueprint_path(self) -> Path:
        return self.root / "demo_blueprint.md"

    @property
    def production_spec_path(self) -> Path:
        return self.root / "production_spec.md"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def review_report_path(self) -> Path:
        return self.root / "review_report.md"

    @property
    def runbook_path(self) -> Path:
        return self.root / "RUN.md"

    @property
    def simulation_path(self) -> Path:
        return self.root / "demo_simulation.html"

    @property
    def tech_note_path(self) -> Path:
        return self.root / "tech_note.md"

    @property
    def manifest_doc_path(self) -> Path:
        return self.root / "MANIFEST.md"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_inside(path: Path, root: Path) -> bool:
    """Return True if *path* is the root or a descendant of root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _escapes(path: Path, root: Path) -> bool:
    """Return True if any component of *path* is a symlink pointing outside *root*."""
    check = Path(root)
    # Walk each part to detect intermediate symlinks
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    for part in parts:
        check = check / part
        if check.is_symlink():
            target = check.resolve()
            if not _is_inside(target, root):
                return True
    return False
