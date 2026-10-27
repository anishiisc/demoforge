"""Tests for workspace path-guard (sandbox) behaviour."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from demoforge.workspace import Workspace, WorkspaceError, make_run_id


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    ws = Workspace("20240101-120000-test01", runs_root=tmp_path)
    ws.create()
    return ws


# ---------------------------------------------------------------------------
# make_run_id
# ---------------------------------------------------------------------------


def test_make_run_id_format():
    run_id = make_run_id("Invoice Reconciliation Demo")
    parts = run_id.split("-")
    assert len(parts) == 3
    # Date part: YYYYMMDD
    assert len(parts[0]) == 8 and parts[0].isdigit()
    # Time part: HHMMSS
    assert len(parts[1]) == 6 and parts[1].isdigit()
    # Slug: at most 6 lowercase alphanumeric chars
    assert len(parts[2]) <= 6
    assert parts[2].isalnum()


def test_make_run_id_empty_title():
    run_id = make_run_id("")
    assert run_id.endswith("-noname")


def test_make_run_id_special_chars():
    run_id = make_run_id("!!! test @#$")
    _, _, slug = run_id.split("-")
    assert slug == "test"


# ---------------------------------------------------------------------------
# Path traversal rejection
# ---------------------------------------------------------------------------


def test_path_traversal_dotdot(ws: Workspace):
    with pytest.raises(WorkspaceError):
        ws.guard_path("../../etc/passwd")


def test_path_traversal_absolute_outside(ws: Workspace):
    with pytest.raises(WorkspaceError):
        ws.guard_path("/etc/passwd")


def test_valid_relative_path(ws: Workspace):
    result = ws.guard_path("agents/intake.py")
    assert result.is_relative_to(ws.demo_dir) or result.is_relative_to(ws.root)


def test_valid_nested_path(ws: Workspace):
    result = ws.guard_path("data/invoices.csv")
    assert str(result).startswith(str(ws.root))


def test_path_within_run_root_via_doc(ws: Workspace):
    result = ws.guard_doc_path("plan.json")
    assert result == ws.root / "plan.json"


def test_doc_path_traversal_rejected(ws: Workspace):
    with pytest.raises(WorkspaceError):
        ws.guard_doc_path("../../etc/passwd")


def test_symlink_escape(ws: Workspace, tmp_path: Path):
    """A symlink inside the workspace pointing outside should be rejected."""
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = ws.demo_dir / "evil_link.txt"
    link.symlink_to(outside)
    with pytest.raises(WorkspaceError):
        ws.guard_path("evil_link.txt")


# ---------------------------------------------------------------------------
# Workspace structure
# ---------------------------------------------------------------------------


def test_workspace_dirs_created(ws: Workspace):
    for subdir in ["data", "agents", "tools", "audit", "gui", "logs", "tests"]:
        assert (ws.demo_dir / subdir).is_dir()
    assert ws.payloads_dir.is_dir()
