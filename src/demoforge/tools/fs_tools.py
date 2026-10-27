"""File-system tools: read_file, write_file, list_files, edit_file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from demoforge.workspace import WorkspaceError

if TYPE_CHECKING:
    from demoforge.tools.registry import ToolContext

_MAX_READ_CHARS = 100_000  # truncate very large files before returning to model


async def handle_read_file(*, ctx: "ToolContext", path: str) -> str:
    """Read a file from the run workspace and return its contents."""
    try:
        resolved = ctx.workspace.guard_path(path, allow_run_root=True)
    except WorkspaceError as exc:
        return f"ERROR: {exc}"
    if not resolved.exists():
        return f"ERROR: File not found: {path}"
    text = resolved.read_text(encoding="utf-8", errors="replace")
    if len(text) > _MAX_READ_CHARS:
        text = text[:_MAX_READ_CHARS] + f"\n... [truncated at {_MAX_READ_CHARS} chars]"
    return text


async def handle_write_file(*, ctx: "ToolContext", path: str, content: str) -> str:
    """Write *content* to *path* inside the workspace."""
    # Check file size limit
    cfg = ctx.config
    limit = cfg.limits.max_file_size_bytes if cfg else 524288
    if len(content.encode("utf-8")) > limit:
        kb = limit // 1024
        return (
            f"ERROR: Content exceeds the {kb} KB file size limit. "
            "Split the file into smaller pieces and write them separately."
        )
    try:
        resolved = ctx.workspace.guard_path(path, allow_run_root=True)
    except WorkspaceError as exc:
        return f"ERROR: {exc}"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return f"OK: wrote {len(content)} chars to {path}"


async def handle_list_files(*, ctx: "ToolContext", subdir: str = ".") -> str:
    """Return a tree listing of the workspace (or a subdirectory)."""
    try:
        base = ctx.workspace.guard_path(subdir, allow_run_root=True)
    except WorkspaceError as exc:
        return f"ERROR: {exc}"
    if not base.exists():
        return f"ERROR: Directory not found: {subdir}"
    lines: list[str] = []
    _tree(base, base, lines, max_entries=500)
    return "\n".join(lines) or "(empty)"


def _tree(root: Path, current: Path, lines: list[str], *, depth: int = 0, max_entries: int) -> None:
    if len(lines) >= max_entries:
        lines.append("... [truncated]")
        return
    indent = "  " * depth
    for child in sorted(current.iterdir()):
        lines.append(f"{indent}{child.name}{'/' if child.is_dir() else ''}")
        if child.is_dir() and not child.is_symlink():
            _tree(root, child, lines, depth=depth + 1, max_entries=max_entries)


async def handle_edit_file(*, ctx: "ToolContext", path: str, old: str, new: str) -> str:
    """Replace the first occurrence of *old* with *new* in *path*."""
    try:
        resolved = ctx.workspace.guard_path(path, allow_run_root=True)
    except WorkspaceError as exc:
        return f"ERROR: {exc}"
    if not resolved.exists():
        return f"ERROR: File not found: {path}"
    text = resolved.read_text(encoding="utf-8")
    if old not in text:
        return f"ERROR: Exact string not found in {path}. No changes made."
    new_text = text.replace(old, new, 1)
    resolved.write_text(new_text, encoding="utf-8")
    return f"OK: replaced 1 occurrence in {path}"
