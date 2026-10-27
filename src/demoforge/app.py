"""DemoForge Streamlit front end."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from pathlib import Path
from typing import Any

import streamlit as st


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="DemoForge", layout="wide")
    st.title("DemoForge")
    st.caption("Generate complete mock AI demos from natural-language descriptions.")

    # ---- Sidebar / options ----
    with st.sidebar:
        st.header("Options")
        no_ask = st.checkbox("No-ask mode (record assumptions)", value=False)
        skip_sim = st.checkbox("Skip HTML Simulator", value=False)
        budget = st.number_input("Budget (USD)", min_value=0.1, value=25.0, step=1.0)
        dry_run = st.checkbox("Dry-run mode (fixtures, no API calls)", value=False)

    # ---- Main input ----
    request_text = st.text_area(
        "Use case description",
        height=150,
        placeholder="Describe the agentic AI demo you want built...",
    )

    run_btn = st.button("Run DemoForge", type="primary", disabled=not request_text.strip())

    # ---- Clarification box (shown when Planner calls ask_user) ----
    if "pending_question" in st.session_state and st.session_state.pending_question:
        q = st.session_state.pending_question
        st.info(f"**Planner asks:** {q}")
        answer = st.text_input("Your answer", key="clarification_answer")
        if st.button("Submit answer"):
            cb = st.session_state.get("clarification_callback")
            if cb:
                cb(answer)
            st.session_state.pending_question = None
            st.session_state.clarification_callback = None
            st.rerun()

    if run_btn:
        _run_pipeline(
            request_text=request_text.strip(),
            no_ask=no_ask,
            skip_sim=skip_sim,
            budget=budget,
            dry_run=dry_run,
        )


def _run_pipeline(
    *,
    request_text: str,
    no_ask: bool,
    skip_sim: bool,
    budget: float,
    dry_run: bool,
) -> None:
    """Launch the orchestrator and display live progress."""
    from demoforge.config import get_config
    from demoforge.orchestrator import create_orchestrator

    config = get_config()

    # Progress event queue
    event_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    # ask_user support
    answer_queue: queue.Queue[str] = queue.Queue()

    async def ask_user_cb(question: str) -> str:
        st.session_state.pending_question = question
        st.session_state.clarification_callback = lambda a: answer_queue.put(a)
        # Block until answered
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, answer_queue.get)

    ask_cb = None if no_ask else ask_user_cb

    def sync_progress(event: str, stage: str, detail: dict) -> None:
        event_queue.put({"event": event, "stage": stage, "detail": detail})

    orchestrator, state = create_orchestrator(
        request=request_text,
        config=config,
        dry_run=dry_run,
        no_ask=no_ask,
        skip_sim=skip_sim,
        budget=budget,
        progress_cb=sync_progress,
        ask_user_cb=ask_cb,
    )

    st.write(f"**Run ID:** `{state.run_id}`")

    # Progress display
    progress_placeholder = st.empty()
    cost_placeholder = st.empty()
    log_placeholder = st.expander("Event log", expanded=False)

    stage_statuses: dict[str, str] = {}
    log_lines: list[str] = []

    # Run in a thread so Streamlit can update the UI
    result_holder: list[Any] = []
    error_holder: list[Exception] = []

    def _thread_target() -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(orchestrator.run(request_text))
            result_holder.append(result)
        except Exception as exc:
            error_holder.append(exc)
        finally:
            event_queue.put({"event": "__done__", "stage": "", "detail": {}})

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()

    # Poll for events
    while True:
        try:
            item = event_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if item["event"] == "__done__":
            break

        ev = item["event"]
        stage = item["stage"]
        detail = item["detail"]
        log_lines.append(f"[{ev}] {stage}: {detail}")

        if ev == "stage_started":
            stage_statuses[stage] = "running"
        elif ev == "stage_done":
            stage_statuses[stage] = "done"
        elif ev == "stage_failed":
            stage_statuses[stage] = "failed"

        # Render progress
        with progress_placeholder.container():
            cols = st.columns(len(orchestrator.STAGE_NAMES))
            for i, sn in enumerate(orchestrator.STAGE_NAMES):
                status = stage_statuses.get(sn, "pending")
                icon = {"done": "✅", "running": "⏳", "failed": "❌", "pending": "⬜"}.get(status, "⬜")
                cols[i].metric(sn, icon)

        with log_placeholder:
            st.text("\n".join(log_lines[-20:]))

    t.join()

    if error_holder:
        st.error(f"Run failed: {error_holder[0]}")
        return

    if result_holder:
        final = result_holder[0]
        cost_placeholder.metric("Total cost", f"${final.cost_usd:.4f}")
        st.success(f"Run complete. Artefacts are in `runs/{final.run_id}/`")
        _show_artefacts(final)


def _show_artefacts(state: Any) -> None:
    """Show download buttons for produced artefacts."""
    from demoforge.workspace import Workspace
    ws = Workspace(state.run_id)

    st.subheader("Artefacts")
    artefact_paths = [
        ws.plan_path,
        ws.blueprint_path,
        ws.production_spec_path,
        ws.manifest_path,
        ws.review_report_path,
        ws.runbook_path,
        ws.tech_note_path,
        ws.manifest_doc_path,
        ws.simulation_path,
    ]
    for path in artefact_paths:
        if path.exists():
            label = path.name
            data = path.read_bytes()
            st.download_button(label=f"Download {label}", data=data, file_name=label)


if __name__ == "__main__":
    main()
