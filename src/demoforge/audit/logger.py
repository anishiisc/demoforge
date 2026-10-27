"""JSONL audit logger for DemoForge orchestration events."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from .redact import redact, redact_env_key

# Maximum number of characters to include in the output_preview field of a tool_call record.
_PREVIEW_MAX = 500


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class AuditLogger:
    """Thread-safe JSONL writer for orchestration events.

    One instance per run; write a JSON object per line to *log_path*.
    Full LLM request/response bodies are written to *payloads_dir*.
    """

    def __init__(self, log_path: Path, payloads_dir: Path, run_id: str) -> None:
        self._log_path = log_path
        self._payloads_dir = payloads_dir
        self._run_id = run_id
        self._lock = threading.Lock()
        self._seq = 0

        log_path.parent.mkdir(parents=True, exist_ok=True)
        payloads_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, record: dict[str, Any]) -> None:
        record.setdefault("ts", _now_iso())
        record.setdefault("run_id", self._run_id)
        line = json.dumps(record, default=str)
        line = redact_env_key(line)
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _write_payload(self, seq: int, agent: str, payload: Any) -> None:
        """Write a full LLM payload to the payloads directory."""
        fname = self._payloads_dir / f"{seq:04d}_{agent}.json"
        safe = redact(payload)
        with self._lock:
            fname.write_text(json.dumps(safe, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public event writers
    # ------------------------------------------------------------------

    def llm_call(
        self,
        *,
        stage: str,
        agent: str,
        model: str,
        turn: int,
        request_message_count: int,
        stop_reason: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_creation_tokens: int,
        latency_ms: float,
        cost_usd: float,
        request_id: str,
        request_payload: Any = None,
        response_payload: Any = None,
    ) -> None:
        """Record a completed LLM call."""
        seq = self._next_seq()
        if request_payload is not None or response_payload is not None:
            self._write_payload(
                seq,
                agent,
                {"request": request_payload, "response": response_payload},
            )
        self._write(
            {
                "event": "llm_call",
                "stage": stage,
                "agent": agent,
                "model": model,
                "turn": turn,
                "request_message_count": request_message_count,
                "stop_reason": stop_reason,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_creation_tokens": cache_creation_tokens,
                "latency_ms": round(latency_ms, 1),
                "cost_usd": cost_usd,
                "request_id": request_id,
                "payload_seq": seq,
            }
        )

    def tool_call(
        self,
        *,
        stage: str,
        agent: str,
        tool: str,
        input_data: Any,
        is_error: bool,
        output: str,
        duration_ms: float,
    ) -> None:
        """Record a tool invocation."""
        self._write(
            {
                "event": "tool_call",
                "stage": stage,
                "agent": agent,
                "tool": tool,
                "input": redact(input_data),
                "is_error": is_error,
                "output_preview": output[:_PREVIEW_MAX],
                "duration_ms": round(duration_ms, 1),
            }
        )

    def stage_start(self, *, stage: str, agent: str) -> None:
        self._write({"event": "stage_start", "stage": stage, "agent": agent, "status": "started"})

    def stage_end(self, *, stage: str, agent: str, status: str, summary: str) -> None:
        self._write(
            {
                "event": "stage_end",
                "stage": stage,
                "agent": agent,
                "status": status,
                "summary": summary,
            }
        )

    def handoff(self, *, from_stage: str, to_stage: str, artefacts: list[str]) -> None:
        self._write(
            {
                "event": "handoff",
                "from_stage": from_stage,
                "to_stage": to_stage,
                "artefacts": artefacts,
            }
        )

    def error(
        self,
        *,
        stage: str,
        agent: str,
        error_type: str,
        message: str,
        traceback: str = "",
    ) -> None:
        self._write(
            {
                "event": "error",
                "stage": stage,
                "agent": agent,
                "error_type": error_type,
                "message": message,
                "traceback": traceback,
            }
        )

    def compaction(
        self, *, agent: str, dropped_turns: int, summary_tokens: int
    ) -> None:
        self._write(
            {
                "event": "compaction",
                "agent": agent,
                "dropped_turns": dropped_turns,
                "summary_tokens": summary_tokens,
            }
        )

    def budget(self, *, spent_usd: float, limit_usd: float) -> None:
        self._write({"event": "budget", "spent_usd": spent_usd, "limit_usd": limit_usd})
