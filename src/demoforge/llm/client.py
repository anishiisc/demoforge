"""Async Anthropic client wrapper with caching, retries, cost accounting, and audit logging."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import anthropic

from demoforge.audit.logger import AuditLogger
from demoforge.config import Config
from demoforge.llm.cost import BudgetExceeded, CostTracker, UsageRecord


class LLMClient:
    """Wraps :class:`anthropic.AsyncAnthropic` with DemoForge-specific concerns.

    - Applies prompt caching on the last system block and tool definitions.
    - Retries on rate-limit and overload errors (SDK retries + one outer layer).
    - After every call, writes an audit record and updates the cost tally.
    - Raises :class:`BudgetExceeded` when the run budget is exceeded.
    """

    def __init__(
        self,
        config: Config,
        cost_tracker: CostTracker,
        audit: AuditLogger,
        stage: str,
    ) -> None:
        self._config = config
        self._cost_tracker = cost_tracker
        self._audit = audit
        self._stage = stage
        self._client = anthropic.AsyncAnthropic(
            api_key=config.api_key,
            max_retries=config.anthropic.max_retries,
            timeout=config.anthropic.timeout_seconds,
        )
        self._turn: int = 0

    async def create(
        self,
        *,
        agent: str,
        model: str,
        system: list[dict[str, Any]] | str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> anthropic.types.Message:
        """Send one Messages API call, handle retries, log, and return the Message.

        Applies ``cache_control: {"type": "ephemeral"}`` to the last system block
        and to the tool definitions block so repeated turns with an identical prefix
        benefit from prompt caching.
        """
        if max_tokens is None:
            max_tokens = self._config.limits.max_tokens_per_response

        # SDK v2: system is a plain string or list of TextBlockParam dicts.
        # cache_control is a top-level parameter (no longer embedded in blocks).
        system_param = system if isinstance(system, str) else _build_system_blocks(system)

        self._turn += 1
        turn = self._turn

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_param,
            "messages": messages,
            # Enable prompt caching at the request level (SDK v2 top-level param)
            "cache_control": {"type": "ephemeral"},
        }
        # Explicitly disable extended thinking for Opus models — they enable it by
        # default and consume large token budgets on thinking blocks before any output.
        if "opus" in model:
            kwargs["thinking"] = {"type": "disabled"}
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        # One outer retry layer on top of the SDK's own retries.
        last_exc: Exception | None = None
        for attempt in range(3):
            if attempt > 0:
                wait = 2 ** attempt + (asyncio.get_event_loop().time() % 1)
                self._audit.error(
                    stage=self._stage,
                    agent=agent,
                    error_type="retry",
                    message=f"Outer retry attempt {attempt} after: {last_exc}",
                )
                await asyncio.sleep(wait)
            try:
                t0 = time.monotonic()
                response: anthropic.types.Message = await self._client.messages.create(
                    **kwargs,
                )
                latency_ms = (time.monotonic() - t0) * 1000
                break
            except (anthropic.RateLimitError, anthropic.InternalServerError) as exc:
                last_exc = exc
        else:
            raise last_exc  # type: ignore[misc]

        # Extract usage
        usage_obj = response.usage
        record = UsageRecord(
            input_tokens=getattr(usage_obj, "input_tokens", 0) or 0,
            output_tokens=getattr(usage_obj, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage_obj, "cache_read_input_tokens", None) or 0,
            cache_creation_tokens=getattr(usage_obj, "cache_creation_input_tokens", None) or 0,
        )

        # Update cost (raises BudgetExceeded if exceeded)
        cost_usd = 0.0
        try:
            total = self._cost_tracker.record(model, record)
            cost_usd = total  # approximate per-call cost via total delta
        except BudgetExceeded:
            self._audit.budget(
                spent_usd=self._cost_tracker.total_usd,
                limit_usd=self._config.limits.run_budget_usd,
            )
            raise

        # Audit log
        self._audit.llm_call(
            stage=self._stage,
            agent=agent,
            model=model,
            turn=turn,
            request_message_count=len(messages),
            stop_reason=response.stop_reason or "unknown",
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cache_read_tokens=record.cache_read_tokens,
            cache_creation_tokens=record.cache_creation_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            request_id=getattr(response, "id", ""),
            request_payload=kwargs,
            response_payload=response.model_dump() if hasattr(response, "model_dump") else None,
        )

        return response

    def for_stage(self, stage: str) -> "LLMClient":
        """Return a shallow copy of this client with a different stage label."""
        clone = object.__new__(LLMClient)
        clone._config = self._config
        clone._cost_tracker = self._cost_tracker
        clone._audit = self._audit
        clone._stage = stage
        clone._client = self._client
        clone._turn = 0
        return clone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_system_blocks(system: list[dict[str, Any]] | str) -> list[dict[str, Any]]:
    """Normalise *system* to a list of content blocks, adding cache control on the last."""
    if isinstance(system, str):
        blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
    else:
        blocks = list(system)
    if blocks:
        last = dict(blocks[-1])
        last["cache_control"] = {"type": "ephemeral"}
        blocks[-1] = last
    return blocks


