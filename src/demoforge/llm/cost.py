"""Token-to-cost accounting for DemoForge LLM calls."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from demoforge.config import Config


@dataclass
class UsageRecord:
    """Cost and token counts for one LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0


class BudgetExceeded(Exception):
    """Raised when the cumulative run cost exceeds the configured limit."""

    def __init__(self, spent: float, limit: float) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(f"Budget exceeded: spent ${spent:.4f} of ${limit:.2f} limit")


class CostTracker:
    """Thread-safe accumulator for run cost and token counts."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._total_usd: float = 0.0
        self._total_input: int = 0
        self._total_output: int = 0
        self._total_cache_read: int = 0
        self._total_cache_creation: int = 0

    def record(self, model: str, usage: UsageRecord) -> float:
        """Add *usage* to the running tally and return the new total cost.

        Raises :class:`BudgetExceeded` if the new total exceeds the limit.
        """
        cost = _compute_cost(model, usage, self._config)
        with self._lock:
            self._total_usd += cost
            self._total_input += usage.input_tokens
            self._total_output += usage.output_tokens
            self._total_cache_read += usage.cache_read_tokens
            self._total_cache_creation += usage.cache_creation_tokens
            total = self._total_usd

        limit = self._config.limits.run_budget_usd
        if total > limit:
            raise BudgetExceeded(total, limit)
        return total

    @property
    def total_usd(self) -> float:
        with self._lock:
            return self._total_usd

    @property
    def summary(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "total_usd": self._total_usd,
                "total_input_tokens": self._total_input,
                "total_output_tokens": self._total_output,
                "total_cache_read_tokens": self._total_cache_read,
                "total_cache_creation_tokens": self._total_cache_creation,
            }


def _compute_cost(model: str, usage: UsageRecord, config: Config) -> float:
    """Return the USD cost for *usage* at *model* pricing."""
    pricing = config.pricing_for(model)
    # Pricing is per million tokens
    cost = (
        usage.input_tokens * pricing.input / 1_000_000
        + usage.output_tokens * pricing.output / 1_000_000
        # Cache reads are priced lower; cache creation is priced higher.
        # When pricing is 0 these are 0 anyway.
        + usage.cache_read_tokens * pricing.input * 0.1 / 1_000_000
        + usage.cache_creation_tokens * pricing.input * 1.25 / 1_000_000
    )
    return cost
