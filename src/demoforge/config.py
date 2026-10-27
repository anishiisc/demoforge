"""Load and validate DemoForge configuration using Pydantic v2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

load_dotenv()

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class AnthropicConfig(BaseModel):
    max_retries: int = 4
    timeout_seconds: int = 600


class ModelsConfig(BaseModel):
    planner: str = "claude-opus-5-5"
    architect: str = "claude-opus-5-5"
    builder: str = "claude-sonnet-5-5"
    reviewer: str = "claude-opus-5-5"
    runbook: str = "claude-haiku-5-5"
    simulator: str = "claude-sonnet-5-5"
    documentation: str = "claude-sonnet-5-5"

    def for_agent(self, agent: str) -> str:
        """Return the model ID for a named agent."""
        return getattr(self, agent)


class LimitsConfig(BaseModel):
    max_turns_per_agent: int = 40
    max_tokens_per_response: int = 16000
    reviewer_max_fix_cycles: int = 3
    run_budget_usd: float = 25.0
    command_timeout_seconds: int = 300
    max_file_size_bytes: int = 524288
    context_compaction_chars: int = 200000
    compaction_keep_turns: int = 6


class ExecutionConfig(BaseModel):
    allow_pip_install: bool = True
    allowed_commands: list[str] = Field(
        default_factory=lambda: ["python", "pip", "pytest", "ruff", "ls", "cat"]
    )
    network_in_generated_code: bool = False


class ModelPricing(BaseModel):
    input: float = 0.0
    output: float = 0.0


class Config(BaseModel):
    """Root configuration object for DemoForge."""

    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    pricing_usd_per_mtok: dict[str, ModelPricing] = Field(default_factory=dict)

    # Injected at load time, not from YAML
    api_key: str = Field(default="", exclude=True)

    @model_validator(mode="after")
    def _check_api_key(self) -> "Config":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        object.__setattr__(self, "api_key", key)
        return self

    def pricing_for(self, model: str) -> ModelPricing:
        """Return pricing for a model, falling back to zero-cost placeholder."""
        return self.pricing_usd_per_mtok.get(model, ModelPricing())


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"


def load_config(path: Path | None = None) -> Config:
    """Load and validate configuration from *path* (defaults to config/config.yaml)."""
    config_path = path or _CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
    return Config.model_validate(raw)


# Module-level singleton (lazy); agents import `get_config()`.
_config: Config | None = None


def get_config() -> Config:
    """Return the singleton Config, loading it on first call."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the singleton (used in tests)."""
    global _config
    _config = None
