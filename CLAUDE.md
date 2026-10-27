# CLAUDE.md — DemoForge conventions for Claude Code sessions

## Project Overview

DemoForge is a multi-agent Python application that takes a natural-language description of
an agentic AI use case and produces a complete, runnable mock demo plus supporting
documentation. It calls Anthropic's Messages API directly; the generated demo never calls any
real LLM.

## Repository Layout

```
src/demoforge/     # main package
config/            # config.yaml (models, budgets, limits)
prompts/           # Jinja2 system-prompt templates (one per agent)
schemas/           # JSON Schema files for structured outputs
tests/             # pytest suite
runs/              # runtime output (gitignored)
```

## Key Conventions

- **Never hard-code model IDs** in agent files. Always read from `config.models.for_agent(name)`.
- **Prompt templates live in `prompts/`** as Jinja2 `.md.j2` files. Do not embed prompts in Python strings.
- **All file I/O goes through tools** (`write_file`, `edit_file`, etc.), never via direct agent text.
- **Path guards**: every file path must be validated by `workspace.guard_path()` before use.
- **Audit everything**: every LLM call and tool call must produce a JSONL record in `orchestration_log.jsonl`.
- **The demo must never make real LLM or network calls**; all LLM responses are read from `mock_llm_instructions.json`.
- **No secrets in logs**: `audit/redact.py` strips API keys and other secrets before writing.

## Decisions

- Grouped Builder loops: default group size = 1 (one file per loop) for clarity; can be overridden by setting `BUILDER_GROUP_SIZE` env var.
- Compaction uses `claude-haiku-5-5` for the summary call (cheapest available at time of decision).
- `ask_user` in `--no-ask` mode records a placeholder assumption string instead of blocking.
- `run_id` slug is derived from the first six non-whitespace, alphanumeric characters of the plan title (lower-cased).
- Simulator validation checks for external `src=` or `href=` URLs with a simple regex (not a full HTML parser).
- On `max_tokens` stop reason, the loop appends a literal `"continue"` user message.

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Dry-Run Mode

```bash
demoforge new --dry-run --file tests/fixtures/sample_request.txt
```

This swaps the real LLM client for the fixture-based `dry_run.py` client. No API key required.

## Adding a New Agent

1. Create `src/demoforge/agents/<name>.py` extending `BaseAgent`.
2. Add a Jinja2 prompt at `prompts/<name>.md.j2`.
3. Register the agent's tool allowlist in `tools/registry.py`.
4. Add the model key to `config/config.yaml` under `models:`.
5. Wire the stage into `orchestrator.py`.
