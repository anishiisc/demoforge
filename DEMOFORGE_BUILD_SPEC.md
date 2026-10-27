# DemoForge: Build Specification for Claude Code

> Purpose of this file: instructions for Claude Code to build **DemoForge**, a multi-agent system powered by the Anthropic API. DemoForge takes a natural-language description of an agentic AI use case and produces a complete, runnable mock demo, an optional standalone HTML simulation, and supporting documentation.
>
> Claude Code: read this whole file before writing any code. Follow the milestones in Section 14 in order. Do not skip the tests.

---

## 1. Objective

Build a Python application that:

1. Accepts a free-text use case description from a user (CLI and a minimal Streamlit front end).
2. Runs a staged pipeline of specialised agents, each backed by Anthropic Messages API calls with tool use.
3. Writes all generated artefacts into an isolated run workspace.
4. Logs every LLM call, tool call, agent handoff and error for post-run audit.
5. Supports resume after failure, cost tracking and a dry-run mode with no API calls.

The **output** of DemoForge is a mock demo project. The mock demo itself must never call a real LLM; it reads canned responses from `mock_llm_instructions.json`. Only DemoForge calls Anthropic.

---

## 2. Pipeline Overview

```
User request
    |
    v
[Stage 0] Planner Agent  -> plan.json (incl. HTML_SIM decision)
    |
    v
[Stage 1] Architect Agent -> demo_blueprint.md, production_spec.md, manifest.json
    |
    +-------------------------------+
    |                               |
    v                               v (only if HTML_SIM = YES)
[Stage 2] Builder Agent        [Parallel] Simulator Agent
    |                               -> demo_simulation.html
    v                               |
[Stage 3] Reviewer Agent            |
    -> review_report.md             |
    |                               |
    v                               |
[Stage 4] Runbook Agent             |
    -> RUN.md                       |
    |                               |
    +---------------+---------------+
                    v
[Stage 5] Documentation Agent -> tech_note.md, MANIFEST.md
```

Stage 5 must wait for both the main chain and the Simulator (if running) to finish.

---

## 3. Technology Stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| LLM access | `anthropic` official Python SDK (async client) |
| Data models | `pydantic` v2 |
| CLI | `typer` |
| Front end | `streamlit` (thin wrapper over the same orchestrator) |
| Config | `pyyaml` + `.env` via `python-dotenv` |
| Concurrency | `asyncio` |
| Templating for prompts | `jinja2` |
| Tests | `pytest`, `pytest-asyncio` |
| Lint/format | `ruff` |

Do not use LangChain, LangGraph, CrewAI or similar frameworks. The agent loop is written directly against the Messages API so every call is visible and auditable.

---

## 4. Repository Layout

```
demoforge/
  CLAUDE.md                     # conventions for future Claude Code sessions
  README.md
  pyproject.toml
  .env.example                  # ANTHROPIC_API_KEY=
  config/
    config.yaml                 # models, budgets, limits per agent
  prompts/
    planner.md.j2
    architect.md.j2
    builder.md.j2
    reviewer.md.j2
    runbook.md.j2
    simulator.md.j2
    documentation.md.j2
    shared_rules.md             # injected into every system prompt
  schemas/
    plan.schema.json
    manifest.schema.json
  src/demoforge/
    __init__.py
    cli.py                      # typer entrypoint
    app.py                      # streamlit entrypoint
    config.py                   # load + validate config
    orchestrator.py             # stage sequencing, parallelism, resume
    state.py                    # RunState model, persistence to run_state.json
    llm/
      client.py                 # wrapper over AsyncAnthropic, retries, logging
      agent_loop.py             # generic tool-use loop
      dry_run.py                # fake client for tests and --dry-run
      cost.py                   # token to cost accounting
    agents/
      base.py                   # BaseAgent: prompt render, tools, loop call
      planner.py
      architect.py
      builder.py
      reviewer.py
      runbook.py
      simulator.py
      documentation.py
    tools/
      registry.py               # tool definitions + dispatch
      fs_tools.py               # read_file, write_file, list_files, edit_file
      exec_tools.py             # run_command (sandboxed)
      control_tools.py          # submit_plan, submit_manifest, finish_stage, ask_user
    audit/
      logger.py                 # JSONL audit writer
      redact.py                 # strip secrets from logged payloads
    workspace.py                # run directory creation, path guards
  tests/
    test_agent_loop.py
    test_tools_sandbox.py
    test_orchestrator_dry_run.py
    test_manifest_schema.py
    fixtures/
      sample_request.txt
      dry_run_responses/        # canned API responses per stage
  runs/                         # created at runtime, gitignored
```

---

## 5. Configuration (`config/config.yaml`)

```yaml
anthropic:
  max_retries: 4
  timeout_seconds: 600

models:
  planner:       claude-opus-5-5
  architect:     claude-opus-5-5
  builder:       claude-sonnet-5-5
  reviewer:      claude-opus-5-5
  runbook:       claude-haiku-5-5
  simulator:     claude-sonnet-5-5
  documentation: claude-sonnet-5-5

limits:
  max_turns_per_agent:      40
  max_tokens_per_response:  16000
  reviewer_max_fix_cycles:  3
  run_budget_usd:           25.0    # hard stop when exceeded
  command_timeout_seconds:  300

execution:
  allow_pip_install: true            # reviewer may create a venv and install requirements
  allowed_commands: ["python", "pip", "pytest", "ruff", "ls", "cat"]
  network_in_generated_code: false   # reviewer flags any HTTP client usage to external hosts

pricing_usd_per_mtok:                # keep editable; used only for the budget guard
  claude-opus-5-5:   {input: 0.0, output: 0.0}
  claude-sonnet-5-5: {input: 0.0, output: 0.0}
  claude-haiku-5-5:  {input: 0.0, output: 0.0}
```

Model IDs must be read from config, never hard-coded in agent files. Leave the pricing values as zero placeholders and document in README that the operator fills them from the current Anthropic pricing page.

---

## 6. Core Components

### 6.1 LLM client wrapper (`llm/client.py`)

- Wraps `anthropic.AsyncAnthropic`.
- Single method: `async def create(self, *, agent: str, model: str, system, messages, tools, tool_choice=None, max_tokens) -> Message`.
- Applies prompt caching on the system prompt and tool definitions using `cache_control: {"type": "ephemeral"}` on the last system block, since agents make many turns with an identical prefix.
- Retries on rate limit and overload errors with exponential backoff and jitter (SDK retries plus one outer layer that logs each attempt).
- After every call, writes an audit record (Section 8) and updates the run cost tally. Raises `BudgetExceeded` if `run_budget_usd` is crossed.

### 6.2 Generic agent loop (`llm/agent_loop.py`)

```
messages = [initial user message]
for turn in range(max_turns):
    resp = await client.create(...)
    append assistant content to messages
    if resp.stop_reason == "end_turn": break
    if resp.stop_reason == "max_tokens": append "continue" user turn, continue
    if resp.stop_reason == "tool_use":
        for each tool_use block:
            result = await registry.dispatch(name, input, ctx)
            collect tool_result (is_error=True on exception, with message)
        append single user message containing all tool_result blocks
    if a control tool `finish_stage` was called: break
return AgentResult(final_text, tool_calls, turns, usage)
```

Requirements:

- All `tool_result` blocks for a turn go back in one user message, in the same order as the `tool_use` blocks.
- Tool exceptions never crash the loop; they are returned as `is_error: true` so the model can recover.
- Exceeding `max_turns` raises `AgentStalled`, which the orchestrator records and surfaces.
- Long conversations: when the message list exceeds a configurable character threshold, keep the first user message and the most recent N turns, and insert a short summary of the dropped middle turns produced by a Haiku call. Log the compaction event.

### 6.3 Tool registry (`tools/registry.py`)

Each tool is declared once with: name, description, JSON input schema, handler, and the list of agents allowed to use it. Agents only receive the tools they are permitted.

| Tool | Purpose | Agents |
|---|---|---|
| `read_file(path)` | Read a file inside the run workspace | all except Planner |
| `list_files(subdir)` | Tree listing of workspace | all except Planner |
| `write_file(path, content)` | Create or overwrite a file | Architect, Builder, Reviewer, Runbook, Simulator, Documentation |
| `edit_file(path, old, new)` | Exact string replacement | Builder, Reviewer |
| `run_command(cmd, cwd)` | Run an allowlisted command in the demo project with timeout | Builder, Reviewer, Runbook |
| `submit_plan(plan)` | Structured Stage 0 output, validated against `plan.schema.json` | Planner |
| `submit_manifest(manifest)` | Structured file manifest, validated against `manifest.schema.json` | Architect |
| `ask_user(question)` | Pause run for clarification (CLI prompt or Streamlit input) | Planner only |
| `finish_stage(summary)` | Signals stage completion with a short summary | all |

### 6.4 Sandboxing (`workspace.py`, `exec_tools.py`)

- Every path argument is resolved and must sit inside `runs/<run_id>/demo/` (or `runs/<run_id>/` for docs). Reject `..`, absolute paths outside, and symlinks escaping the root.
- `run_command` uses `subprocess` with `shell=False`, the configured allowlist on the first token, a timeout, and a scrubbed environment (no `ANTHROPIC_API_KEY` passed to child processes).
- If `allow_pip_install` is true, the Reviewer creates `runs/<run_id>/demo/.venv` and installs `requirements.txt` there only.
- Capture stdout and stderr, truncate to a configurable length before returning to the model, and log the full output to the audit trail.

---

## 7. Agent Specifications

Each agent has: a Jinja2 system prompt template in `prompts/`, the shared rules block, a model from config, a tool allowlist, a defined input and a defined output. The initial user message to each agent is assembled by the orchestrator from earlier stage outputs, not from the raw request alone (except the Planner).

### 7.0 Shared rules (`prompts/shared_rules.md`)

Injected into every agent:

- You are one stage in a pipeline. Do only your stage's job.
- Write all files with tools; never paste file contents in plain text as a substitute.
- The demo being built must make no real LLM or external network calls. Every LLM interaction in the demo is served from `mock_llm_instructions.json`.
- Every LLM call and tool call in the demo must be logged by the demo's audit module.
- No secrets or API keys in generated code.
- Call `finish_stage` with a summary of at most 120 words when done.

### 7.1 Planner Agent (Stage 0)

- **Input:** raw user request.
- **Job:** extract domain, persona, core workflow steps, agents and tools the demo needs, data entities, success criteria, assumptions. Ask at most 3 clarifying questions via `ask_user`, and only for details that would materially change the design. In non-interactive mode (`--no-ask`), record assumptions instead.
- **Feasibility gate:** decide `html_sim` (true/false) with a reason. True when the workflow can be scripted deterministically and the UI fits a single page.
- **Output:** `plan.json` via `submit_plan` (forced with `tool_choice: {"type": "tool", "name": "submit_plan"}` on the final turn).

`plan.schema.json` fields: `title`, `domain`, `persona`, `workflow_steps[]`, `demo_agents[] {name, role, inputs, outputs}`, `demo_tools[] {name, purpose}`, `data_entities[] {name, fields[], approx_rows}`, `success_criteria[]`, `assumptions[]`, `html_sim {feasible: bool, reason: str}`, `gui_framework` (default `streamlit`).

### 7.2 Architect Agent (Stage 1)

- **Input:** `plan.json`.
- **Output files:**
  1. `demo_blueprint.md` with sections: Use Case Summary; Assumptions; Agent and Tool Architecture (roles, inputs, outputs, call sequence); Synthetic Data Spec (schemas, row counts, distributions, deliberate edge cases); Mock GUI Spec (screens, components, user actions); Mocked LLM Layer (structure of `mock_llm_instructions.json`, keyed by `agent_name` then `step_id`, with selection rules for variant responses); Audit Logging Spec (JSONL, fields listed in Section 8.2); Folder Structure; Acceptance Checklist (numbered, testable items).
  2. `production_spec.md`: functional requirements, non-functional requirements, real data sources and integrations, model and framework choices with rationale, security and compliance, observability, evaluation strategy, scalability, indicative cost drivers, phased rollout.
  3. `manifest.json` via `submit_manifest`: ordered list of files the Builder must create, each with `path`, `purpose`, `depends_on[]`, `blueprint_section`.
- **Rule:** the blueprint is the contract. Downstream agents treat it as the source of truth.

### 7.3 Builder Agent (Stage 2)

- **Input:** `demo_blueprint.md`, `manifest.json`.
- **Execution pattern:** the orchestrator iterates the manifest in dependency order and runs one Builder loop per file group (default group size 1 to 3 files) so each call has a focused context. Each loop receives: the blueprint, the manifest, the list of files already written, and the target files for this group.
- **Required build order:**
  1. `data/generate_synthetic_data.py` and run it to produce the data files
  2. `mock_llm_instructions.json`
  3. `audit/logger.py` (demo-side audit module)
  4. Tool modules
  5. Agent modules and orchestration
  6. GUI
  7. `requirements.txt`
  8. `tests/test_smoke.py` covering one end-to-end path with mocked responses
- **Output:** the demo project under `runs/<run_id>/demo/`.

### 7.4 Reviewer Agent (Stage 3)

- **Input:** blueprint, manifest, full demo codebase.
- **Job, per cycle (max `reviewer_max_fix_cycles`):**
  1. Static pass: imports resolve, every manifest file exists, no external HTTP calls, no hard-coded keys, every LLM call path goes through the mock layer, every tool and LLM call is logged.
  2. Dynamic pass: create venv, install requirements, run the data generator, run `pytest`, run a headless import of the GUI module.
  3. Check each Acceptance Checklist item; mark pass/fail with evidence.
  4. Fix failures with `edit_file` / `write_file`, then re-run.
- **Output:** `review_report.md` with a table of checklist items (status, evidence), issues found, fixes applied, and anything left unresolved.

### 7.5 Runbook Agent (Stage 4)

- **Input:** blueprint, review report, file tree.
- **Output:** `RUN.md` covering prerequisites, environment setup, data generation, launch command, a numbered click-by-click demo script for a presenter, where to find audit logs and how to read them, troubleshooting for the issues the Reviewer saw.
- The agent must verify each command in `RUN.md` by executing it with `run_command` where possible.

### 7.6 Simulator Agent (parallel, only if `html_sim.feasible`)

- **Starts:** immediately after Stage 1 completes, as a concurrent `asyncio` task.
- **Input:** blueprint, plan.
- **Output:** `demo_simulation.html`, a single self-contained file (inline CSS and JS, no CDN dependencies) that replays the workflow: user input, each agent step, tool calls with arguments, outputs, and a live-updating audit log panel. Include play, pause, step-forward and reset controls. Embed a small synthetic sample generated from the blueprint's data spec so the file does not depend on Stage 2.
- **Validation:** after writing, the orchestrator checks the file parses as HTML and contains no external `src`/`href` URLs.

### 7.7 Documentation Agent (Stage 5)

- **Input:** all artefacts plus the orchestration log summary.
- **Output:**
  1. `tech_note.md`: purpose; architecture; one subsection per artefact explaining its role; how the mock layer maps to the production design in `production_spec.md`; limitations; next steps; references.
  2. `MANIFEST.md`: one line per artefact with path and description.
- **Style rules for this agent:** formal register, no em dashes, no filler transitions, references section with real citations only.

---

## 8. Audit Logging

Two separate logs. Do not confuse them.

### 8.1 DemoForge log (the builder system)

File: `runs/<run_id>/orchestration_log.jsonl`. One JSON object per line.

| Event type | Fields |
|---|---|
| `llm_call` | ts, run_id, stage, agent, model, turn, request_message_count, stop_reason, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, latency_ms, cost_usd, request_id |
| `tool_call` | ts, run_id, stage, agent, tool, input (redacted), is_error, output_preview, duration_ms |
| `stage_start` / `stage_end` | ts, run_id, stage, agent, status, summary |
| `handoff` | ts, run_id, from_stage, to_stage, artefacts[] |
| `error` | ts, run_id, stage, agent, error_type, message, traceback |
| `compaction` | ts, run_id, agent, dropped_turns, summary_tokens |
| `budget` | ts, run_id, spent_usd, limit_usd |

Full request and response bodies go to `runs/<run_id>/llm_payloads/<seq>_<agent>.json` so the JSONL stays scannable. `redact.py` strips anything matching key patterns before writing.

### 8.2 Demo log (inside the generated demo)

The Architect must specify, and the Reviewer must verify, that the generated demo writes `logs/audit.jsonl` with: ts, run_id, agent, event_type (`llm_call` | `tool_call` | `handoff` | `error`), step_id, input, output, latency_ms, mock_source_key.

---

## 9. Run Workspace Layout

```
runs/<run_id>/
  request.txt
  run_state.json
  orchestration_log.jsonl
  llm_payloads/
  plan.json
  demo_blueprint.md
  production_spec.md
  manifest.json
  review_report.md
  RUN.md
  demo_simulation.html          # if html_sim
  tech_note.md
  MANIFEST.md
  demo/
    data/
    agents/
    tools/
    audit/
    gui/
    logs/
    tests/
    mock_llm_instructions.json
    requirements.txt
```

`run_id` format: `YYYYMMDD-HHMMSS-<6 char slug of title>`.

---

## 10. Orchestrator Behaviour (`orchestrator.py`)

- Persists `RunState` (stage statuses, artefact paths, cost so far, errors) to `run_state.json` after every stage.
- `demoforge resume <run_id>` restarts from the first stage not marked `done`.
- Starts the Simulator task with `asyncio.create_task` after Stage 1 if feasible; awaits it before Stage 5. A Simulator failure is logged and reported but does not fail the run.
- On `BudgetExceeded` or `AgentStalled`, marks the stage `failed`, writes state, and exits with a clear message.
- Emits progress events (stage started/finished, files written) through a callback so both CLI (rich console output) and Streamlit (status panel) can display them.

---

## 11. Interfaces

### 11.1 CLI

```
demoforge new "<request text>" [--no-ask] [--dry-run] [--skip-sim] [--budget 25]
demoforge new --file request.txt
demoforge resume <run_id>
demoforge status <run_id>
demoforge logs <run_id> [--stage builder] [--type llm_call]
```

### 11.2 Streamlit

Single page: request text area, options (no-ask, skip-sim, budget), Run button, live stage progress, clarification prompt box when the Planner asks, artefact browser with download buttons, a cost meter, and a log viewer filterable by stage and event type.

---

## 12. Dry-Run Mode

`--dry-run` swaps `llm/client.py` for `llm/dry_run.py`, which returns canned `Message`-shaped objects from `tests/fixtures/dry_run_responses/<stage>/<turn>.json`. Tool calls in those fixtures are executed for real against the workspace. This lets the whole pipeline be tested end to end with no API key and no cost. Ship fixtures that produce a small but complete demo for the sample request.

---

## 13. Error Handling and Guardrails

- API errors: SDK retries, then outer retry with backoff, then stage failure.
- Invalid structured output (`submit_plan`, `submit_manifest` failing schema validation): return the validation error as a tool error so the model can correct it; fail after 3 attempts.
- File writes larger than a configured size limit are rejected with an error message telling the agent to split the file.
- The Reviewer must reject generated code that imports `anthropic`, `openai`, `requests` to external hosts, or reads API key environment variables.
- Never log the API key. Never pass it to subprocesses.

---

## 14. Implementation Milestones for Claude Code

Complete each milestone, run its tests, then move on.

1. **Scaffold:** `pyproject.toml`, package layout, config loading with pydantic validation, `CLAUDE.md`, `.env.example`, ruff config.
2. **Audit + workspace:** JSONL logger, redaction, path guards. Tests for path traversal rejection.
3. **LLM client + dry-run client:** wrapper with caching, retries, cost accounting, budget guard. Unit tests with the dry-run client.
4. **Agent loop + tool registry:** generic loop, all tools, sandboxed `run_command`. Tests: multi-tool turn, tool error recovery, max-turn stall, compaction trigger.
5. **Planner + schemas:** prompt, `submit_plan`, `ask_user` (CLI and no-ask modes).
6. **Architect:** prompt, three outputs, manifest schema validation.
7. **Builder:** manifest-driven grouped loops.
8. **Reviewer:** static and dynamic passes, fix cycles, report.
9. **Runbook, Simulator, Documentation agents.**
10. **Orchestrator:** sequencing, parallel simulator, state, resume, progress callbacks.
11. **CLI and Streamlit front end.**
12. **Dry-run fixtures + end-to-end test** using `tests/fixtures/sample_request.txt`.
13. **README:** setup, configuration, usage, cost notes, architecture diagram, limitations.

---

## 15. Acceptance Criteria for DemoForge

- [ ] `demoforge new --dry-run --file tests/fixtures/sample_request.txt` completes with all artefacts present and `pytest` passing.
- [ ] A live run on the sample request produces a demo whose own smoke test passes and whose GUI launches with the command in `RUN.md`.
- [ ] Every Anthropic call appears in `orchestration_log.jsonl` with token counts and cost.
- [ ] Every tool call appears in the log with redacted inputs.
- [ ] The generated demo contains no real LLM or network calls and logs every mocked LLM call and tool call to `logs/audit.jsonl`.
- [ ] `demo_simulation.html` opens offline in a browser and steps through the workflow (when feasible).
- [ ] Killing the process mid-run and calling `resume` continues from the failed stage.
- [ ] The budget guard stops a run when the limit is set below the expected cost.
- [ ] No file is written outside `runs/<run_id>/`.

---

## 16. Sample Request for Testing (`tests/fixtures/sample_request.txt`)

```
Build a demo of an agentic invoice reconciliation assistant for a mid-size
FMCG distributor. An intake agent reads uploaded invoices and purchase orders,
a matching agent compares line items, quantities and prices, an exception agent
flags mismatches above 2 percent or missing POs, and a summary agent drafts an
email to the AP manager. The user should be able to upload a batch, see each
agent's reasoning, approve or reject exceptions, and download a reconciliation
report.
```

---

## 17. Notes for Claude Code

- Keep agent prompts in `prompts/`, not in Python strings, so they can be tuned without code changes.
- Prefer small, typed functions. Every public function has a docstring.
- Write tests alongside each milestone; do not batch them at the end.
- When a design choice here is ambiguous, pick the simpler option and record it in `CLAUDE.md` under "Decisions".

---

## References

1. Anthropic. *Tool use with Claude*. Claude Platform documentation. https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview
2. Anthropic. *Prompt caching*. Claude Platform documentation. https://docs.claude.com/en/docs/build-with-claude/prompt-caching
3. Anthropic. *Models overview*. Claude Platform documentation. https://docs.claude.com/en/docs/about-claude/models/overview
4. Anthropic. *anthropic-sdk-python*. GitHub repository. https://github.com/anthropics/anthropic-sdk-python
5. Schluntz, E. and Zhang, B. (2024). *Building effective agents*. Anthropic Engineering. https://www.anthropic.com/engineering/building-effective-agents
6. JSON Schema. *JSON Schema Specification, Draft 2020-12*. https://json-schema.org/specification
