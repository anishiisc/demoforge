# DemoForge

A multi-agent Python application that takes a natural-language description of an agentic AI use case and produces a complete, runnable mock demo plus supporting documentation. DemoForge calls the Anthropic Messages API directly; the generated demo never calls any real LLM.

---

## Architecture

```
User request
    |
    v
[Stage 0] Planner     -> plan.json
    |
    v
[Stage 1] Architect   -> demo_blueprint.md, production_spec.md, manifest.json
    |
    +-----------------------------+
    |                             |
    v                             v (if html_sim=true)
[Stage 2] Builder           [Parallel] Simulator -> demo_simulation.html
    |
    v
[Stage 3] Reviewer    -> review_report.md
    |
    v
[Stage 4] Runbook     -> RUN.md
    |
    +-------------+
                  v
[Stage 5] Documentation -> tech_note.md, MANIFEST.md
```

All generated artefacts live under `runs/<run_id>/`.

---

## Setup

### Prerequisites

- Python 3.11+
- An Anthropic API key (for live runs; not required for `--dry-run`)

### Install

```bash
git clone <repo>
cd demoforge
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### Model and pricing configuration

Edit `config/config.yaml`. The model IDs are read at runtime — update them when new models are released. **Fill in the `pricing_usd_per_mtok` values from the [Anthropic pricing page](https://www.anthropic.com/pricing)** to enable accurate cost tracking. The defaults are zero (no cost guard unless you set prices).

---

## Usage

### Start a new run

```bash
demoforge new "Build a demo of an agentic customer support triage system"
```

From a file:

```bash
demoforge new --file tests/fixtures/sample_request.txt
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--no-ask` | off | Skip Planner clarifying questions; record assumptions |
| `--dry-run` | off | Use fixture responses; no API calls, no API key needed |
| `--skip-sim` | off | Skip the HTML Simulator stage |
| `--budget N` | 25.0 | Hard stop at N USD spent |

### Dry-run (no API key needed)

```bash
demoforge new --dry-run --file tests/fixtures/sample_request.txt
```

### Resume a failed run

```bash
demoforge resume 20240115-143022-invoic
```

### Check run status

```bash
demoforge status 20240115-143022-invoic
```

### View logs

```bash
demoforge logs 20240115-143022-invoic --stage builder --type tool_call
```

### Streamlit UI

```bash
streamlit run src/demoforge/app.py
```

---

## Output

A completed run produces:

| File | Description |
|------|-------------|
| `plan.json` | Structured plan from Stage 0 |
| `demo_blueprint.md` | Full architecture contract |
| `production_spec.md` | Real-deployment specification |
| `manifest.json` | Ordered file list for the Builder |
| `demo/` | The complete generated demo project |
| `review_report.md` | Static and dynamic review findings |
| `RUN.md` | Presenter runbook with verified commands |
| `demo_simulation.html` | Self-contained browser simulation (if feasible) |
| `tech_note.md` | Technical documentation |
| `MANIFEST.md` | One-line artefact index |
| `orchestration_log.jsonl` | Every LLM call, tool call, and event |

---

## Testing

```bash
pytest
```

The test suite does not require an API key. All tests run against the dry-run client or local fixtures.

---

## Cost Notes

- Set `pricing_usd_per_mtok` in `config/config.yaml` to enable accurate USD tracking.
- Use `--budget` to set a hard spending cap.
- `--dry-run` makes zero API calls and costs nothing.
- A typical live run on the sample request uses roughly 100k–300k tokens across all stages. At current (late 2024) Opus pricing this is approximately $5–$15.

---

## Key design decisions

See `CLAUDE.md` for a full list. Key points:

- No LangChain / CrewAI — the agent loop is written directly against the Messages API for full auditability.
- Prompt caching is applied on the system prompt and tool definitions to reduce cost on multi-turn agents.
- The generated demo **never** calls a real LLM; all responses are served from `mock_llm_instructions.json`.

---

## Limitations

- The dry-run fixtures produce a structurally complete but content-minimal demo. A live run is needed for a realistic, domain-specific demo.
- The HTML Simulator is a scripted replay, not a live execution; it embeds hardcoded synthetic data.
- Pricing values default to zero; cost tracking is only meaningful after `config.yaml` is updated.
- `run_command` sandboxing uses an allowlist but does not use a full container sandbox; do not run untrusted code.

---

## References

1. Anthropic. *Tool use with Claude*. https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
2. Anthropic. *Prompt caching*. https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
3. Anthropic. *Models overview*. https://docs.anthropic.com/en/docs/about-claude/models/overview
4. Anthropic. *anthropic-sdk-python*. https://github.com/anthropics/anthropic-sdk-python
5. Schluntz, E. and Zhang, B. (2024). *Building effective agents*. https://www.anthropic.com/engineering/building-effective-agents
