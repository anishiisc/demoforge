# Running DemoForge

## Prerequisites

- Python 3.12
- An Anthropic API key with credits
- Set `ANTHROPIC_API_KEY` in your environment (or a `.env` file in the project root)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Quickstart (foreground, live API)

Run directly in your terminal — **do not background this**; it prints live progress:

```bash
./run.sh new --file tests/fixtures/sample_request.txt --no-ask
```

Or equivalently:

```bash
.venv/bin/demoforge new --file tests/fixtures/sample_request.txt --no-ask
```

Output appears in your terminal as each stage completes. A typical run takes **3–8 minutes** depending on API latency.

---

## Dry-run (no API key required, instant)

Uses pre-recorded fixture responses — good for testing the pipeline without spending credits:

```bash
./run.sh new --dry-run --file tests/fixtures/sample_request.txt
```

---

## Writing your own request

Create a plain text file describing the agentic use case you want to demo:

```
Build a demo of an AI-powered customer support triage system.
An intake agent classifies incoming tickets by urgency and category,
a routing agent assigns them to the right team, and a response agent
drafts a suggested reply. Support agents should see the AI's reasoning
and be able to approve or edit before sending.
```

Then run:

```bash
./run.sh new --file my_request.txt --no-ask
```

---

## CLI reference

| Command | Description |
|---|---|
| `./run.sh new --file FILE` | Start a new run from a request file |
| `./run.sh new --file FILE --no-ask` | Skip clarification prompts (use assumptions) |
| `./run.sh new --dry-run --file FILE` | Dry-run using fixture responses, no API calls |
| `./run.sh resume RUN_ID` | Resume a failed or interrupted run |
| `./run.sh status RUN_ID` | Show stage statuses for a run |
| `./run.sh logs RUN_ID` | Tail the audit log for a run |

Run IDs look like `20261027-163913-build` and are printed at the start of each run.

---

## Output

All artefacts land in `runs/<run-id>/demo/`:

| File | Description |
|---|---|
| `plan.json` | Structured plan from the Planner agent |
| `manifest.json` | File manifest from the Architect |
| `demo_blueprint.md` | High-level demo design |
| `production_spec.md` | Technical specification |
| `*.py`, `*.html`, etc. | Generated demo code (Builder) |
| `review_report.md` | Code review findings (Reviewer) |
| `RUN.md` | How to run the generated demo |
| `tech_note.md` | Technical documentation |
| `MANIFEST.md` | File manifest documentation |
| `sim/index.html` | Standalone HTML simulation (if generated) |

---

## What the stages do & how long they take

| Stage | Model | Typical turns | ~Time |
|---|---|---|---|
| Planner | Sonnet | 2–4 | 20–60s |
| Architect | Sonnet | 4–8 | 1–2 min |
| Builder | Sonnet | 1 per file | 1–3 min |
| Reviewer | Sonnet | 2–5 | 30–90s |
| Runbook | Haiku | 1–3 | 10–30s |
| Documentation | Haiku | 2–4 | 15–40s |
| Simulator | Sonnet | 3–6 | 30–90s |

---

## Running the tests

```bash
.venv/bin/pytest tests/ -v
```

All 37 tests should pass in under 2 seconds (they use mocks and dry-run fixtures — no API calls).

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'demoforge'`**
Run `.venv/bin/pip install -e . -q` from the project root, then retry.

**`Your credit balance is too low`**
Add credits at console.anthropic.com → Plans & Billing.

**`AgentStalled` after 20 turns**
The agent hit `max_turns_per_agent`. Either the task is too complex or the prompts need tuning. You can raise the limit in `config/config.yaml` under `limits.max_turns_per_agent`.

**Run failed mid-way**
Resume from the last successful stage:
```bash
./run.sh resume <run-id>
```
