"""Tests for plan.schema.json and manifest.schema.json validation."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

_SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"


def _load(name: str) -> dict:
    return json.loads((_SCHEMAS_DIR / name).read_text())


# ---------------------------------------------------------------------------
# plan.schema.json
# ---------------------------------------------------------------------------

VALID_PLAN = {
    "title": "Invoice Reconciliation Demo",
    "domain": "Finance / Accounts Payable",
    "persona": "AP Manager",
    "workflow_steps": ["Upload invoices", "Match line items", "Flag exceptions", "Send summary"],
    "demo_agents": [
        {"name": "intake", "role": "Read documents", "inputs": ["invoices"], "outputs": ["parsed_docs"]},
    ],
    "demo_tools": [{"name": "read_invoice", "purpose": "Parse invoice PDF"}],
    "data_entities": [{"name": "Invoice", "fields": ["id", "amount"], "approx_rows": 20}],
    "success_criteria": ["All invoices processed"],
    "assumptions": [],
    "html_sim": {"feasible": True, "reason": "Deterministic workflow"},
}


def test_valid_plan_passes():
    schema = _load("plan.schema.json")
    jsonschema.validate(VALID_PLAN, schema)


def test_plan_missing_required_field():
    schema = _load("plan.schema.json")
    bad = {k: v for k, v in VALID_PLAN.items() if k != "title"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_plan_demo_agents_empty_fails():
    schema = _load("plan.schema.json")
    bad = {**VALID_PLAN, "demo_agents": []}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_plan_html_sim_missing_reason():
    schema = _load("plan.schema.json")
    bad = {**VALID_PLAN, "html_sim": {"feasible": True}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


# ---------------------------------------------------------------------------
# manifest.schema.json
# ---------------------------------------------------------------------------

VALID_MANIFEST = {
    "files": [
        {
            "path": "data/generate_synthetic_data.py",
            "purpose": "Generates synthetic invoice and PO data",
            "depends_on": [],
            "blueprint_section": "Synthetic Data Spec",
        },
        {
            "path": "mock_llm_instructions.json",
            "purpose": "Canned LLM responses keyed by agent and step",
            "depends_on": ["data/generate_synthetic_data.py"],
            "blueprint_section": "Mocked LLM Layer",
        },
    ]
}


def test_valid_manifest_passes():
    schema = _load("manifest.schema.json")
    jsonschema.validate(VALID_MANIFEST, schema)


def test_manifest_empty_files_fails():
    schema = _load("manifest.schema.json")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"files": []}, schema)


def test_manifest_missing_purpose_fails():
    schema = _load("manifest.schema.json")
    bad = {"files": [{"path": "foo.py"}]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)
