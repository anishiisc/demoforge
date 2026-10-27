#!/usr/bin/env bash
# DemoForge — quick launcher script
# Usage: ./run.sh [--dry-run] [--file path/to/request.txt] [--no-ask]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3.12 -m venv .venv
  .venv/bin/pip install -e ".[dev]" -q
fi

# Reinstall in case source changed
.venv/bin/pip install -e . -q

exec .venv/bin/demoforge "$@"
