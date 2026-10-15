#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
./.venv/bin/python scripts/audit_genericity.py
./.venv/bin/python -m unittest tests.test_semantic_foundation -q
