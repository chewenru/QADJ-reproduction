#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
"$VENV/bin/pip" install -r "$ROOT/qadj_reproduction/requirements.txt"

echo "Environment ready:"
echo "  source $VENV/bin/activate"
