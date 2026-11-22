#!/usr/bin/env bash
# FormBack launcher for macOS / Linux. Needs Python 3.10, 3.11 or 3.12.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

if ! "$PY" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)'; then
  echo "FormBack needs Python 3.10, 3.11 or 3.12 (found: $("$PY" --version 2>&1))." >&2
  echo "Install one, or use Docker instead:  docker compose up" >&2
  exit 1
fi

[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q -r requirements.txt

echo "FormBack is running at http://localhost:8000  (Ctrl+C to stop)"
exec uvicorn api.main:app --host 127.0.0.1 --port 8000
