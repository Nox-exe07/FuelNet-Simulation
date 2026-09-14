#!/usr/bin/env bash
# run.sh — bash launcher (git-bash / WSL / Linux / macOS)
set -e
echo "=== Fuel & CO2 Simulation — http://127.0.0.1:8000 ==="

if command -v python3.13 &>/dev/null; then PY=python3.13
elif command -v py &>/dev/null && py -3.13 --version &>/dev/null; then PY="py -3.13"
elif command -v python3 &>/dev/null; then PY=python3
elif command -v python &>/dev/null; then PY=python
else echo "No Python found. Install Python 3.13."; exit 1
fi

echo "Using: $PY"
$PY -m pip show fastapi >/dev/null 2>&1 || $PY -m pip install -r requirements.txt
$PY -m uvicorn simulation.backend.api:app --host 127.0.0.1 --port 8000 --reload
