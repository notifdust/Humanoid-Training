#!/usr/bin/env bash
# Copy-paste from the repo root. Do not cd to /path/to/...
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f pyproject.toml ]]; then
  echo "This script must live in the Humanoid-Training clone (pyproject.toml missing)." >&2
  exit 1
fi
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q -e ".[dev]"
echo
echo "Open http://127.0.0.1:8000"
echo "Click Cartpole → Train. Wait for a video."
echo
exec python -m humanoid_training.cli serve --host 127.0.0.1 --port 8000
