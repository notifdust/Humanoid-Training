#!/usr/bin/env bash
# Start the studio from the repo root.
# Copy-paste friendly. Do not cd to /path/to/...
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f pyproject.toml ]]; then
  echo "This script must live in the Humanoid-Training clone (pyproject.toml missing)." >&2
  exit 1
fi

HOST="${HT_HOST:-127.0.0.1}"
PORT="${HT_PORT:-8000}"

pick_python() {
  if [[ -n "${HT_PYTHON:-}" ]]; then
    echo "${HT_PYTHON}"
    return
  fi
  # Prefer an already-working install (editable or site-packages).
  if python3 -c "import humanoid_training, fastapi" 2>/dev/null; then
    echo "python3"
    return
  fi
  if [[ -x .venv/bin/python ]] && .venv/bin/python -c "import humanoid_training, fastapi" 2>/dev/null; then
    echo ".venv/bin/python"
    return
  fi
  echo ""
}

ensure_venv() {
  # Repair a broken half-created .venv (no pip / ensurepip missing).
  if [[ -x .venv/bin/python ]] && .venv/bin/python -c "import pip" 2>/dev/null; then
    return
  fi
  echo "Setting up .venv …"
  rm -rf .venv
  if ! python3 -m venv .venv 2>/tmp/ht-venv.err; then
    echo "python3 -m venv failed (often missing ensurepip). Trying --without-pip …" >&2
    python3 -m venv --without-pip .venv
    if ! .venv/bin/python -c "import ensurepip" 2>/dev/null; then
      echo "Bootstrapping pip with get-pip.py …"
      curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/ht-get-pip.py
      .venv/bin/python /tmp/ht-get-pip.py
    else
      .venv/bin/python -m ensurepip --upgrade
    fi
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install -q -U pip
  python -m pip install -q -e ".[dev]"
}

PY="$(pick_python)"
if [[ -z "${PY}" ]]; then
  ensure_venv
  PY=".venv/bin/python"
fi

echo
echo "Humanoid Training studio"
echo "  python: ${PY}"
echo "  Open http://${HOST}:${PORT}"
echo "  Click Cartpole → Train. Wait ~30s for a video."
echo

exec "${PY}" -m humanoid_training.cli serve --host "${HOST}" --port "${PORT}"
