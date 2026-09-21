#!/bin/sh
# CPU image entrypoint. xvfb gives GLFW a display so eval.mp4 can still be written.
set -e
if command -v xvfb-run >/dev/null 2>&1 && [ -z "${DISPLAY:-}" ]; then
  exec xvfb-run -a python -m humanoid_training.cli "$@"
fi
exec python -m humanoid_training.cli "$@"
