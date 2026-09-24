from __future__ import annotations

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MUJOCO_GL", "glfw")


@pytest.fixture(autouse=True)
def _cpu_studio_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests assume the CPU studio unless a case opts into a fake GPU CLI."""
    monkeypatch.setenv("HT_GPU", "0")
    monkeypatch.setenv("HT_PLAYGROUND_GPU", "0")
    monkeypatch.setenv("HT_PLAYGROUND_CLI", "0")
    monkeypatch.setenv("HT_MJLAB_CLI", "0")
    monkeypatch.setenv("HT_ISAAC_CLI", "0")
    monkeypatch.setenv("HT_OSMO_CLI", "0")
    monkeypatch.setenv("HT_LEROBOT_CLI", "0")
    monkeypatch.setenv("HT_DOCKER_GPU", "0")

