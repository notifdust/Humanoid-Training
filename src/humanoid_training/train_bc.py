from __future__ import annotations

from pathlib import Path

import numpy as np


def fit_linear_bc(
    observations: np.ndarray,
    actions: np.ndarray,
    ridge: float = 1e-4,
) -> np.ndarray:
    """Least-squares policy: action = [obs, 1] @ W. CPU stand-in for ACT."""
    obs = np.asarray(observations, dtype=np.float64)
    act = np.asarray(actions, dtype=np.float64)
    if obs.ndim != 2 or act.ndim != 2 or len(obs) != len(act):
        raise ValueError("observations and actions must be 2D arrays of equal length")
    bias = np.ones((obs.shape[0], 1), dtype=np.float64)
    x = np.concatenate([obs, bias], axis=1)
    xtx = x.T @ x + float(ridge) * np.eye(x.shape[1])
    xty = x.T @ act
    return np.linalg.solve(xtx, xty)


def predict_linear_bc(weights: np.ndarray, observation: np.ndarray) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float64).reshape(-1)
    x = np.concatenate([obs, [1.0]])
    return x @ np.asarray(weights, dtype=np.float64)


def save_bc(path: Path, weights: np.ndarray) -> Path:
    path = Path(path)
    np.savez(path, weights=np.asarray(weights, dtype=np.float64))
    return path


def load_bc(path: Path) -> np.ndarray:
    data = np.load(Path(path))
    if "weights" in data.files:
        return np.asarray(data["weights"], dtype=np.float64)
    if "W" in data.files:
        return np.asarray(data["W"], dtype=np.float64)
    raise KeyError(f"checkpoint has no weights/W: {list(data.files)}")
