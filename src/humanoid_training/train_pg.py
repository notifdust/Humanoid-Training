from __future__ import annotations

from collections.abc import Callable

import numpy as np

LogFn = Callable[[str], None]


def train_linear_policy(
    env_id: str,
    episodes: int,
    seed: int,
    log: LogFn | None = None,
) -> tuple[np.ndarray, list[float]]:
    """REINFORCE with a linear softmax policy. Enough to solve CartPole on CPU."""
    import gymnasium as gym

    env = gym.make(env_id)
    assert env.observation_space.shape is not None
    obs_dim = int(env.observation_space.shape[0])
    n_act = int(env.action_space.n)
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.1, size=(obs_dim, n_act))
    lr = 0.02
    gamma = 0.99
    history: list[float] = []

    def _log(msg: str) -> None:
        if log:
            log(msg)

    for ep in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        observations: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        done = False
        while not done:
            logits = obs @ weights
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / probs.sum()
            action = int(rng.choice(n_act, p=probs))
            observations.append(np.asarray(obs, dtype=np.float64))
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            done = bool(terminated or truncated)

        returns = []
        g = 0.0
        for reward in reversed(rewards):
            g = reward + gamma * g
            returns.append(g)
        returns.reverse()
        adv = np.asarray(returns, dtype=np.float64)
        adv = adv - adv.mean()
        std = adv.std()
        if std > 1e-6:
            adv = adv / std

        grad = np.zeros_like(weights)
        for ob, action, advantage in zip(observations, actions, adv, strict=True):
            logits = ob @ weights
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / probs.sum()
            one_hot = np.zeros(n_act)
            one_hot[action] = 1.0
            grad += advantage * np.outer(ob, one_hot - probs)
        weights += lr * grad / max(len(actions), 1)

        ep_return = float(sum(rewards))
        history.append(ep_return)
        if ep == 0 or (ep + 1) % 50 == 0 or ep + 1 == episodes:
            avg = float(np.mean(history[-100:]))
            _log(f"episode {ep + 1}/{episodes} return={ep_return:.0f} avg100={avg:.1f}")

    env.close()
    return weights, history
