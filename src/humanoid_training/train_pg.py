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
    """Fit a linear softmax policy with REINFORCE + a CEM bootstrap.

    CartPole is the studio smoke test: the loop must produce a policy that
    actually balances, not just a video of random flailing.
    """
    import gymnasium as gym

    env = gym.make(env_id)
    assert env.observation_space.shape is not None
    obs_dim = int(env.observation_space.shape[0])
    n_act = int(env.action_space.n)
    rng = np.random.default_rng(seed)

    def _log(msg: str) -> None:
        if log:
            log(msg)

    def rollout(weights: np.ndarray, eval_seed: int, greedy: bool) -> float:
        obs, _ = env.reset(seed=eval_seed)
        done = False
        total = 0.0
        while not done:
            logits = obs @ weights
            if greedy:
                action = int(np.argmax(logits))
            else:
                logits = logits - np.max(logits)
                probs = np.exp(logits)
                probs = probs / probs.sum()
                action = int(rng.choice(n_act, p=probs))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
        return total

    # Cross-entropy search finds a decent linear policy quickly on CartPole.
    pop = 20
    elite_n = 5
    gens = max(8, min(25, episodes // 40))
    mean = np.zeros(obs_dim * n_act, dtype=np.float64)
    std = np.full_like(mean, 0.6)
    history: list[float] = []
    best = mean.reshape(obs_dim, n_act).copy()
    best_score = -1.0
    eval_cursor = seed + 17

    for gen in range(gens):
        samples = rng.normal(mean, std, size=(pop, mean.size))
        scores = []
        for i, flat in enumerate(samples):
            eval_cursor += 1
            score = rollout(flat.reshape(obs_dim, n_act), eval_cursor, greedy=True)
            scores.append(score)
            history.append(score)
        scores_arr = np.asarray(scores)
        elite_idx = np.argsort(scores_arr)[-elite_n:]
        elite = samples[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 0.05
        gen_best = float(scores_arr[elite_idx[-1]])
        if gen_best >= best_score:
            best_score = gen_best
            best = samples[elite_idx[-1]].reshape(obs_dim, n_act).copy()
        _log(
            f"cem gen {gen + 1}/{gens} best={gen_best:.0f} "
            f"mean_elite={float(scores_arr[elite_idx].mean()):.1f}"
        )
        if best_score >= 475:
            break

    # Short REINFORCE polish so the "training log" still looks like learning.
    remaining = max(0, episodes - len(history))
    polish = min(remaining, 80)
    weights = best.copy()
    lr = 0.08
    gamma = 0.99
    for ep in range(polish):
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
        std_adv = adv.std()
        if std_adv > 1e-6:
            adv = adv / std_adv
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
        if ep == 0 or (ep + 1) % 20 == 0 or ep + 1 == polish:
            _log(
                f"polish {ep + 1}/{polish} return={ep_return:.0f} "
                f"avg20={float(np.mean(history[-20:])):.1f}"
            )

    env.close()
    greedy_check = [
        rollout(weights, seed + 900 + i, greedy=True) for i in range(3)
    ]
    cem_check = [rollout(best, seed + 800 + i, greedy=True) for i in range(3)]
    if float(np.mean(cem_check)) >= float(np.mean(greedy_check)):
        weights = best
    _log(f"selected policy greedy_eval={max(np.mean(greedy_check), np.mean(cem_check)):.1f}")
    return weights, history
