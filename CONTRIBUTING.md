# Contributing

Keep changes small and honest. Prefer projecting recipe / run `facts` over
hardcoding task ids in the browser.

## Local loop

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
./run-studio.sh
# or: pytest
```

Open http://127.0.0.1:8000 (or `HT_PORT`). **Cartpole → Train** should produce
a video when a display (or xvfb) is available.

Do not install the Ubuntu TeX package named `ht`. The project CLI is installed
into the venv by `pip install -e .`.

## Studio gates

Train / compare / deploy decisions that beginners can misuse live in
`studio/gates.js` and are covered by `tests/test_studio_gates.py` (Node).
Wire new gates there before teaching `app.js` to call them.

## GPU walk proof

See [README.md](./README.md#gpu-prove-a-real-g1-walk-phase-3c). Pin
`jax[cuda12]==0.9.2` and confirm `jax.default_backend()` is `gpu` before
claiming a walk.

## Commit identity

Commits on this repository should use the maintainer GitHub identity
(`Gabriele Sepolvere` / `145381062+notifdust@users.noreply.github.com`),
not a third-party bot or automation account.

## Troubleshooting

See the table in [README.md](./README.md#troubleshooting).

## Docs map

- [docs/NEXT.md](./docs/NEXT.md) — mission scorecard + next implementation steps
- [docs/ROADMAP.md](./docs/ROADMAP.md) — GPU / hardware phase exit tests
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) — job spec contract
- [docs/BETTERMENT.md](./docs/BETTERMENT.md) — completed CPU-studio polish (historical)
