# Contributing

Keep changes small and honest. Prefer projecting recipe / run facts over
hardcoding task ids in the browser.

## Local loop

```bash
./run-studio.sh
# or: python3 -m pip install -e ".[dev]" && pytest
```

Open http://127.0.0.1:8000 (or `HT_PORT`). Cartpole → Train should produce
a video when a display (or xvfb) is available.

## Studio gates

Train / compare / deploy decisions that beginners can misuse live in
`studio/gates.js` and are covered by `tests/test_studio_gates.py` (Node).
Wire new gates there before teaching `app.js` to call them.

## Troubleshooting

See the table in [README.md](./README.md#troubleshooting) for Menagerie
downloads, missing display / `HT_NO_RENDER`, Docker, and walk exit 12.

## Docs map

- [docs/BETTERMENT.md](./docs/BETTERMENT.md) — polish the CPU studio
- [docs/ROADMAP.md](./docs/ROADMAP.md) — GPU / hardware phases
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) — job spec contract
