# Humanoid Training

Pick a canned task → click **Train** → watch a video.

A beginner studio over MuJoCo / Isaac Lab / LeRobot — not a new simulator.
Think Canva on top of existing engines, not Photoshop.

## Quick start (CPU laptop)

```bash
git clone https://github.com/notifdust/Humanoid-Training.git
cd Humanoid-Training
chmod +x run-studio.sh
./run-studio.sh
```

Open **http://127.0.0.1:8000** → **Cartpole → Train** → play the video (~30s).

`./run-studio.sh` creates `.venv` when needed. Overrides: `HT_HOST`, `HT_PORT`, `HT_PYTHON`.

Manual install (use a venv on Debian/Ubuntu — system pip is blocked):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m humanoid_training.cli serve --host 127.0.0.1 --port 8000
```

Do **not** install the Ubuntu TeX package named `ht`. After install, the CLI is
`ht` or `python -m humanoid_training.cli`.

### What to click

1. **Cartpole → Train** — pole stays up. Proves Train → video on this machine.
2. **G1 stand → Train** — humanoid holds a pose and waves. Not walking.
3. **Pick and place → Train** — mustard into the bowl. Not finger grasping.

**G1 walk** needs an NVIDIA GPU (see below). There is no G1 reach recipe.

Rooms: **Robots · Tasks · Data · Runs**. Job spec is under **Advanced**.

## GPU: prove a real G1 walk (Phase 3c)

On a machine with an NVIDIA GPU and working network (CUDA wheels are large):

```bash
cd Humanoid-Training          # existing clone is fine
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[playground]"
pip install "jax[cuda12]==0.9.2"   # ~1h on a slow link — one-time

python -c "import jax; print(jax.__version__, jax.default_backend())"
# want: 0.9.2 gpu

python -m humanoid_training.cli proof walk --check   # want ok:true
python -m humanoid_training.cli proof walk           # writes eval.mp4 + proof_3c.json
```

Important pins:

- Bare `pip install playground` can pull **jax 0.11** (breaks brax) and a **CPU** jaxlib.
- Always use `jax[cuda12]==0.9.2`. Confirm `default_backend()` is `gpu` before proving.
- If the cudnn download stalls, retry with `pip install --resume-retries 50 'jax[cuda12]==0.9.2'`.

Without a local GPU, use OSMO harvest or Hugging Face Jobs when credentials exist
(`ht proof osmo` / `ht proof hf-jobs`) — that is Phase 3d, not a local walk proof.

Full operator notes: [docs/NEXT.md](docs/NEXT.md).

## CLI

```bash
python -m humanoid_training.cli recipes
python -m humanoid_training.cli train spec/examples/cartpole-balance.json
python -m humanoid_training.cli fetch-assets unitree_g1
python -m humanoid_training.cli train spec/examples/g1-stand.json
python -m humanoid_training.cli train spec/examples/g1-walk.json --compile-only
python -m humanoid_training.cli train spec/examples/cartpole-balance.json --docker
```

`--docker` is the Phase 1 CPU container runner. GPU recipes are refused there.

Eval video needs a display (MuJoCo GLFW). Headless scoring without a clip:

```bash
HT_NO_RENDER=1 python -m humanoid_training.cli train spec/examples/cartpole-balance.json
```

Outputs live under `runs/<id>/` (`eval.mp4`, `manifest.json`). Deploy stays
fail-closed (`ht deploy <run_id>`) until a hardware profile exists.

## Recipes

| Recipe | Where it runs | Result |
|---|---|---|
| `cartpole-balance` | CPU | Pole stays up. Gold clip in-tree. |
| `g1-stand` | CPU | Stand + wave. Not walking. Gold clip. |
| `pick-and-place` | CPU linear-BC; ACT on GPU+LeRobot | Mustard→bowl. Not grasping. |
| `g1-walk` | GPU + Playground / mjlab / Isaac (or OSMO harvest) | Walking eval from the engine. No gold clip. |
| `g1-walk-rough` | GPU + mjlab / Isaac | Rough terrain walk. No gold clip. |
| `g1-track` | GPU + mjlab + `HT_MJLAB_MOTION` | Motion tracking (LAFAN1 Hub pin). No gold clip. |
| `g1-pickplace` | GPU + Isaac + `HT_ISAAC_DATASET` | Locomanipulation PickPlace. No gold clip. |
| `g1-pickplace-fixed` | GPU + Isaac + dataset | Fixed-base PickPlace. No gold clip. |

## Docs

| Doc | What it is |
|---|---|
| [docs/NEXT.md](docs/NEXT.md) | Mission scorecard + what to do next (N1b walk proof first) |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Phase exit tests (GPU / remote / hardware) |
| [docs/VISION.md](docs/VISION.md) | Why we compile into engines instead of replacing them |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Job spec, adapters, runners |
| [docs/BETTERMENT.md](docs/BETTERMENT.md) | Completed CPU-studio polish (B0–B5) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Local loop and studio gates |

## Non-goals

- A new physics engine, policy family, or dataset format
- Competing with LeLab on SO-ARM101
- Fleet ops (use Foxglove / Formant later)
- Fake walk / ACT / OSMO success on CPU CI

## Tests

```bash
source .venv/bin/activate   # if you use a venv
pytest
# Optional gold retrain (needs ffmpeg + display or xvfb):
# HT_GOLD=1 xvfb-run -a pytest -m gold
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `externally-managed-environment` | Use `python3 -m venv .venv && source .venv/bin/activate` |
| `ht` wants TeX / apt | Wrong package. Use the venv: `pip install -e .` then `ht` or `python -m humanoid_training.cli` |
| First G1 stand hangs offline | Needs git + network once (~30MB Menagerie). Or `ht fetch-assets unitree_g1` |
| Success but no `eval.mp4` | Need a display (or xvfb). `HT_NO_RENDER=1` skips the clip on purpose |
| Docker train missing daemon | Start Docker, or use in-process Train (default) |
| G1 walk exit 12 on laptop | Expected without GPU. Not a stand clip |
| `jax … Falling back to cpu` | CUDA wheels incomplete. `pip install --resume-retries 50 'jax[cuda12]==0.9.2'` until `default_backend()` is `gpu` |
| `device_put_replicated` AttributeError | jax too new. Pin `jax[cuda12]==0.9.2` |
| Studio port in use | `HT_PORT=8060 ./run-studio.sh` |

## License

TBD.
