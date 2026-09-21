# Humanoid Training

A studio for training robots without making people become Isaac Lab
experts first.

**What you do:** pick a canned task → click Train → watch a video.
That is the product. It is Canva on top of MuJoCo / Isaac Lab, not a
new simulator.

## Run it on your computer

From a **new terminal** (your home directory is fine):

```bash
git clone https://github.com/notifdust/Humanoid-Training.git
cd Humanoid-Training
./run-studio.sh
```

Then open **http://127.0.0.1:8000**. Click **Cartpole → Train**.
Wait ~30s. Play the video. That is the loop.

If `./run-studio.sh` is not executable: `chmod +x run-studio.sh` and run it again.

Same steps by hand:

```bash
cd Humanoid-Training          # the folder that contains pyproject.toml
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m humanoid_training.cli serve --host 127.0.0.1 --port 8000
```

Do **not** `cd /path/to/Humanoid-Training` — that was a placeholder. Do **not**
install the Ubuntu `ht` TeX package. After `pip install -e ".[dev]"` the
command is `ht` **or** `python -m humanoid_training.cli`.

You need a display for eval video (MuJoCo uses GLFW). Headless:
`HT_NO_RENDER=1 python -m humanoid_training.cli train spec/examples/cartpole-balance.json`
still scores success; it will not write `eval.mp4`.

### What to click

1. **Cartpole → Train.** Pole stays up. Proves Train → video on your machine.
2. **G1 stand → Train.** Humanoid holds a pose and waves. Not walking.
3. **Pick and place → Train.** Mustard goes in the bowl; the arm follows. Not finger grasping.

Skip **G1 reach** on a laptop. **G1 walk** also stops on a laptop (needs
an NVIDIA GPU and `pip install playground`). On a GPU box it launches
for real — not a stand clip.

Rooms: Robots · Tasks · Data · Runs. Job spec is under **Advanced**.

### Same jobs from the CLI

```bash
python -m humanoid_training.cli recipes
python -m humanoid_training.cli train spec/examples/cartpole-balance.json
python -m humanoid_training.cli fetch-assets unitree_g1
python -m humanoid_training.cli train spec/examples/g1-stand.json
python -m humanoid_training.cli train spec/examples/g1-walk.json --compile-only
python -m humanoid_training.cli train spec/examples/g1-walk.json
python -m humanoid_training.cli train spec/examples/cartpole-balance.json --docker
```

`--docker` is the Phase 1 container runner (CPU image in `Dockerfile`).
GPU recipes are refused there. If Docker is missing it stops with a next
step; in-process Train still works.

On a machine with an NVIDIA GPU:

```bash
pip install playground
python -m humanoid_training.cli train spec/examples/g1-walk.json
```

That runs `train-jax-ppo --env_name G1JoystickFlatTerrain`, copies
`rollout0.mp4` to `eval.mp4`, and stamps `facts.engine=playground`.
Without a GPU the same command compiles payloads and exits 12.

Outputs: `runs/<id>/eval.mp4` and `manifest.json`.

## Read this first

- **[Roadmap](docs/ROADMAP.md)** — what is live, what this round verified, what comes next
- **[Product vision](docs/VISION.md)** — landscape and why we compile instead of replacing engines
- **[Architecture](docs/ARCHITECTURE.md)** — job spec, adapters, runners

## What works today

| Recipe | What happens |
|---|---|
| `cartpole-balance` | Gymnasium RL on CPU, eval video, gold clip |
| `g1-stand` | MuJoCo G1 from Menagerie, stand + both-arm wave, eval video, gold clip |
| `g1-walk` | Compile to Playground / mjlab / Isaac Lab. **Launches** Playground PPO when a GPU and `train-jax-ppo` are present. |
| `g1-reach` | Compile to mjlab / Isaac Lab |
| `pick-and-place` | Demos → linear BC steers mustard; G1 arm plays pick/lift/place. Gold clip. Not finger grasping, not ACT. |

## Non-goals (for now)

- A new physics engine
- A new policy architecture or dataset format
- Competing with LeLab on SO-ARM101 unboxing
- Fleet operations (use Foxglove / Formant later)

## Tests

```bash
pytest
# Retrain CPU recipes and compare eval.mp4 to gold clips (needs ffmpeg + a display or xvfb):
# HT_GOLD=1 xvfb-run -a pytest -m gold
```

## License

TBD.
