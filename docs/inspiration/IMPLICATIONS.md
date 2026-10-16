# Research → next actions

Prioritized for Humanoid Training after N1b. Full mission table stays in
[NEXT.md](../NEXT.md).

## Do next (operator-blocking)

1. **N2b — OSMO live harvest**  
   Same contract as `adapters/osmo_remote.py`: submit → poll → rsync `eval.mp4`,
   stamp `facts.launch=osmo` / `facts.device=remote`.  
   *Inspiration:* [NVIDIA OSMO](https://developer.nvidia.com/osmo) YAML
   orchestrator — we emit workflows, we do not become the cluster.

2. **N2c — HF Jobs live harvest (parallel path)**  
   LeLab’s cloud train pattern for people who already have `HF_TOKEN`.  
   *Inspiration:* [LeLab](https://huggingface.co/docs/lerobot/en/lelab) +
   LeRobot `--job.target`. Prefer if OSMO credentials are harder to get.

## Do soon (GPU already proven)

3. **N3b — ACT on mustard demos**  
   Same demos as linear-BC; badge `policy=act`.  
   *Inspiration:* Zhao et al. ACT; LeRobot `lerobot-train`.

4. **Longer Playground walk recipe (optional recipe flag)**  
   Proof stays short; add a “full walk” timesteps preset so eval video
   looks like the Playground paper (~minutes, not 2048 steps).  
   *Inspiration:* [MuJoCo Playground](https://arxiv.org/abs/2502.08844).

## Do when pins meet hardware

5. **N4b — Isaac Robomimic PickPlace live** — needs `HT_ISAAC_DATASET`.  
6. **`g1-track` live** — LAFAN1 NPZ + mjlab / BeyondMimic stack.  
7. **N5 — Unitree reduced-speed deploy** — only after sim walk (done) +
   driver; follow Unitree RL mjlab ONNX / DDS patterns, fail closed.

## Explicit non-goals (reinforced by research)

| Temptation | Why not |
|---|---|
| Browser WASM physics (SimArena) | Different product; we compile to engines |
| Compete with LeLab on SO-ARM101 | Their turf; we stay humanoid |
| Reimplement BeyondMimic diffusion | Wrap mjlab/Isaac Tracking recipes |
| New dataset format | Stay on LeRobotDataset |
| Fleet / Foxglove clone | Observability later, not the wedge |
