# google-deepmind/mujoco_playground

- **URL:** https://github.com/google-deepmind/mujoco_playground  
- **Site:** https://playground.mujoco.org  
- **License:** Apache-2.0  
- **Paper:** [arXiv:2502.08844](https://arxiv.org/abs/2502.08844)

GPU-accelerated MuJoCo MJX environments + `train-jax-ppo`. Ships
`G1JoystickFlatTerrain` / rough variants; paper reports G1 joystick
locomotion under ~30 minutes on two RTX 4090s and indoor sim-to-real.

**Our use:** primary Phase 3c engine. Adapter launches `train-jax-ppo`;
harvests rollout mp4 → `eval.mp4`. Pin `jax==0.9.2` + CUDA jaxlib
(Brax still calls `device_put_replicated`).

**Verified:** operator run `20261007T193550Z-g1-walk-9e796b`.
