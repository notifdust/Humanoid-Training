# NVIDIA OSMO

- **Product:** https://developer.nvidia.com/osmo  
- **Docs:** https://nvidia.github.io/OSMO/  

YAML orchestrator for sim → train → eval → HIL across heterogeneous GPUs.
Not a studio; infra for people who already write workflows.

**Our use:** Phase 3d / N2b. `adapters/osmo_remote.py` already implements
submit → poll → rsync. Exit test: laptop without CUDA gets walking
`eval.mp4` with `facts.launch=osmo`.
