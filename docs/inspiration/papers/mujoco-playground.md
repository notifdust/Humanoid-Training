# MuJoCo Playground (Zakka et al., 2025)

- **arXiv:** https://arxiv.org/abs/2502.08844  
- **Code:** https://github.com/google-deepmind/mujoco_playground  

Open GPU-accelerated suite for locomotion and manipulation with MJX;
includes Unitree G1/H1 joystick envs and documented sim-to-real on G1.

**Takeaway for us:** our `g1-walk` → Playground adapter is on the right
engine. Short proof (2048 steps) closes honesty; longer trains + domain
randomization are how you get paper-quality gaits for demos.
