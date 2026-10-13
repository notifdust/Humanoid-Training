# BeyondMimic (2025)

- **arXiv:** https://arxiv.org/abs/2508.08241  
- **Code:** https://github.com/HybridRobotics/whole_body_tracking  

Motion tracking that scales across agile human clips with shared
hyperparameters, then guided diffusion for test-time tasks (nav, joystick,
obstacle avoidance) with real hardware transfer.

**Takeaway for us:** recipe `g1-track` + LAFAN1 Hub is the correct thin
wrap. Do not build diffusion controllers in-tree — pin engines and motions.
