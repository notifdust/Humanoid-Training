# AlexandreBrown/G1-Playground

- **URL:** https://github.com/AlexandreBrown/G1-Playground  

Minimal Unitree G1 policy ↔ DDS path shared by `unitree_mujoco` sim and
real robot. Write policy once; validate sim2sim then sim2real.

**Our use:** pattern for N5 hardware driver — same code path for sim and
real, domain/channel separation so scripts cannot talk to a live robot by
accident. Fail closed until profile + reduced speed exist.
