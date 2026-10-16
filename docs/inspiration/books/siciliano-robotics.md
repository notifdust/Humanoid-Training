# Robotics textbooks (kinematics / control baseline)

Representative: Siciliano et al., *Robotics: Modelling, Planning and Control*;
plus Unitree / vendor application notes for G1 joint limits and safety.

**How we use it:** Deploy stays fail-closed — reduced speed, NaN/pose-limit
kills, sim-only until a hardware profile exists. Learning does not replace
basic joint safety.
