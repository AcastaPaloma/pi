# Vendored robot assets

Source: https://github.com/google-deepmind/mujoco_menagerie

Pinned revision: `f054586a8e90465d49ee5be15335c4a0c7f57caf`.

- `vendor/universal_robots_ur5e`: BSD-3-Clause; original LICENSE included.
- `vendor/robotiq_2f85`: BSD-2-Clause; original LICENSE included.

The upstream model files, assets and licenses are copied without edits. `pi_sim/scene.py` composes them at runtime by prefixing names, resolving mesh paths, attaching grippers at the arm attachment frames, and adding cameras and a tabletop. Upstream standalone lights and home keyframes are omitted during composition. Linkage constraints, inertias, collision meshes, servo gains, and force limits are preserved.

The runtime is self-contained: running the environment does not clone or download models.
