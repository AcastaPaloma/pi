# SO-101 asset provenance

`vendor/robotstudio_so101` is vendored unmodified from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/f054586a8e90465d49ee5be15335c4a0c7f57caf/robotstudio_so101), revision `f054586a8e90465d49ee5be15335c4a0c7f57caf`.

The Robot Studio SO-101 model is Apache-2.0; its original `LICENSE`, `README.md` and `CHANGELOG.md` are included. `UPSTREAM.json` records the exact repository revision and copied paths. The gallery PNG is omitted.

`pi_sim/scene.py` composes two copies by prefixing names and resolving mesh paths, then adds a tabletop, colored primitive object, target pad, fixed front/overhead cameras, and base placements. Wrist camera position and robot contact/inertial/servo definitions come from the upstream model. No claim is made that this model is a measured calibration of your physical arms.

The previous UR5e and Robotiq assets have been replaced by this SO-101 model.
