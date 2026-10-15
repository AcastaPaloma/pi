# SO-101 asset provenance

`vendor/robotstudio_so101` is vendored unmodified from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/f054586a8e90465d49ee5be15335c4a0c7f57caf/robotstudio_so101), revision `f054586a8e90465d49ee5be15335c4a0c7f57caf`.

The Robot Studio SO-101 model is Apache-2.0; its original `LICENSE`, `README.md` and `CHANGELOG.md` are included. `UPSTREAM.json` records the exact repository revision and copied paths. The gallery PNG is omitted.

`pi_sim/scene.py` composes two copies by prefixing names and resolving mesh paths, then adds a tabletop, colored primitive object, target pad, a fixed overhead 32×32 mm camera proxy on a gantry, and base placements. `so101_camera_rig.json` identifies the Arducam B030401 IMX708 USB autofocus modules and specifies all three cameras' native resolution, intrinsics, distortion and mounting poses. Its wrist poses start from upstream; optics use a nominal pinhole estimate from the manufacturer's horizontal FOV, with unmeasured distortion. Robot contact/inertial/servo definitions remain upstream. See [camera calibration](../../docs/CAMERA_RIG.md). No claim is made that this model is a measured calibration of your physical arms.

The previous UR5e and Robotiq assets have been replaced by this SO-101 model.
