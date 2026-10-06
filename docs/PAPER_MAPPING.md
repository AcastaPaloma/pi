# What matches the paper

Reference: [π₀.₇: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities](https://arxiv.org/html/2604.15483v2), especially Sections VI-B and VIII. The original PDF is in this folder.

This project recreates an accessible **simulation approximation of one robot platform**, not the paper's experiments, training system, or policy. The paper describes physical robot deployments and does not provide an exact MuJoCo scene to reproduce.

| Paper detail | This environment |
| --- | --- |
| Bimanual UR5e with Robotiq parallel-jaw grippers | Two Menagerie UR5e arms with Robotiq 2F-85 models. The 2F-85 variant is our choice; the cited system description only says Robotiq. |
| UR5e control at 20 Hz | One environment step is 0.05 s, with 25 physics steps at 0.002 s. |
| Joint targets applied by a PD controller | Menagerie's position servos with damping; its gains and torque limits are retained. Exact paper gains are not specified. We additionally limit target changes to 0.1 rad per control step. |
| End-effector targets converted by numerical IK | Damped least-squares pose IK using MuJoCo Jacobians, followed by the same joint servos. This is not a collision-aware planner. |
| Joint and end-effector control modes | `joint` and `ee`; `ee_delta` is an additional simplified interface for learning and keyboard play. Our units and conventions are explicit, not claimed to match a released π₀.₇ checkpoint schema. |
| Front camera plus a wrist camera on each arm | `front`, `left_wrist`, `right_wrist`. Camera placements and intrinsics are approximations. |
| Images resized to 448 × 448 | Optional RGB observations default to 448 × 448. |
| 50-action predictions, executing 15 or 25 at a time | `step_chunk()` accepts any length. Use 50-step chunks and execute `chunk[:15]` or `chunk[:25]` if desired. No policy inference or asynchronous RTC is implemented. |
| Task text, subtask text, metadata and subgoal image conditioning | `info['instruction']` supplies task text; cameras and state are exposed so your later policy can add context. No language understanding, subgoal generator or memory encoder is included. |

## Deliberate simplifications

- The starter tasks are reaching a visible target and moving a 4 cm cube onto a disk. They are local learning tasks, not the paper's laundry, kitchen, or benchmark tasks.
- Both arms are available; task scoring concerns the left arm. The right arm holds its pose until you command it.
- Placement, tabletop dimensions, camera poses, reset distribution, reward, workspace bounds and episode length are local choices.
- The cube is grasped through ordinary contacts and friction. There is no attachment/weld shortcut. The Robotiq model's equality constraints describe its mechanical linkages.
- No PyTorch code, trained model, π checkpoint, neural network, dataset download, or RL algorithm is part of this project.
- The convenient task/object state is privileged simulator information. A vision policy can use just `state` and `images` instead. The scripted demo uses exact object/goal state.
- `images=False` avoids rendering during physics-only loops. Camera rendering is opt-in and needs a working graphics context.
- `ee_delta` keeps the gripper's reset orientation. Full `ee` control exposes rotation vectors. Joint commands have no Cartesian workspace constraint; contacts remain enabled.

## Sources

- [Paper, v2](https://arxiv.org/html/2604.15483v2)
- [MuJoCo Python API and macOS viewer notes](https://mujoco.readthedocs.io/en/stable/python.html)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), revision `f054586a8e90465d49ee5be15335c4a0c7f57caf`; see [asset provenance](../pi_sim/assets/README.md).
