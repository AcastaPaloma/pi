# Relationship to the original paper

This playground was originally inspired by a bimanual UR5e platform from the π₀.₇ paper. The project now targets **two SO-101 followers** for inexpensive printed-object manipulation. It does not reproduce that paper's robot platform or reported results.

Retained interfaces include 20 Hz position-servo control, approximate numerical IK, optional RGB observations, task text, and action chunks. The arms, grippers, dimensions, camera placements and task workspace now follow an SO-101 experiment. Each arm has five arm axes plus a gripper; arbitrary six-axis end-effector poses may be infeasible. `ee_delta` keeps the approach direction downward, while `ee` attempts the requested pose and reports the positional residual.

The provided tasks are reaching and contact-based pick/place with a 30 mm cube or cylinder. Both arms can be commanded, but success scoring concerns the left arm. Scripted controllers use privileged object/goal state and are debugging demonstrations. No weld/attachment shortcut is used for grasping. There is no trained policy or claim of real-world transfer.

See [the manipulation plan](SO101_MANIPULATION.md), [observation interface](OBSERVATIONS.md), and [asset provenance](../pi_sim/assets/README.md).
