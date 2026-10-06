# Start here

Everything is installed in `mini-vla`. Run these from this project folder.

**Watch the robot pick up the block:**

```bash
./play.sh --demo --task pick_place
```

**Control it yourself:**

```bash
./play.sh
```

Click the window. **W/S/A/D** moves sideways, **Q/E** moves up/down, **G** opens/closes the fingers. **1/2** selects the arm, **R** resets, **P** pauses, **B** switches to a demo. Move the left gripper to the green sphere. Close the window to quit.

**Control it with your own Python:**

```bash
conda activate mini-vla
python examples/tap_in.py
```

Edit the `action = ...` line in that example to plug in your controller. The core API is `env.reset()` and `env.step(action)`. Read joint positions through `env.robot["left"].joint_positions`; get images through `env.render_camera("front")`; access raw MuJoCo through `env.model` and `env.data`.

The [full guide](README.md) explains actions, cameras, recording demonstrations, and a fresh installation. The [paper mapping](docs/PAPER_MAPPING.md) explains which π₀.₇ robot details are reproduced and which are approximated. No training model is included.
