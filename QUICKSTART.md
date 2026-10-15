# Start here

Everything is installed in `mini-vla`. Run these from this project folder.

**Batch 128 observations for your encoder:**

```bash
conda activate mini-vla
python examples/batch_observations.py
python examples/batch_observations.py --source real --prepare-demo
```

Use `robot.observations()` to get the NumPy batch directly. See the [short batch guide](docs/BATCH_ENCODER.md) for Python snippets, grids of 128 scenes, and the real-data cache.

**Capture camera images and proprioception directly:**

```bash
conda activate mini-vla
python -m pi_observe sim
python examples/observe_robot.py
```

Captures are saved under `outputs/observations/`. The [observation guide](docs/OBSERVATIONS.md) also explains reading your physical SO-101 pair after configuring its serial ports, calibration files, and cameras.

All three camera PNGs preserve the native lens image (640×480). See [camera rig matching](docs/CAMERA_RIG.md) to share measured optics/poses with hardware. In the sim window, press **V** to cycle the camera views, or start with `./play.sh --view overhead`.

**Get data for your low-level encoder:**

```bash
conda activate mini-vla
python -m pi_data sample --episode 0 --frame 100 --goal-seconds 2
python -m pi_data serve
```

Open http://127.0.0.1:8000/docs. The [manipulation guide](docs/SO101_MANIPULATION.md) explains the selected real SO-101 dataset, API, camera placement and printed objects. The sample is saved to `outputs/encoder_sample.npz`.

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

Edit the `action = ...` line in that example to plug in your controller. The core API is `env.reset()` and `env.step(action)`. Read joint positions through `env.robot["left"].joint_positions`; get images through `env.render_camera("overhead")`; access raw MuJoCo through `env.model` and `env.data`.

The [full guide](README.md) explains actions, cameras, recording demonstrations, and a fresh installation. The [paper mapping](docs/PAPER_MAPPING.md) explains which π₀.₇ robot details are reproduced and which are approximated. No training model is included.

For colored printed objects, public manipulation datasets and the three-camera mounting layout, see [the SO-101 plan](docs/SO101_MANIPULATION.md). To train the π₀.₇ reimplementation, see [the implementation notes](docs/PI07_IMPLEMENTATION.md). Try `./play.sh --demo --task pick_place --object-shape cylinder --object-color blue`.
