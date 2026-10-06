# Read the robot and capture everything your encoder needs

Use `pi_observe.Observer` directly from Python. Each read gives you current RGB images, named joint/gripper positions, timestamps, task text, optional subtask text, optional reference-goal images, and optional history. There is no HTTP hop or PyTorch dependency.

## Try it now in MuJoCo

```bash
conda activate mini-vla
python -m pi_observe sim
```

This creates a timestamped folder under `outputs/observations/`. Inside each `000000/` capture:

```text
left_wrist.png    native RGB view
right_wrist.png   native RGB view
front.png        native RGB view
encoder.npz      resized images + proprioception + text + goals/masks + history
metadata.json    joint names/units, timestamps, source, camera shapes
COMPLETE         present only after all files were saved
```

For multiple captures: `python -m pi_observe sim --frames 10 --fps 5`. The CLI advances the simulator between captures using a hold command; `Observer.read()` itself never advances it. Simulation steps remain 0.05 seconds each, independently of the CLI's wall-clock capture rate. The saved `simulation_time_seconds` distinguishes the clocks. The CLI holds the robot stationary; your control loop can move it between observations.

```python
from pi_sim import RobotEnv
from pi_observe import Observer, SimSource

with RobotEnv(control_mode="joint") as env:
    _, info = env.reset(seed=0)
    with Observer(SimSource(env), task=info["instruction"]) as observer:
        obs = observer.read()
        x = obs.inputs
        images = x["images"]                 # [3,448,448,3], RGB uint8
        state = x["proprioception"]          # [14] for this UR5e simulation
        prompt = str(x["task"])
        native_front = obs.images["front"]
        # Your encoder consumes x here.
        # obs.save("outputs/my_capture")     # Use a new directory.
        # env.step(your_action)
        # obs = observer.read()
```

An executable version is in `examples/observe_robot.py`. Use `observer.clear_history()` whenever you reset the environment or begin a new episode. Keep environment stepping and observation rendering on the same thread.

## Read your two physical SO-101s

The physical reader targets **two already configured/calibrated SO-101 follower arms with STS3215 motors, IDs 1–6 on each arm**, and USB/OpenCV cameras. The six position registers are read together per arm; both arms are read concurrently. Camera threads continuously drain frames into one latest-frame slot per camera. No sensor queue grows behind the encoder.

There were no SO-101 serial devices visible on this Mac during implementation. The physical transport is implemented and tested with mocked devices; actual camera/servo operation still needs a hardware check on your connected robot computer.

1. Install the small capture dependencies on that computer:

   ```bash
   python -m pip install -e '.[capture]'
   python -m pi_observe devices
   ```

2. Copy `configs/so101_observe.example.json` to `configs/so101_observe.local.json`. Set the two serial ports, the two existing LeRobot calibration JSON paths, and the three camera device IDs or paths. Linux `/dev/video*` paths are supported; macOS cameras usually use integer indices. Relative calibration/goal file paths are resolved relative to the config file. Device listing only lists serial devices and does not probe cameras.

3. Run:

   ```bash
   python -m pi_observe robot --config configs/so101_observe.local.json
   ```

The config's default camera order matches the folding dataset: `left_wrist`, `right_wrist`, `right_front`. Additional named cameras are supported and all configured cameras are included in the returned arrays. Check the first saved images to confirm the camera identities. Optional camera `rotation` is 0, 90, 180, or 270 degrees counterclockwise. Camera size/FPS requests are best-effort; metadata records actual image dimensions.

For your own live loop:

```python
import json
from pi_observe import Observer
from pi_observe.hardware import SO101Source

# For this Python example use absolute calibration paths in the config.
config = json.load(open("configs/so101_observe.local.json"))
source = SO101Source(config)
with Observer(source, task=config["task"], history=1) as observer:
    source.connect()
    obs = observer.read()
    x = obs.inputs
    print(x["camera_names"])
    print(x["proprioception"])     # [12]: left 6, then right 6
    # obs.save("outputs/real_capture")
```

The reader verifies the motor model and compares the existing calibration file against the servo's limits/homing offset before producing calibrated values. It sends only PING/READ/SYNC_READ protocol commands: no motor targets, torque changes, calibration writes, or configuration writes. Closing it preserves the motors' current settings. If calibration is missing or mismatched, complete the normal [LeRobot SO-101 calibration](https://huggingface.co/docs/lerobot/en/so101) separately. The serial ports should have one owner; do not run this standalone reader alongside another process using those same ports. Integrating observation capture into an existing teleoperation/controller process should reuse that process's connection.

The register mapping and unit conversion follow LeRobot's [SO follower](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py), [motor normalization](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/motors_bus.py), and [Feetech tables](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/tables.py). Five arm joints are in degrees, and each gripper uses 0–100 closed-to-open. Firmware already applies homing offsets; the reader does not subtract them twice. The full LeRobot/PyTorch stack is not installed by the capture extra.

On this Mac, importing the installed PyAV and OpenCV wheels in the same process produced duplicate AVFoundation-class warnings. Run dataset video replay and physical OpenCV capture in separate processes; the supplied CLIs already use separate paths. The live capture package does not import PyAV.

## What goes into the encoder

`obs.inputs` contains NumPy arrays, matching the core keys of `FoldingDataset.arrays(...)`:

| Field | Meaning |
| --- | --- |
| `images` | `[C,448,448,3]`, RGB uint8 current images |
| `camera_names` | `[C]`, explicit camera order |
| `proprioception` | `[D]`, measured positions in the documented native units |
| `task` | Scalar text prompt you supplied |
| `subtask`, `subtask_valid` | Optional text instruction and availability flag |
| `control_mode` | Requested action representation; does not command a motor |
| `subgoal_images`, `subgoal_valid` | `[C,448,448,3]` reference RGB images plus a `[C]` availability mask |
| `history_images` | `[T,C,448,448,3]`, oldest to current |
| `history_proprioception` | `[T,D]` |
| `history_valid` | `[T]`, false for missing history slots |
| `history_timestamps_ns` | Host monotonic times for selected frames; bookkeeping |
| `metadata_json` | Unicode JSON with timing, source and sensor schema |

Use your vision model's preprocessing after reading these arrays. No hidden standardization occurs. For the SO-101 dataset/robot pair, apply the same training-set state normalization in both cases. Always check joint names and calibration rather than relying only on vector length.

`history=1` means current only. Values up to 6 give approximately 1-second spacing in host time. The observer retains a bounded, downsampled buffer and masks slots without a sample within 250ms of their target time. The current image appears both in `images` and at the end of `history_images`; consume it once. This supplies temporal data but does not implement π0.7's MEM compression.

**The existing MuJoCo robot is UR5e:** its state has 14 values, arm angles in radians, and grippers in 0–1. The physical SO-101 and LeHome dataset have 12 values, arm angles in degrees, and grippers in 0–100. The simulation's external camera is named `front`; the dataset's is `right_front`. These differences are explicit, not silently remapped. A 12D SO-101 encoder needs the corresponding robot data or an explicit embodiment adapter.

## Prompts and visual subgoals

Camera images and joint state come from sensors. Task text, subtask instructions, and reference-goal images are supplied separately. For example:

```python
import numpy as np
from PIL import Image

observer.set_context(
    task="Fold the sheet of paper in half",
    subtask="Bring the left edge to the right edge",
    goal_images={
        "right_front": np.asarray(Image.open("desired_fold.png").convert("RGB"))
    },
)
```

Or put paths in the CLI config's `goal_images` mapping. Reference images are resized and accompanied by per-camera masks. Missing goals are zero-filled with `subgoal_valid=False`; **the encoder must mask them**, rather than interpreting them as a black desired scene. No future camera observation or world model is needed to read the live robot. Future-demonstration goals remain a training-only option in the separate dataset adapter.

The capture reader does not manufacture action labels or episode quality/mistake annotations. For supervised action training on your own demonstrations, additionally record the actual controller/teleoperator action sent after each observation, with its timestamp and units. Measured joint positions are proprioception, not a substitute for commanded-action labels.

## Timing and saving

Every live camera has a host timestamp recorded immediately after `VideoCapture.read()` completes. Each arm has a read interval and its midpoint timestamp. All are from the same monotonic host clock. The observer reports frame ages and the maximum timing spread; defaults reject samples older than 200ms or spread over 100ms. Treat these as initial configuration values and tighten them based on measured frame rates and motion.

These timestamps **do not establish hardware exposure synchronization** or reveal buffering inside a camera. Continuous reads reduce application backlog; metadata exposes host timing, repeated camera-frame sequence numbers, and actual read intervals. Use hardware timestamps/triggered cameras if the task later needs stronger synchronization. A repeated frame can still be accepted while fresh; inspect frame sequence numbers if every control step must use a new frame.

`obs.save(...)` preserves native-resolution PNGs and an encoder-ready NPZ loadable with `allow_pickle=False`. It refuses to overwrite an existing capture. Saving PNG/NPZ synchronously can be slower than the sensor loop; the CLI reports elapsed capture time and never promises its requested FPS. For a long high-rate recording, put a bounded recorder queue/video encoder behind this read interface and log dropped frames. For immediate inference, use `obs.inputs` directly without saving or HTTP.

The simulator stores additional joint velocities and end-effector poses in metadata. The physical capture path currently supplies joint/gripper positions only; it does not claim measured velocity, force, depth, tactile data, or Cartesian pose.

## Verification

Run all checks in `mini-vla`:

```bash
python -m unittest discover -s tests -v
```

Coverage includes real MuJoCo rendering, capture round trips, goal/history masks, stale/skew rejection, camera RGB conversion/rotation/failure cleanup, bimanual state ordering, calibration mismatch rejection, servo errors, and assertions that the mocked motor transport uses read operations only. Physical validation remains pending connection of the actual robot and cameras.
