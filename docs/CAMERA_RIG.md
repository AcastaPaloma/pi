# Matching the simulated and physical camera rig

The simulator has three sensor cameras in the same order as physical capture:
`overhead`, `left_wrist`, `right_wrist`. The wrists attach to each follower's
wrist-roll camera mount and move with it. The overhead camera hangs from a gantry
0.70 m above the table, looking straight down, and plays the role of the paper's
base view. There is no front camera in this rig.

All three are **Arducam B030401** modules: the Sony IMX708 sensor of the Raspberry
Pi Camera Module 3 on a USB 2.0 UVC adapter, autofocus from 10 cm to infinity,
75° diagonal lens (61° horizontal, 50° vertical), 32×32 mm board that fits the
official SO-101 32×32 UVC wrist mount. The fixed-focus wide variant (B0305) focuses
only from 1.5 m, which rules it out for wrist use. The shipped
[rig profile](../pi_sim/assets/so101_camera_rig.json) records the module for every
view at 640×480, 30 fps.

**Optics and mounting poses remain `calibration_status: "nominal"`.** The pinhole
uses the 61° horizontal figure at 640×480 (fx = fy ≈ 543 px), which implies 47.8°
vertical against the listed 50°; distortion is an unmeasured zero placeholder, and
autofocus shifts the effective focal length with focus distance. Replace the
intrinsics with a checkerboard calibration of each unit at its working focus, and
lock the focus through the UVC focus control for the overhead camera.

## See and capture all three views

```bash
./play.sh --view overhead
# Press V to cycle: scene, overhead, left_wrist, right_wrist.

conda activate mini-vla
python -m pi_observe sim --output outputs/camera_rig_capture
```

The capture directory contains native `overhead.png`, `left_wrist.png`,
`right_wrist.png`, `encoder.npz`, and `metadata.json`, under `000000/`.
The initial native frames are **640×480 RGB**; encoder tensors retain the
requested square size. Both simulation and physical capture use the same
bilinear resize. Rendering directly into a square camera view would change the
field of view; the new path renders the native lens image first.
The interactive viewer shows MuJoCo's pinhole preview. Captures and
`render_camera()` additionally apply any configured distortion and image rotation.

```python
from pi_sim import RobotEnv
from pi_observe import Observer, SimSource

with RobotEnv(camera_rig="configs/so101_camera_rig.local.json") as env:
    env.reset(seed=0)
    observer = Observer(SimSource(env), task="Pick up the red cube")
    capture = observer.read()                       # Reads every camera; no physics step.
    native = capture.images["left_wrist"]          # [H, W, 3], RGB uint8
    encoder = capture.inputs["images"]             # [3, 448, 448, 3]
    calibration = env.camera_calibration("left_wrist")
    # Existing Gym/batch APIs return resized images, preserving native optics:
    square = env.render_camera("left_wrist")
    native = env.render_camera("left_wrist", native=True)
```

## Use one measured profile for sim and hardware

Copy `pi_sim/assets/so101_camera_rig.json` to
`configs/so101_camera_rig.local.json`. Edit each camera independently:

| Field | Meaning |
| --- | --- |
| `hardware_model` | Exact purchased camera **and lens variant** |
| `width`, `height` | Actual capture mode, before optional rotation |
| `fps` | Hardware request; sim reads render the current state |
| `intrinsics` | `fx`, `fy`, `cx`, `cy` in pixels at that exact resolution |
| `distortion` | OpenCV Brown-Conrady `[k1, k2, p1, p2, k3]`; five zeros for pinhole |
| `rotation` | 0/90/180/270 degrees counterclockwise, applied after lens rendering/capture |
| `pose.pos` | Optical center in meters in the parent body's frame |
| `pose.quat_wxyz` or `pose.xyaxes` | Camera orientation; supply exactly one |
| `calibration_status` | Record `measured` once optics **and** mounting pose have been checked |
| `optics_note`, `reference_url` | Provenance and remaining assumptions; update after calibration |

Use a checkerboard calibration at the actual resolution/focus to obtain
intrinsics and distortion. Marketing field-of-view numbers alone do not specify
principal point, distortion, crop or focus-dependent optics. The simulator
supports unequal focal lengths and an off-center principal point. A nonzero
distortion profile uses OpenCV inverse ray mapping and an overscanned pinhole
render, so lens warping does not introduce artificial black borders. Install
`pip install -e '.[capture]'` if OpenCV is absent. Fisheye and depth sensors are
not implemented by this RGB pipeline.

Pose axes are MuJoCo's camera convention: X right, Y up, view along **−Z**.
The `overhead` pose is world-relative; world Z=0 is the tabletop.
The wrist poses are relative to `left_camera_mount`/`right_camera_mount`, fixed
to their grippers. Use the optical center, rather than the outer housing center.
For a measured OpenCV camera-to-parent rotation, convert with
`R_mujoco = R_opencv @ diag(1, -1, -1)` before extracting a WXYZ quaternion.
An OpenCV parent-to-camera extrinsic must first be inverted.

Select the file in both paths:

```bash
./play.sh --camera-rig configs/so101_camera_rig.local.json
python -m pi_observe sim --camera-rig configs/so101_camera_rig.local.json
```

In `configs/so101_observe.local.json`, set
`"camera_rig": "so101_camera_rig.local.json"` and keep the three camera device
IDs under `cameras`. Rig paths are relative to the hardware config. Width,
height, FPS and rotation are taken from the shared profile; conflicting
duplicate settings are rejected. Physical capture also rejects a driver's
fallback frame size that would invalidate the shared intrinsics. Existing
configs without `camera_rig` still work with best-effort camera mode requests.
`RobotBatch(..., camera_rig=...)` uses the same profile and preprocessing.

Sim capture metadata stores every camera's intrinsics/distortion and current
`world_from_camera_opencv_before_rotation` 4×4 pose. OpenCV axes are X right,
Y down, Z forward. Calibration matrices describe the native feed **before**
rotation and encoder resizing; account for both operations when projecting into
encoder tensors. Physical metadata records the selected profile; it does not
claim to measure the cameras' live world poses.

## Remaining transfer gap

Matching camera identities, mounting poses, resolution, lens projection,
distortion, rotation and preprocessing removes avoidable observation
differences. Exact matching still requires measurements from the assembled
rig, especially the wide lens's distortion and actual capture-mode crop.
Housings/gantry are visual approximations; the wrist
mount's mass/contact model remains the upstream model. Lighting, exposure,
white balance, focus blur, sensor noise, rolling shutter, USB latency and real
exposure timing are not replicated. Sim captures share one frozen state;
the real reader uses independently timed camera streams. The profile's 30 FPS
does not change the simulator's 50 Hz control interval or emulate frame drops.

Camera regression checks project a test object through unequal/off-center
intrinsics and a distorted lens, verify wrist attachment, ensure fixed cameras
stay fixed, and compare native-to-encoder preprocessing across interfaces:

```bash
python -m unittest discover -s tests -p test_cameras.py -v
```
