# SO-101 manipulation: data, printed objects, cameras

The local simulator now has **two SO-101 follower arms**, each with five arm joints and a gripper. It uses the [Robot Studio SO-101 model from MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101), including the wrist-camera mount and contact geometry. The bases are 36 cm apart, pointing into the table. Both arms can be commanded; the starter task scores the left arm.

```bash
conda activate mini-vla
python -m pi_sim --headless --task pick_place --object-shape cube --object-color red
./play.sh --demo --task pick_place --object-shape cylinder --object-color blue
python -m pi_observe sim
```

Objects are 30 mm cubes or 30 mm diameter × 30 mm cylinders. The goal disk is 80 mm across. [Ready-to-print STL files](../printables/README.md) include 24/30/36 mm objects and the disk. Filament supplies the color. These primitives approximate a reproducible tabletop experiment; they are not the original datasets' exact objects.

## Which datasets to use

Checked on 2026-10-07 against publisher cards and immutable `meta/info.json` revisions. Episode/frame counts below use metadata, rather than the Hugging Face viewer's sometimes misleading row count.

| Dataset | Verified size | Fit for this project | Cameras / license |
| --- | --- | --- | --- |
| [Project-IRA multitask](https://huggingface.co/datasets/Project-IRA/TPSoSe2026_Dataset_Full_Merged_Final_LeRobot_SO101_V1) | 930 episodes, 844,208 frames, 30 Hz; four task families, 93 prompt phrasings | **Start here for language-conditioned manipulation.** Color sorting is 460 episodes; desk cleanup is 250. The other tasks are dice pouring (120) and ball handoff (100). Colored blocks, pads and cups are practical prints. | `wrist_left`, `desk_view`; CC BY-SA 4.0 |
| [PHI cubes/cylinder](https://huggingface.co/datasets/BrutalCaesar/phi_so101_cubes_cylinder_v1) | 120 episodes, 66,873 frames, 30 Hz | **Closest match to simple printed objects.** Pick cubes/cylinder into a box. One shared task label, so it is object variation rather than language-rich multitask supervision. | `wrist`, `front`, `top`; Apache-2.0 |
| [TakuyaHiraoka diverse objects](https://huggingface.co/datasets/TakuyaHiraoka/so101_pick_diverse_objects) | 962 episodes, 284,147 frames, 30 Hz at revision `869b81c2dc4a2b854e029133131b9d8b6332ef7d` | Publisher reports approximately 70 objects. Useful supplementary grasp diversity; many household objects are less reproducible as simple prints. One pickup skill. | `wrist`; check publisher terms before mixing |
| [gpudad simulated cube pickup](https://huggingface.co/datasets/gpudad/so101_pick_cube) | 10,993 episodes, 1,456,901 frames, 30 Hz at revision `0168e7bfb4d6c97dc1bfa085fa2fc1b78834993f` | Cheap simulation pretraining/ablation. One cube-to-bin skill; quantity alone does not establish real-world generalization. | `front`, `wrist`, `overhead`; Apache-2.0 |

The last dataset's actual metadata lists **512×512 CHW images** and alphabetically ordered joint names, unlike some card descriptions. It needs its own explicit ordering/layout/unit adapter and is not enabled in the local loader.

Project-IRA is the strongest initial multitask fit among these candidates, but its publisher explicitly describes one desk, lighting setup and object collection. Printed substitutes change appearance, contact friction and mass. **Plan to fine-tune on your own rig**, rather than expect zero-shot transfer from public data. The dataset's share-alike terms matter if redistributing adapted datasets.

All four sources above are **single-arm 6D data**. They do not provide 12D demonstrations for coordinated two-arm manipulation. Use an arm-specific action head/masked action loss during pretraining, then collect genuinely bimanual demonstrations. Never invent stationary right-arm labels by padding six zeros and treating those as observed actions.

## Working data interface

The default `pi_data` source is now `multitask`; `cubes` is a second pinned option. The folding source remains explicitly selectable with `--dataset folding`.

```bash
python -m pi_data --dataset multitask prepare --episode 0
python -m pi_data --dataset multitask sample --episode 0 --frame 100 --size 224
python -m pi_data --dataset multitask serve
python -m pi_data --dataset cubes prepare --episode 0
```

```python
from pi_data import ManipulationDataset, ObservationBatch, prepare_observation_cache

ds = ManipulationDataset(dataset="multitask")
x = ds.arrays(0, 100, size=224, horizon=50, history=6)
# images [2,224,224,3], proprioception [6], actions [50,6]
# camera_names preserves wrist_left/desk_view; action_valid masks episode padding.

prepare_observation_cache("cache/manipulation224", dataset=ds, image_size=224,
                          episodes=ds.episode_ids("train")[:2], stride=10)
batch = ObservationBatch("cache/manipulation224", batch_size=32).observations()
```

`prepare` downloads whole video shards, not just one episode's seconds. Project-IRA's first two video shards total roughly 413 MB. `--all` downloads the entire source. Decoded image caches grow quickly: Project-IRA's full two-camera uint8 cache is approximately 254 GB at 224 px (both splits together), before filesystem overhead. Use explicit small episode selections while developing the encoder.

The loader reads **all episode metadata shards**, verifies numeric frame/time ordering, preserves camera offsets, prevents action chunks crossing episode boundaries, and computes normalization from train episodes only. A deterministic 90/10 episode split is supplied for development; it is not a held-out-object or held-out-session experiment.

**Local audit:** all 930 Project-IRA episode intervals passed the metadata duration check. PHI episode 62 has video intervals 14 frames longer than its numeric episode at pinned revision `014a212f6a30d68b39015546a9590c0927b8b9fa`. The `cubes` adapter explicitly excludes it and reports that exclusion in sample/API metadata, leaving **119 usable episodes**. This is a metadata screen, not a full success-quality audit. Episode 0's numeric rows and aligned frames from all three PHI cameras were downloaded and decoded into `outputs/so101_cubes_sample.npz`.

Each source retains its **native motor values**. These public cards do not establish every recording's normalization/calibration; the adapter marks units `source_native_unverified`. Do not assume similarly sized values are degrees or send them directly to a physical arm. Audit recorder configuration and calibrated zero/sign/range before combining sources or replaying actions. The simulator's degree convention uses the Menagerie joint origin; it is not an independently measured hardware calibration.

The π₀.₇ reimplementation expects up to four views (three here) and 12D proprioception. A two-view/6D public sample needs an explicit model adapter: preserve view identities, mask absent views and absent arm dimensions, and use a source-specific state projector or normalization. Repeating images to manufacture four views supplies no extra supervision. The loader preserves the original evidence instead of hiding these differences.

## Where the three cameras go

**Three cameras total across the two follower arms**, all Arducam B030401 IMX708 USB autofocus modules:

| Name in our rig | Physical mount | Purpose |
| --- | --- | --- |
| `overhead` | Rigid crossbar/gantry 0.70 m above the workspace center, looking straight down; or the official printed base-connected overhead mount | The base view: object color, layout, sorting targets and inter-arm coverage |
| `left_wrist` | Left follower's wrist-roll camera mount, just behind and above the fingertips | Close view of the grasp and left-arm contact |
| `right_wrist` | Same mount on the right follower | Close view of right-arm contact and handoffs |

The [official SO-101 wrist adapter](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Optional/SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module) uses hex-nut recesses in the **wrist-roll element**, two M3×8 screws/nuts, and four M2 screws for a 32×32 mm USB camera board; the B030401 board is 32×32 mm. Print one per follower. Keep the moving jaw unobstructed and add cable slack/strain relief across wrist movement.

The [official overhead assembly](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Optional/Overhead_Cam_Mount_Webcam/README.md) connects the camera support to one or two printed follower bases; a separate rigid gantry is another option. At 0.70 m the 61° horizontal field covers about 0.82 × 0.63 m of tabletop, enough for both workspaces. Lock the overhead autofocus with the UVC focus control; the wrists use autofocus (10 cm to infinity). Capture at 640×480, 30 fps; keep exposure and white balance fixed for color experiments. The [camera guide](CAMERA_RIG.md) records the profile; its optics are nominal until calibrated. Buy a fourth module as the spare.

[The example hardware configuration](../configs/so101_observe.example.json) uses these three names in the same order as simulation. The physical observation reader remains read-only and requires already calibrated arms.

## Train for generalization, not just repeated placement

My suggested local curriculum:

1. Print baseline cubes/cylinders in four matte colors and several colored disks. Match object dimensions across colors. Keep 24/36 mm variants for later size shifts.
2. Start one follower with pick-and-place, push-to-pad, color sorting, and two-object stacking. Collect approximately 50–100 varied successful demonstrations per task as an initial pilot, then add data where failures concentrate. This is a collection target, not a promised sample-complexity threshold. The current simulator implements reach and pick/place only; push/sort/stack need local task definitions and demonstrations. `python -m pi_sim.record` produces π₀.₇ training episodes with subtask segments, mistake retries and visual randomization.
3. Randomize object position/orientation, destination position, distractor count, and task instructions. Train each color across several destinations, so “red” does not become a synonym for one location. Include corrections and recovery behavior.
4. Split by collection session first, then reserve color–shape–destination combinations. Test known colors/shapes in new combinations, unseen sizes, new layouts, and a second background/lighting condition separately. Unseen-color evaluation is harder than novel combinations and should be reported separately.
5. Evaluate at least 20 physical rollouts per condition with a fixed success definition. Report per-task success and failure type. Neighboring frames from the same demonstration must not fall on opposite sides of a train/test split.
6. Only then add two-arm handoff and cooperative placement, recording all 12 motor values and all three cameras together. Keep initial single-arm pretrained supervision separate from coordinated action supervision.

For a quick baseline, fine-tune a pretrained language-conditioned LeRobot policy such as SmolVLA on the public source, then on your printed-object recordings. ACT is a useful control baseline for a narrow task. No policy training or hardware motion was run as part of this repository migration.
