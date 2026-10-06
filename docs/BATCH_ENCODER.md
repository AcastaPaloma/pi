# A batch of 128 robot observations, directly in Python

The interface for this stage is `robot.observations()`. It returns NumPy images and proprioception immediately after any initial render/cache preparation. You implement the encoder and its training objective. There is no policy, action execution, HTTP call, or per-batch capture-file writing in this interface.

## See 128 simulated scenes

```python
from pi_sim import RobotBatch

with RobotBatch(batch_size=128, image_size=224, seed=0) as robot:
    obs = robot.observations()

    images = obs["images"]                  # uint8 [128, 3, 224, 224, 3]
    proprio = obs["proprioception"]          # float32 [128, 14]
    prompts = obs["task"]                    # strings [128]
    camera_order = obs["camera_names"]        # left_wrist, right_wrist, front

    # In a notebook:
    display(robot.grid(obs))

    # Read again: same scenes, already-rendered images, fresh writable arrays.
    obs = robot.observations()

    # Explicitly replace the static scenes with new samples when desired.
    robot.resample()
    obs = robot.observations()
```

Outside a notebook, replace `display(...)` with `robot.grid(obs).save("robots.png")`.

The batch contains 128 separate **static bimanual scenes**, each with two UR5e arms, a table, and an object. It uses one shared model/rendering context and independent pose arrays, avoiding 128 copies of robot meshes and rendering contexts. Scene reads never advance physics. The initial home pose is settled once; `resample()` changes joint/object poses explicitly. Randomized poses are for testing encoder inputs, not collision-validated folding demonstrations.

Images are rendered lazily, once per scene generation. Repeated reads reuse those images. Rendering a fresh generation still requires 128 × 3 camera renders in stock MuJoCo; this is not a GPU-vectorized simulator. For states alone, `robot.observations(images=False)` never renders. `copy=False` returns read-only cached views; the default returns writable copies suitable for handing to your own tensor code. Keep rendering on the creating thread and close the batch afterward.

Run the working example and generate the preview:

```bash
conda activate mini-vla
python examples/batch_observations.py
```

It writes `outputs/sim_128_robots_224.png` and timing measurements. The example also accepts `--size 448` and `--batch-size 128`.

## Train from real folding observations

For the actual SO-101 encoder, prefer recorded SO-101 observations. Batch size 128 means 128 recorded observation samples; it does not require 128 physical robots or simulators.

A small **153-observation demo cache** is already prepared under `outputs/folding_encoder_demo_224`: every tenth frame of train episodes 0 and 1. It exists to verify shapes, speed and integration, not to serve as your full training set.

```python
from pi_data import FoldingBatch

robot = FoldingBatch("outputs/folding_encoder_demo_224", batch_size=128)

obs = robot.observations()
images = obs["images"]                   # uint8 [128, 3, 224, 224, 3]
proprio = obs["proprioception"]           # float32 [128, 12]
prompts = obs["task"]                     # strings [128]
episode_ids = obs["episode"]
frame_ids = obs["frame"]

# Another random batch; no video decoding or HTTP in this call.
next_obs = robot.observations()

# Show exactly the sampled batch in a notebook.
display(robot.grid(obs))
```

The cache is uncompressed `.npy` arrays read through memory mapping. Random indexing yields owned, writable NumPy batches. Video is decoded **once, sequentially per video file**, during preparation. Frame/state alignment uses each camera's recorded file offset and each row's timestamp. Each batch samples without replacement; samples can recur across batches. You may pass explicit cache row indices, e.g. `robot.observations(indices=[0, 3, 10])`, to reproduce particular observations.

To prepare the full training and validation splits on your Lambda SSD:

```python
from pi_data import prepare_observation_cache, FoldingBatch

# One-time preparation. Existing output directories are never overwritten.
prepare_observation_cache("cache/train224", split="train", image_size=224,
                          progress=print)
prepare_observation_cache("cache/val224", split="validation", image_size=224,
                          progress=print)

train = FoldingBatch("cache/train224", batch_size=128, seed=0)
validation = FoldingBatch("cache/val224", batch_size=128, seed=1)

obs = train.observations()  # Feed the desired fields to your own encoder.
```

The source split remains 450 training episodes / 50 validation episodes. Cache preparation refuses episodes from a different split. Defaults retain every frame; `stride` and `limit` explicitly reduce the prepared set if desired. Use broad episode coverage for training. A small, correlated cache repeatedly sampled does not become more diverse by increasing batch size.

The full three-camera image cache across both splits requires approximately **84.5 GB at 224px**, or **338 GB at 448px**, plus the 4.48 GB compressed source and small metadata/state files. This trades storage for CPU decode work. On first access, uncached pages still need an SSD read; warm RAM/cache timings do not guarantee full-dataset throughput. The preparation routine checks local disk space, verifies video timestamps, and publishes only a completed cache. It never starts a cloud instance.

To reproduce the small prepared example:

```bash
python examples/batch_observations.py --source real --prepare-demo
```

## The encoder contract

Both sources expose the same core fields: `images`, `proprioception`, `task`, `camera_names`, `state_names`, `state_units`, and `source`. Each image is RGB with channels last; normalize/reorder according to your model. No tensors, device transfers, model weights, learned heads, losses, optimizers, or PyTorch code are added here.

| Source | State shape for batch 128 | Units / external camera |
| --- | --- | --- |
| Existing UR5e simulation | `[128,14]` | Six arm angles in radians + 0–1 gripper per arm; `front` |
| Real SO-101 dataset | `[128,12]` | Five arm angles in degrees + 0–100 gripper per arm; `right_front` |

These are distinct embodiments. Do not feed the simulation's 14D state into a 12D SO-101 projector without an explicit design for that difference. For your SO-101 encoder, the real-data batch is the matched input source. Text is a generic observation prompt for the static sim and the dataset's native task prompt for real samples; neither supplies rich language supervision.

Current images/state/text are the default batch. This avoids paying for unused histories, goals, or action chunks while you build the encoder head. The existing `FoldingDataset.arrays(...)` remains available for labeled action chunks and future-image goals if you later choose an objective requiring them. The existing physical `Observer` also now supports `observer.observations()`, returning one live sample's input dictionary without saving files; a live sample has no batch dimension.

At batch 128 and three cameras, RGB uint8 input occupies **55.1 MiB at 224px** or **220.5 MiB at 448px**. Float32 images take four times that before model activations. A batch that loads comfortably into RAM is not a claim that a particular trainable vision backbone fits batch 128 on a GPU. If you train only a fusion/projection head, freezing the backbone and optionally caching its features can reduce later compute; feature extraction/training stays in your own model code.

## Measured locally

The example records setup time, first `.observations()` time, and the median of five subsequent calls. On this Apple Silicon Mac, the initial 224px run produced a 128×3-camera simulation batch in about **2.8 seconds**, followed by cached copies in about **5 ms**. The real demo cache returned random 128-sample batches in about **3 ms** once warm. These measure only CPU NumPy input delivery, not fresh rendering at that warm rate, GPU transfers, or model training. Re-run the example to benchmark your machine and full cache.

The 448px path was also exercised with batch 128 and all three cameras: about **2.83 seconds** for the first simulation render and **15 ms** for cached copies; the real cache returned warm batches in about **14 ms**. A 153-observation 448px demo cache is ready at `outputs/folding_encoder_demo_448`. Its initial preparation took about 2.93 seconds from already-downloaded source videos. Benchmark JSON and both-resolution preview grids are in `outputs/`.

Generated previews: `outputs/sim_128_robots_224.png` and `outputs/real_128_robots_224.png`. The batch tests verify cache reuse/invalidation, independent seeded scene states, no time advancement during reads, camera/state alignment across video offsets, split restrictions, failed-cache cleanup, and writable-batch isolation.
