# Batches for the SO-101 encoder

```python
from pi_sim import RobotBatch

with RobotBatch(128, image_size=224, seed=0) as robot:
    obs = robot.observations()
    images = obs["images"]           # uint8 RGB [128,4,224,224,3]
    state = obs["proprioception"]     # float32 [128,12]
    print(obs["camera_names"])       # left_wrist, right_wrist, front, overhead
    robot.grid(obs, camera="overhead").save("outputs/so101_batch.png")
    robot.resample(seed=1)           # explicitly change static scene poses
```

Each row is an independent static two-SO-101 scene. One model/rendering context renders stored poses sequentially. Reading never advances physics; unchanged renders are cached. Randomized poses are for input development, not collision-validated expert demonstrations. Call on the creating thread and close afterward. `images=False` avoids rendering; `copy=False` returns read-only cache views.

The state is five model-origin arm angles in degrees plus 0–100 gripper opening per arm, left then right. A matching dimension/unit label does not establish identical hardware calibration.

```bash
conda activate mini-vla
python examples/batch_observations.py
python examples/batch_observations.py --source real --prepare-demo
```

The real example now uses the pinned Project-IRA manipulation dataset. Its images have **two views**, and state has **six values**. The [dataset and experiment guide](SO101_MANIPULATION.md) explains source choice, shapes, units, missing-view/arm masks and episode splits.

```python
from pi_data import ManipulationDataset, ObservationBatch, prepare_observation_cache

ds = ManipulationDataset(dataset="multitask")
prepare_observation_cache("cache/pilot224", dataset=ds, image_size=224,
                          episodes=ds.episode_ids("train")[:2], stride=10)
real = ObservationBatch("cache/pilot224", batch_size=32)
obs = real.observations()            # [32,2,224,224,3], state [32,6]
```

Decoded caches are memory-mapped `.npy` arrays; preparation decodes each source video sequentially and validates timestamp alignment. Cache sampling returns writable copies and performs no video decoding or network access. Choose one split per cache; normalization belongs to train episodes only. `FoldingBatch` remains an alias for the same generic reader, and `FoldingDataset` remains available for the older folding experiments.

At 128×224 px, four-view simulation RGB occupies about **73.5 MiB**; two-view real RGB occupies **36.75 MiB**. Float32 images use four times that memory, before model activations. Full decoded datasets can require hundreds of GB; start with explicit episode/stride/limit choices. Warm cache timings measure NumPy delivery, not fresh rendering, GPU transfers or training speed. Re-run the example for current hardware measurements; older UR5e benchmarks are not representative of this migrated scene.
