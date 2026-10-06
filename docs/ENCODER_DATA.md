# Start the low-level encoder here

Use **LeHome real garment folding**, specifically [`lehome/dataset_challenge_real/four_types_merged`](https://huggingface.co/datasets/lehome/dataset_challenge_real/tree/main/four_types_merged). It gives you real bimanual SO-ARM101 folding experience with camera, state, and action records. Cloth folding is related to origami through grasping, aligning edges, and coordinated folding; it does not teach paper creasing or establish origami performance.

The data loader and HTTP API are implemented in `pi_data/`. They use NumPy, PyArrow, PyAV, and FastAPI. No encoder, action expert, world model, or PyTorch training loop is included.

## 1. Get a sample

The data dependencies and first episode's three video files are already installed/cached in your `mini-vla` environment. From this repository:

```bash
conda activate mini-vla
python -m pi_data sample --episode 0 --frame 100 --goal-seconds 2
python examples/encoder_inputs.py
```

This writes `outputs/encoder_sample.npz`. Open it with `np.load(path, allow_pickle=False)`. `outputs/encoder_preview.png` shows the verified current/goal images from that example.

For a fresh machine, install the data extra. This does not install PyTorch:

```bash
python -m pip install -e '.[data]'
python -m pi_data prepare           # metadata, actions, three episode-0 video files; ~450 MB
python -m pi_data prepare --all     # complete selected subset; ~4.48 GB
```

Downloads use the normal Hugging Face cache (`HF_HOME` can relocate it), pinned to revision `5a28286fb60db8bf9fab9a552a001645722752cd`. Files are fetched lazily unless prepared. The repository contains duplicate garment-specific folders; only download `four_types_merged/`. For an offline extracted copy: `python -m pi_data --root /path/to/four_types_merged serve`.

## 2. Start the API

```bash
python -m pi_data serve
```

Open **http://127.0.0.1:8000/docs** for interactive endpoints. Use `--port 8001` if needed. These endpoints replay recorded observations; they do not read your physical robot or the existing MuJoCo scene.

Let `F = /v1/episodes/{episode}/frames/{frame}`. All endpoints below use GET:

| Endpoint | Returns |
| --- | --- |
| `/v1/dataset` | Source, schema, cameras, counts, missing annotations |
| `/v1/episodes?split=train` | Training episode IDs and lengths; also `validation` |
| `F/observations` | All three camera URLs, timestamp, proprioception |
| `F/cameras/{camera}.png` | RGB image; `size=448` default |
| `F/proprioception` | 12 joint/gripper positions, names, per-dimension units |
| `F/text` | Native task prompt, control mode, absent annotations as `null` |
| `F/subgoal` | Future-frame index, delay, all three goal-image URLs |
| `F/actions` | 50×12 action labels and padding mask |
| `F/sample` | The above plus optional observation history |
| `F/sample.npz` | Actual image arrays, state, text, goals, actions, masks |
| `/v1/normalization` | State/action mean and scale fitted on training episodes only |

`sample` and `sample.npz` accept `history=1..6`, `horizon=1..200`, `goal_seconds=0..4`, and `seed`. Without `goal_seconds`, goals are sampled reproducibly from the available next 0–4 seconds. `sample.npz` also accepts `size`. JSON image URLs are relative to the server origin.

```bash
curl 'http://127.0.0.1:8000/v1/episodes/0/frames/100/sample?goal_seconds=2'
curl 'http://127.0.0.1:8000/v1/episodes/0/frames/100/sample.npz?goal_seconds=2' \
  -o outputs/from_api.npz
```

The HTTP API is convenient for inspection and integration. Use the Python interface in a training data loader to avoid serial HTTP requests and PNG round trips:

```python
from pi_data import FoldingDataset

ds = FoldingDataset()
episode = ds.episode_ids("train")[0]
x = ds.arrays(episode, 100, history=1, goal_seconds=2)
# x contains NumPy arrays; feed the input fields to your own encoder.
```

Each loader worker should create its own `FoldingDataset`. Prepare all videos on the training machine first. This reference loader seeks compressed video on demand; profile data loading before scaling GPUs. Sequential video decoding or cached frame/features can be added if random seeking becomes the bottleneck. Caching vision features only works while that vision encoder is frozen.

## 3. Encoder contract

Shapes below are per example, before adding a batch dimension. Camera order is always **left_wrist, right_wrist, right_front**.

| Field | Shape / meaning |
| --- | --- |
| `images` | `[3,448,448,3]` RGB uint8, current observation |
| `proprioception` | `[12]` float32, native joint/gripper positions |
| `task` | Unicode scalar: `Fold the Garment` |
| `control_mode` | Unicode scalar: `joint_position`, adapter-supplied |
| `subgoal_images` | `[3,448,448,3]` RGB uint8, optional visual conditioning |
| `history_images` | `[T,3,448,448,3]`, oldest to current, 1-second spacing |
| `history_proprioception` | `[T,12]` |
| `history_valid` | `[T]` bool; ignore repeated start padding |
| `actions` | `[50,12]` float32, supervised labels starting at current row |
| `action_valid` | `[50]` bool; ignore repeated end padding |

Here `T` includes the current observation: `history=1` means current only. The returned `images` is also the last entry of `history_images`; do not feed it twice. Images are directly resized to a square. Convert layout/range and apply normalization required by your chosen pretrained vision encoder. There is no hidden image normalization or state/action conversion.

The 12 components are left then right, each ordered:

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

The [released real-data transform](https://github.com/IliaLarchenko/lehome_solution/blob/main/src/lehome_solution/training/real_data_transforms.py) documents arm positions in degrees and grippers in LeRobot's 0–100 closed-to-open convention. Both states and action targets retain those native values. Fit standardization using `/v1/normalization`: `(value - mean) / scale`. Do not use the source's all-episode statistics for a strict held-out evaluation. These are position targets, not joint velocities, torques, or Cartesian actions; there are no force/tactile labels.

**Only inputs enter the encoder.** Action targets, masks, episode IDs, and sampled goal timestamps remain supervision/bookkeeping. A useful first architecture is a shared pretrained image encoder with camera identity embeddings, a pretrained text encoder, a projection of the 12D state, and a small fusion block. Add goal-image role embeddings so current and desired views are distinguishable. Freeze the language encoder initially: one repeated task string provides no language-diversity supervision.

An encoder by itself has no action-learning objective. To assess whether its representation contains useful control information, attach a small temporary head predicting the normalized action chunk and train with a masked regression loss. That is a baseline, not the paper's complete objective. π0.7's VLM uses tokenized action supervision and its separate action expert uses flow matching. You can implement the encoder interface now and choose that training objective afterward.

## 4. How close this is to π0.7

The [π0.7 paper, Sections VI-B/C](https://arxiv.org/html/2604.15483v2) conditions on multiple camera views, proprioception, language, history, and optional goal images, with 448×448 images and 50-action chunks. This adapter exposes those data roles for your SO-101 pair; it does not reproduce the complete model or training recipe.

- The paper samples real subgoals from subtask ends 25% of the time and uniformly 0–4 seconds ahead 75% of the time, and also uses generated goals. We implement the future-frame branch, restricted to available frames in the same episode. LeHome has no subtask boundaries, so we cannot honestly implement the end-of-subtask branch.
- At training time, future demonstration images give useful goal conditioning **without a world model**. At deployment those future observations do not exist. Supply a human-selected goal image, a curated reference goal, or an independently chosen reference-sequence subgoal. Match camera viewpoints and test transfer; reference images are not guaranteed reachable from every state.
- The dataset provides one generic task string. Subtask text, quality, mistake, and speed labels are absent and returned as `null`. Do not invent ground truth for them. There are no paper-style success/quality annotations to filter by.
- The adapter returns optional history, but implements neither MEM history compression nor the paper's attention masks or conditioning dropout. Add those in the encoder. To follow the paper's optional-goal training, sometimes mask the goal images; do not force them into every training example.
- At 20 Hz, 50 actions cover **2.5 seconds**. This differs from a 50 Hz robot's 1-second chunk.

The split is 450 train / 50 validation episodes, deterministically selected before sampling. Future goals and action chunks never cross episode boundaries. Near the end, fixed goals clamp to the last frame, and action padding is explicitly masked. A zero-second goal is allowed, as in the paper's 0–4-second interval; `strictly_future` identifies that case. Validation with oracle future goals measures goal-conditioned prediction, not autonomous performance. Report a no-goal baseline separately. The split does not prove generalization to unseen garment instances or environments.

## 5. H100 plan

**Start with one H100 80 GB.** For a pretrained 100–400M vision encoder, frozen/small text encoder, and trainable state/fusion layers, this is a sensible development allocation. Begin with three current images, BF16, microbatch 8, and gradient accumulation toward effective batch 64. If goals double the image input or you unfreeze more weights, measure memory and reduce the microbatch. These are starting settings, not measured capacity guarantees.

| Scope | Initial allocation |
| --- | --- |
| Smaller encoder/fusion training | 1× H100 80 GB |
| LoRA adaptation of a paper-scale 4B VLM backbone | Try 1× H100 80 GB; microbatch 1–2, checkpointing, short image context |
| Full 4B backbone fine-tuning with Adam | Budget 2–4× H100 80 GB with sharding; profile before reserving |

A conventional mixed-precision Adam setup can require about 16 bytes per trainable parameter: 4B parameters already imply roughly 64 GB for weights/gradients/optimizer state before activations. Actual implementations differ. That is why full fine-tuning changes the allocation. Image history also matters: this API's six-time-step setting gives 18 observation images plus three goal images per example before any encoder compression.

[Lambda's posted rates](https://lambda.ai/instances), checked October 5, 2026, list a single **H100 PCIe 80 GB at $3.29/hour**, with 26 vCPUs, 225 GiB RAM, and 1 TiB SSD; a single **H100 SXM 80 GB is $4.29/hour**. Start with whichever is available; the PCIe instance is enough for this initial allocation. Availability, tax, and actual credit terms can differ.

Reserve an initial **10–20 GPU-hour experiment budget**: about **$33–66 PCIe** or **$43–86 SXM**, before tax. This is a spending envelope, not a runtime prediction. Benchmark 200 optimizer updates after warm-up with actual images, goal frequency, batch/accumulation, and trainable layers, then use:

```text
hours = optimizer_updates × measured_seconds_per_update / 3600
cost  = hours × instance_hourly_price
```

For illustration, 20,000 updates at 0.25 / 1 / 3 seconds per update take 1.4 / 5.6 / 16.7 hours, respectively. These rates are hypothetical. No H100 benchmark or cloud job has been run. Keep at least 50 GB free for the compressed dataset, model downloads, caches, and a few checkpoints; full decoded 448px RGB frames for all three cameras alone would occupy about 338 GB.

If selecting Gemma 3 for paper similarity, its [stock Transformers preprocessing](https://huggingface.co/docs/transformers/model_doc/gemma3) uses a different native image resolution. Do not assume that this adapter's 448px images make a stock checkpoint equivalent to the paper's modified vision encoder.

## 6. Toward your physical SO-101 pair

The [LeHome reference system](https://github.com/IliaLarchenko/lehome_solution) uses bimanual SO-ARM101 hardware. That is a much closer action schema than a dexterous-hand origami dataset. Start with one wrist camera per arm plus a fixed external camera and preserve this same observation/label contract. Match joint order, calibration, units, timestamps, camera identity, and action semantics when recording your own paper-folding demonstrations. Fine-tune on those demonstrations before expecting origami behavior.

The existing `pi_sim` scene uses UR5e arms and a different 14D joint schema. **LeHome's 12D action labels cannot be sent directly to it.** For direct simulation/physical observations, use the separate [observation interface](OBSERVATIONS.md). It now includes a read-only SO-101 capture path; hardware validation remains pending connection of your robot.

## Verified source and checks

The pinned [metadata](https://huggingface.co/datasets/lehome/dataset_challenge_real/blob/5a28286fb60db8bf9fab9a552a001645722752cd/four_types_merged/meta/info.json) records 500 episodes, 187,135 frames, 20 Hz, and three cameras: two 640×480 wrists and a 1280×720 `right_front` camera. That is 2.60 trajectory-hours. The dataset repository declares Apache-2.0. Duration/index metadata was checked across all episodes; this is not a complete visual or success-quality audit. Sample images and first/last frame alignment of episodes 0 and 1 were decoded and checked.

Offline tests exercise camera-specific video offsets, action alignment, episode boundary masks, deterministic goal sampling, train-only statistics, and the HTTP/NPZ endpoints:

```bash
python -m unittest discover -s tests -p test_data.py -v
```
