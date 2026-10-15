# Hardware, compute and training setup: π₀.₇ on SO-101, sim first

Checked on 2026-10-14 against vendor pages, Lambda's pricing page, the openpi and
LeRobot repositories and community SO-101 fine-tunes. Prices are on-demand list
prices before tax and will drift; re-check at checkout.

## 1. Decision

Train the repository's own **π₀.₇ reimplementation** (`pi_model/`, see
[PI07_IMPLEMENTATION.md](PI07_IMPLEMENTATION.md)) on two SO-101 followers with an
overhead camera and two wrist cameras. Data comes first from scripted MuJoCo
episodes, then from teleoperated demonstrations on the printed objects. The
paper-size model (5.2B parameters) is initialised from Gemma 3 4B and trained on
the two-H100 machine; a from-scratch `small` configuration (0.3B) runs the whole
pipeline without the gated weights.

## 2. Why this stack

| Choice | Reason |
| --- | --- |
| Own implementation of π₀.₇ | The paper's model is the project's point. No public π₀.₇ weights exist; the public π₀.₅ checkpoints have a different backbone (PaliGemma), no history encoder, no subgoal images and no metadata prompting. |
| Gemma 3 4B initialisation | The paper initialises the VLM and the vision encoder from Gemma 3 4B. `pi_model/weights.py` loads the HF safetensors; the license must be accepted on the Hub first. |
| Single arm first, bimanual sim kept | Public SO-101 data is single-arm; the sim records both arms (12-D state and action) so bimanual data needs no format change later. |
| MuJoCo sim in this repo | Menagerie SO-101, the Arducam camera profile, printed objects, 50 Hz control, scripted demonstrations with subtask segments and mistake retries, visual randomization. |

## 3. Constraints that shape the purchase

1. **Three cameras**: the paper's model takes a base view plus two wrist views (and an optional rear view). Here the base view is an overhead camera. The π₀.₇ history encoder needs frames every second, so every camera streams continuously at 30 fps.
2. **Wrist cameras must focus close.** A grasp happens 5–15 cm from the lens. Fixed-focus Raspberry-Pi-sensor USB modules focus from 1.5 m, so only the autofocus variant works on the wrist.
3. **50 Hz control** on the SO-101 bus, matching the paper's robots; a 50-step chunk is one second.
4. **Degrees and gripper 0–100** everywhere; the sim's commanded joint targets are the action labels.
5. **Six STS3215 servos, IDs 1–6, model number 777**, SO-101 geometry, so the simulator's Menagerie model matches the hardware.

## 4. Bill of materials

### Phase 1: one leader + one follower, overhead plus one wrist camera

| Item | Qty | Price (USD) | Source and notes |
| --- | --- | --- | --- |
| SO-101 Low-Cost AI Arm Kit Pro (servos, bus adapter boards, cables for leader + follower) | 1 | 249.90 | [Seeed Studio](https://www.seeedstudio.com/SO-101-Low-Cost-AI-Arm-Kit-Pro-p-6427.html). "Pro" is the 12 V follower variant; the leader is always 7.4 V. The page is vague: confirm it lists 12 servos and two adapter boards before paying. |
| SO-101 3D-printed structural parts | 1 | 29.90 | [Seeed Studio](https://www.seeedstudio.com/SO-101-3D-printed-Enclosure-p-6428.html), or print from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) in PLA+, 0.4 mm nozzle, 0.2 mm layers, 15% infill, supports everywhere except slopes over 45°. |
| Spare 12 V STS3215 (ST3215-C047) | 2 | 22.00 each | Seeed add-on. Follower servos are where failures concentrate. |
| Arducam B030401, 12 MP Sony IMX708 (Raspberry Pi Camera Module 3 sensor), USB 2.0 UVC, autofocus 10 cm–∞, 75° diagonal, 32×32 mm board | 4 | 49.98 (Arducam); about 60 at UK resellers | Profiled in `pi_sim/assets/so101_camera_rig.json`. Overhead plus left wrist in phase 1, right wrist in phase 3, one spare. The board drops into the official 32×32 UVC wrist mount. Do not buy the fixed-focus wide B0305 for wrists: it focuses from 1.5 m. |
| Wrist camera mount, 32×32 UVC hex-nut version | 1 printed | filament | [Official mount](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Optional/SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module); needs 2× M3×8 with nuts and 4× M2. |
| Overhead gantry | 1 | 15–40 | Rigid crossbar (aluminium extrusion or the official printed overhead mount) with the lens 0.70 m above the workspace center, looking straight down, matching the rig profile's `overhead` pose. |
| Powered USB 3.0 hub, 7 ports | 1 | 20–30 | Two serial adapters plus three cameras. Each camera is USB 2.0; keep them on separate controllers if 640×480 at 30 fps drops frames. |
| Table clamps | 2 | 5 | In the official BOM; the Seeed kit may include them. |
| LED panel light, neutral white | 1 | 25–40 | Fixed lighting across every recording session. |
| Matte filament, four colors | 1 kg | 20–30 | Objects from `printables/` plus goal disks. Matte PLA, 3 perimeters, 10–15% infill. |
| Neutral matte mat or tablecloth | 1 | 10–20 | One background for all sessions; a second one later for the background-shift test. |

Rough phase-1 total: **about 650 USD** with all four camera modules and spare servos, or about
480 USD with two modules and self-printed parts.

Alternatives if Seeed is out of stock: Hiwonder SO-ARM101 DIY kit at 269.99 or
assembled starter kit at 439.99 ([Hiwonder](https://www.hiwonder.com/products/lerobot-so-101)),
MakerMods leader + follower at 300 ([MakerMods](https://www.makermods.ai/so101)),
Robonine black kit with camera at 349 ([Robonine](https://robonine.com/shop/so-arm101-black-robotic-arm-kit/)).
WowRobo lists an assembled leader + follower + camera package at 199 but every
variant showed sold out on 2026-10-14 ([WowRobo](https://shop.wowrobo.com/products/so-arm101-diy-kit-assembled-version-1)).
Any of these works as long as it is the SO-101 geometry with STS3215 servos.

### Phase 3 add-on: second follower (and second leader for bimanual teleop)

A second Seeed Kit Pro plus printed parts (about 280 USD) gives a matched pair of
leaders and followers. If you only want a second follower for now, source six 12 V
STS3215, one bus adapter board (about 10.60) and one 12 V 5 A supply from the
official BOM, roughly 160–200 USD. The third Arducam module goes on its wrist.

### Robot host computer

| Phase | Machine | Notes |
| --- | --- | --- |
| Assembly, calibration, teleop, recording, sim | Your Mac | LeRobot's SO-101 tooling runs on macOS for calibration and teleoperation; `pi_observe` reads the arms and the three UVC cameras. |
| Policy inference | The two-H100 box as a policy server over an SSH tunnel; the robot client on the Mac | The paper runs the 5B model on one H100 at 38–127 ms per chunk; 50-step chunks with training-time RTC absorb the network round trip. |
| Later, optional | Linux box with an RTX 4090 (24 GB) | Local inference of the full model in bf16 (10 GB of weights) when the lab link is too slow. |

## 5. Compute

The project currently trains on a dedicated machine with **two NVIDIA H100 80 GB,
52 CPU cores, 442 GB RAM and 5.4 TB of disk** (`ssh ubuntu@192.222.52.191`).
Rendering there is CPU-side OSMesa, parallelised over 24 processes;
`scripts/box/` holds the launch scripts: `generate_data.sh`, `train_small.sh`, `train_full.sh`, `eval.sh`, and `eval_full_when_ready.sh`, which waits for each named full-run checkpoint and evaluates it on the CPU with an oracle subtask (the GPUs are full while the run trains; a GPU evaluation alongside it caused out-of-memory retries).

| Run | Config | Memory and speed |
| --- | --- | --- |
| `small` from scratch | 0.3B, 224 px, batch 32 per GPU, bf16 autocast | Fits comfortably; two GPUs with DDP via `torchrun`. |
| `full` from Gemma 3 4B | 5.2B, 448 px | Runs with FSDP2 on both GPUs (`--fsdp`, bf16 parameters, fp32 reductions) plus `--gradient-checkpointing`, batch 4 per GPU with 4 accumulation steps, 77–80 GB per GPU and about 10 s per step (3 samples/s), so 10k steps take 28–30 h. Gated weights: accept the Gemma license, then `hf download google/gemma-3-4b-pt --local-dir weights/gemma-3-4b-pt`. |

Lambda on-demand prices for reference (2026-10-14, per GPU-hour, plus tax): 1× H100 PCIe 3.29, 1× H100 SXM 4.29, 8× H100 SXM 3.99, 8× A100 80 GB 2.79, 1× A100 40 GB 1.99, 1× A10 24 GB 1.29.

## 6. Plan and exit criteria

**Phase 0, now, no hardware.** Record scripted pick/place episodes with the three-camera rig at 50 Hz (`python -m pi_sim.record`), fit the FAST tokenizer and statistics (`python -m pi_model.data prepare`), train, and evaluate closed loop in MuJoCo with the paper's runtime prompting (`python -m pi_model.eval_sim`). Exit: over 80% success on held-out seeds and one held-out color, with a simulated inference delay.

**Phase 1, hardware arrives.** Assemble, set motor IDs and calibrate with LeRobot's tooling. Mount the overhead and wrist cameras at the rig profile's poses, checkerboard-calibrate each module at its working focus, write the intrinsics into `configs/so101_camera_rig.local.json`, and re-render sim data with the measured optics.

**Phase 2, sim to real.** Run the sim checkpoint zero-shot for 20 rollouts and record failure types. Teleoperate 50–100 demonstrations per task on the printed objects, label segments, and co-train sim plus real. Evaluate 20 rollouts per condition with the fixed success definition from `docs/SO101_MANIPULATION.md`.

**Phase 3, bimanual.** Add the second follower and the right wrist camera; action dimensions 7–12 and the third image slot become live. Collect genuinely bimanual demonstrations; never pad the right arm with zeros and call it supervision.

## 7. Sources

- π₀.₇ paper (arXiv 2604.15483v2): architecture, prompt, dropout, runtime prompting, inference timing.
- MEM (arXiv 2603.03596): space-time separable attention every 4th ViT layer, state projection, history stride.
- Training-time RTC (arXiv 2512.05964): clean prefix with timestep 1, loss on the postfix only.
- Gemma 3 4B config (`google/gemma-3-4b`): text and vision dimensions.
- Arducam B030401 listing (The Pi Hut): autofocus 10 cm–∞, 75°×61°×50°, 32×32 mm, USB 2.0, 640×480/1280×720/1920×1080 at 30 fps; Arducam B0305 datasheet (DigiKey): fixed focus 1.5 m–∞.
- TheRobotStudio SO-ARM100 BOM; Seeed Studio kit and printed-parts pages; Lambda pricing page.
