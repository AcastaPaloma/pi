# Research log: getting a closed-loop SO-101 policy working in simulation

Protocol (from the user, 2026-10-15): at every wake-up, record the new results; if they are bad,
state hypotheses for why and the changes each implies; watch a hypothesis for two or three
wake-ups before deciding; then apply the change or drop it. Extra compute can be rented if needed.

Status key: **watching** (gathering evidence), **acting** (change being applied), **closed**.

## Results so far (closed loop, scripted-subtask prompts)

| Model | Task | Step | Success | Lift |
| --- | --- | --- | --- | --- |
| small 0.3B, Gemma-style, KI on | single arm | 5000 | 1/20 (a push) | 3/20 |
| full 5.2B from Gemma 3 4B, KI on | single arm | 1500–5000 | 0–2/20 | 0–6/20 |
| full_v2 warm start, two-arm data | ladder rungs | 500–2500 | 0/8 each | ≤1/8 |
| exp C, small, KI off | single arm | 3000 | 2/20 (real grasps) | 3/20 |
| **pi05_single** (π0.5 backbone) | single arm | 1000 | **3/10** | 3/10 |
| pi05_v1 (π0.5 backbone) | ladder rungs | 1000–2000 | 0/8 each | 0 |

## Closed findings

- **Data labels are correct**: recorded actions replay to success in 17/18 episodes.
- **Gemma 3 / from-scratch backbones were blind**: losses identical with shuffled images; fixed by
  the π0.5 backbone (actions shift 0.7–2.2° with another episode's images at step 1000) and partly by
  turning knowledge insulation off (exp C at step 3000).

## Open hypotheses

### H1: the two-arm run's mixture is too hard to learn quickly (watching, since 2026-10-15 22:00)
Four tasks, both arms, arm chosen by the object's side, 30 s episodes and wider placements; 7× the
samples of the single-arm data, so at step 2000 it has seen ~13% of one pass.
Evidence for: pi05_single gets 3/10 at step 1000 while pi05_v1 gets 0 on every rung at 1000–2000.
Possible changes: (a) warm-start the two-arm run from the best pi05_single checkpoint; (b) curriculum:
train pick_place_any and bin first, add handover and two_bins later; (c) more compute (8×H100) so a
full pass fits in hours. Decide after pi05_v1 steps 3000–4000 are scored.

### H2: demonstrations never show recovery (watching, since 2026-10-15 22:00)
Every recorded trajectory is clean apart from retried grasps; a policy that drifts 1–2 cm sees
states the data never covered and the error compounds (consistent with 5–9 cm misses).
Possible changes: (a) DART-style noise injection: perturb the teacher's executed actions during
recording while labelling the clean corrective action, so the data covers off-path states; (b) bump
events: nudge the object mid-approach and let the teacher re-plan; (c) DAgger: roll out the policy,
label each visited state with the teacher (privileged state makes this cheap). Decide after seeing
whether pi05_single plateaus below ~60% and whether its failures are near-misses.

### H3: the ladder evaluation is stricter than the single-arm one (watching)
Rungs use 8 episodes, 30 s, both sides of the table and container walls; the single-arm test uses
the original placement. Not a model problem, but it changes how to read 0/8. Check by scoring
pi05_single and pi05_v1 on the same task (pick_place_any, left side only) once both have ~3000 steps.

### H4: single-arm failures now happen after the grasp (weakened 2026-10-16 00:30; see entry)
pi05_single step 3000 lifts the cube in 8/10 episodes but succeeds in 5/10: the remaining failures
are in carrying, placing or releasing, not in reaching. Possible changes: (a) longer training (the
place phase is the last to be learned); (b) more place/release supervision via recovery episodes
(knocks during transfer and lower-to-goal); (c) check the success rule (released, settled, on disk)
against videos of the failures. Decide after steps 4000–5000 and a video review.

## Experiments and compute (from 2026-10-15 ~22:40)

| Run | Machine | Change vs pi05_v1 | Tests | Evaluated by |
| --- | --- | --- | --- | --- |
| pi05_v1 | main box GPU0 | — (π0.5 base, two-arm mix) | baseline | first A10, 8/rung |
| pi05_single | main box GPU1 | single-arm data only | backbone works at all | GPU1 (10) + second A10 (30) |
| pi05_warm2arm | 209.20.159.198 H100 | warm start from pi05_single step 2000 | H1a | own GPU, 16/rung |
| pi05_recover | 209.20.157.232 H100 | + 2,000 recovery episodes | H2 | own GPU, 16/rung |
| pi05_easy | 209.20.158.194 H100 | pick_place_any + bin only | H1b (curriculum) | own GPU, 16/rung |

Monitoring: `python3 scripts/leaderboard.py` collects every result into docs/RESULTS.md with Wilson
95% intervals. **Stop rule** (user: do not kill tries too early): stop a run only after at least four
evaluated checkpoints, when the upper bound of its best checkpoint is below the lower bound of the
leading run on the same task, or when its loss diverges. Otherwise it keeps training.

## Wake-up entries

- **2026-10-15 ~22:00** pi05_single step 1000: 3/10 (genuine grasps, one in 194 steps). pi05_v1
  step 2000: 0/8 on pick_place_any (61 mm) and bin (76 mm). Opened H1–H3. Started a per-checkpoint
  evaluator for pi05_single on GPU1.
- **2026-10-15 ~22:40** pi05_single step 2000: **5/10 success, 6/10 lift** (up from 3/10). pi05_v1
  step 2000: 0/8 on all four rungs (handover 1/8 lift). H1 gains evidence (single-arm improves fast,
  two-arm flat); per protocol, not acting on pi05_v1 itself yet. User added three 1×H100 PCIe nodes
  and an A10: launching pi05_warm2arm (H1a), pi05_recover (H2), pi05_easy (H1b) so each hypothesis is
  tested by its own run instead of sequentially. Recovery data generated: 2,000 episodes, 1,481 successful.

- **2026-10-15 ~23:30** Leaderboard fixed (one malformed old result file had hidden later ones).
  pi05_single: 3/10 → 5/10 → 5/10 at steps 1000–3000, lift 8/10 and 10 mm median closest at 3000;
  30-episode check at step 2000: 10/30 (0.19–0.51). pi05_v1 step 3000: still 0 successes but clearly
  learning (closest 61→20 mm on pick_place_any, handover lift 3/8, two_bins lift 2/8). H1 evidence:
  two-arm is slower, not stuck — keep running, no stop. Opened H4 (post-grasp failures). Launched
  pi05_warm2arm, pi05_recover and pi05_easy on the three new H100 PCIe nodes (≈10 samples/s each;
  the warm start's checkpoint copy failed once on a missing folder and was redone).
- **2026-10-15 ~23:55** pi05_single step 3000 on 30 episodes: **19/30 success (0.46–0.78), lift 24/30**,
  up from 10/30 at step 2000 — the leading model. pi05_v1 step 4000 pick_place_any: 0/8, lift 2/8,
  closest 21 mm (holding the step-3000 gains). H4 (post-grasp failures) still open: lift 80% vs
  success 63%.
- **2026-10-16 ~00:30** pi05_v1 step 4000 fully scored: first two-arm success (bin 1/8); every rung now
  13–24 mm from the object (was 46–76 mm at step 2000) — converging, slower than single-arm (H1, keep).
  Video review of pi05_single step 3000 (10 episodes, 6/10 success): of 4 failures, 3 never lifted
  (gripper within 1–2 cm) and 1 lifted then lost the cube; in both reviewed failures the arm makes
  repeated small re-grasps that nudge the cube, ending in a retry loop near the arm's base. **H4
  weakened**: the dominant failure is grasp precision plus poor recovery once the cube moves, which is
  **H2**'s prediction; pi05_recover tests it directly. Decision: no change yet; watch pi05_single at
  steps 4000–5000 (still rising: 10/30 → 19/30) and pi05_recover's first scores.
- **2026-10-16 ~00:55** pi05_single step 4000 on 30 episodes: **21/30 (0.52–0.83), lift 26/30** —
  33% → 63% → 70% over steps 2000–4000, still rising; it is the working sim policy for the original
  task. No changes. pi05_v1 step 5000 being scored; the three experiments reach step 1000 shortly.
- **2026-10-16 ~00:40** (user asked to verify two-arm successes by video, and for much more recovery
  data). Full three-camera captioned videos (`eval_sim --video --full-video 2`): the two-arm success
  (pi05_v1 step 5000, bin) is genuine — right arm grasps (gripper 0.16, contact), lifts to 7.2 cm,
  carries over the container (wrist camera looks into it), releases, cube lands 2.2 cm from centre.
  A single-arm success (pi05_single step 4000) is a clean 4 s pick-and-place. Replays: two-arm bin on
  new seeds 0/12 (lift 4/12), on the original seeds 1/8; single-arm 4/6 on camera.
  **Bug found (H3 confirmed in a new form):** the ladder's subtask oracle only advanced when the arm hit
  the teacher's own waypoints, so the policy was prompted "Lift the cube" for ~10 s while hovering over
  the bin. Replaced with a state-based oracle (object held/at destination, hand position, gripper) with
  hysteresis; it reproduces the teacher's phase sequence on every rung (45–53% exact agreement on the
  single-object rungs, the rest adjacent labels a dwell apart). **Two-arm scores from step 6000 on are
  not comparable with earlier ones.**
  **Recovery data v2 (user request, also H2):** object bumps during approach/grasp (teacher re-reads the
  pose and corrects, re-approaching from above if it moved >1.5 cm while low) and deliberate off-target
  grasps (30% of attempts, mistake-labelled, then clean retries), on top of target knocks. Teacher
  success 70–85% with 4–8 bumps and 1–2.5 misses per episode. Generating 6,000 episodes on the second A10.
  Infra: the setup script had failed to install NVIDIA EGL on the four new machines (software rendering,
  0.9 s/frame); fixed (3 ms/frame) and the script now verifies the render path.
