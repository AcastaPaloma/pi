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

### H7: two_bins fails after the grasp (closed 2026-10-16 ~07:40 — oracle deadlock, fixed)
Video of pi05_v1 step 10000 (6 two_bins episodes, 0 successes, lift 6/6): the left arm drops its cube in
the bin by ~3 s (cube 1.4–2.6 cm from centre in 5/6) but then hovers over the bin; the right arm holds its
cube and never carries it. In the teacher's demos the right arm carries only after the left arm is back
at rest, labelled "Move the left arm back to its rest position". The state oracle declared the left arm
done 6 cm from its cube and prompted "Carry the blue cube" while the left arm still hovered — a
combination never seen in training. Fix: after placing, the oracle keeps the arm in raise-away until it
is back at carrying height and in return-home until its hand is within 3 cm of its rest pose. Same
checkpoint and seeds: first two_bins success, video-verified (both grasp, left places and goes home,
right carries and places beside it, 6.4 s). Remaining failure seen: the right arm drops its cube outside
the rim and then hovers over the bin instead of re-grasping (recovery, H2). Two_bins scores before this
fix understate the policy.

### H8: pi05_2arm_v2 dips after its strong start (watching, since 2026-10-16 ~09:20)
Step 2000: pick_place_any 4/12, bin 6/12; step 3000: 0/12 on every rung (lift ≤ 2/12). Re-scored on the
same 6 seeds: step 2000 1/6, step 3000 0/6 — reproducible, not an evaluation fault; checkpoint files
intact; training flow loss steady at ~0.03. Hypothesis: the warm start first carries over single-arm skill,
then temporarily loses it while adapting to the two-arm action statistics and the recover2 mix (the
opposite curve of pi05_warm2arm, weak at 1000–2000 and strong at 4000). Possible changes if it does not
recover: lower LR or a longer warmup for warm starts, or mix in single-arm data to anchor the skill.
Decide after steps 4000–5000.

### H9: raw-weight checkpoints oscillate; an EMA of the weights stabilises them (watching, since ~09:55)
pi05_warm2arm: 11/16, 9/16 at step 4000 then 5/16, 5/16 at step 5000; pi05_2arm_v2: 4/12, 6/12 at 2000 then
0/12 at 3000 (reproduced). Training losses are smooth, so the swings come from where the raw weights sit
between saves. π0-family and diffusion policies are usually evaluated with averaged weights. Added
`--ema-decay` (train.py; model.pt = averaged weights, model_raw.pt = raw) and queued **pi05_2arm_ema** (the
2arm_v2 recipe with EMA 0.999) second in the H100 queue. If its curve is smoother and higher, make EMA the
default for every run.

### H10: two_bins is the weakest two-arm rung (watching, since 2026-10-16 ~13:00)
30-episode confirmation of pi05_v1 step 13000: two_bins 9/30 (30%) vs 57–90% on the other rungs; lifts are
high (8-episode runs show 6–8/8), so the failures are after the grasp. Seen on video earlier: the second
arm drops its cube outside the rim and then hovers over the bin instead of re-grasping (recovery), and
the oracle deadlock (fixed). Possible changes: DAgger data on two_bins specifically (aggregated in
pi05_dagger1), more two_bins teacher episodes with the right arm placing first as well (the teacher always
places left first), a wider container. Decide after pi05_dagger1's first scores and a two_bins video review
of step 17000.

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
- **2026-10-16 ~01:30** pi05_single step 5000: **23/30 (0.59–0.88)** on 30 episodes, 7/10 on 10
  (33% → 63% → 70% → 77% over steps 2000–5000). pi05_v1 step 6000 (first scores with the state-based
  oracle): handover lift 5/8 at 11 mm, two_bins lift 2/8. New runs at step 1000 (16/rung):
  pi05_recover pick_place_any 1/16, pi05_warm2arm 0/16 (lift 2/16), pi05_easy 0/16 — too early to read.
  recover2 generation was crawling because 11 orphaned workers from the software-rendering attempt kept
  running (load 66 on 30 cores); killed. No decisions: every run keeps training.
- **2026-10-16 ~02:00** **pi05_single step 6000: 28/30 success (≈0.79–0.98), lift 27/30** on 30
  held-out episodes — the single-arm sim policy works (33% → 63% → 70% → 77% → 93% over steps 2000–6000).
  pi05_v1 step 6000 fully scored: bin 1/8 (lift 5/8), handover lift 5/8, pick_place_any and two_bins
  lift 2/8; 10–19 mm everywhere — grasps and lifts are learned, placing is not yet. Next for two-arm:
  a warm start from pi05_single step 6000 plus recover2 data is the strongest candidate (H1a + H2);
  it needs a GPU, so either replace a run that loses on the leaderboard or rent one more H100.
- **2026-10-16 ~02:30** Steps: pi05_v1 7560, pi05_single 6900, warm2arm 2240, recover 2060, easy 2080; no
  errors. Step-1000 scores of the three experiments are all 0–1/16 per rung (lifts 0–19%), the same as
  pi05_v1 at step 1000 — no early separation; H1a/H1b/H2 still open, decided at steps 3000–4000.
  **Decision (compute):** pi05_single reached its goal (93% at step 6000). If its 30-episode score at
  step 7000 confirms ≥ 85%, its GPU goes to **pi05_2arm_v2**: warm start from the best pi05_single
  checkpoint on two-arm data + 6,000 recover2 episodes (data/sim_mixrec2_v1). This reassigns a finished
  run rather than stopping an experiment early. recover2 generation at 5,494/6,000.
- **2026-10-16 ~03:15** pi05_v1 step 7000 (new oracle): **pick_place_any 2/8, bin 4/8** (lift 5/8),
  handover 0/8 with both arms touching in 1/8, two_bins 0/8 (both 1/8) — six successes vs one at 6000.
  **pi05_easy step 2000: pick_place_any 3/16, bin 5/16** — first strong evidence for H1b (easy rungs
  learn fast; pi05_v1 had 0 at step 2000, though its early checkpoints used the old stalling oracle).
  pi05_single step 7000: 9/10 and **24/30 (0.63–0.90)**; steps 6000/7000 pooled 35/40 vs 33/40 —
  plateau in the 80s. The ≥85% reassignment bar was not met at 7000, so it keeps training; decide at
  8000. recover2 finished: 5,965 episodes (4,243 successful, 11 crashed and skipped); copying to the
  main box as data/sim_mixrec2_v1.
- **2026-10-16 ~03:40** Step-2000 scores (16/rung): pi05_easy pick_place_any 3/16, bin 5/16;
  pi05_recover pick_place_any 0/16, bin 1/16; pi05_warm2arm bin 0/16, handover 0/16. H1b (curriculum)
  leads; H1a (warm start from the 50% single-arm checkpoint) and H2 v1 (target-knock recovery only)
  show no gain yet — keep watching to step 4000 per the stop rule. recover2 data (bumps + missed
  grasps) is on the main box as data/sim_mixrec2_v1 (10,794 episodes), ready for pi05_2arm_v2.
- **2026-10-16 ~04:00** **pi05_v1 step 8000: first two-arm handover successes** — handover 3/8 (lift 8/8,
  both arms touched 4/8), pick_place_any 4/8 (lift 7/8), two_bins 0/8 (lift 5/8, both arms 3/8), bin
  0/8 (lift 0/8; was 4/8 at step 7000 — watch for a regression vs. 8-episode noise). Video check on the
  original seeds: 1/8 handover succeeded on replay (GPU nondeterminism); frame by frame it is a genuine
  handover — right arm grasps (0.18), lifts to 6.3 cm, sets the cube on the relay pad, retreats; left
  arm grasps from the pad (0.17), lifts to 6.2 cm, releases inside the container; both grippers
  touched; success at 11.3 s. Copied to ~/Downloads/pi_sim_videos for the user. Cosmetic oracle quirk:
  after the final release the caption reverts to "Move the left gripper above the cube".
- **2026-10-16 ~04:30** **Decision:** pi05_single plateaued (pooled 35/40 at step 6000, 33/40 at 7000,
  7/10 at 8000) and was stopped at step ~8200; its best checkpoint is **step 6000 (28/30)**, the
  single-arm demo model. Its GPU (main box GPU1) now runs **pi05_2arm_v2**: warm start from
  pi05_single step 6000 on data/sim_mixrec2_v1 (two-arm mix + 6,000 recover2 episodes), 12,000 steps,
  scored on the second A10 at 12 episodes per rung. First launch silently ignored RESUME/META because
  the main box had a stale launcher; synced and relaunched.
- **2026-10-16 01:56** [scheduler] 132.145.199.243 was idle; launched **single6000_execute_sweep** (H6: re-planning more often (execute 10 or 15 actions per chunk) raises closed-loop success of the solved single-arm model)
- **2026-10-16 ~05:15** **pi05_v1 step 9000: pick_place_any 6/8, bin 5/8**, handover 1/8 (both arms 1/8),
  two_bins 0/8 (lift 5/8) — single-object two-arm rungs now work most of the time; coordination tasks
  are the frontier. pi05_easy step 3000: 2/16 and 3/16 (down from 3/16, 5/16; 16-episode noise).
  Scheduler bug: its first A10 launch (execute sweep) failed because double-quoting let the login
  shell expand `$E`; switched to shlex quoting, tagged jobs (SCHEDJOB_<name>) and let an A10 job run
  beside checkpoint evaluations when memory allows. Re-queued the sweep; dagger1_generate launches next.
- **2026-10-16 ~05:35** **Shortcut check (user's concern):** pick_place success only requires the cube
  to end on the disk, so a push counts. The leaderboard now shows **strict success = success with the
  cube lifted**. Correction: pi05_single step 6000 is **26/30 strict (≈87%)**, not 28/30 — two of its
  successes were pushes; step 5000 22/30, step 7000 21/30. pi05_warm2arm step 3000 pick_place_any 3/16
  (2/16 strict). Container, handover and two-object rungs need a lift to succeed (cube inside walls),
  so they are unaffected. Report strict numbers from now on. pi05_warm2arm step 3000 otherwise improved
  (bin 5/16, handover lift 7/16) vs pi05_v1 at step 3000 (0/8): H1a now has some support.
- **2026-10-16 02:28** [scheduler] 132.145.199.243 was idle; launched **single6000_execute_sweep** (H6: re-planning more often (execute 10 or 15 actions per chunk) raises closed-loop success of the solved single-arm model)
- **2026-10-16 ~06:40** Step 3000 of the three experiments (16/rung): pi05_recover bin 5/16, two_bins lift
  6/16; pi05_warm2arm bin 5/16, handover lift 7/16; pi05_easy bin 3/16. All three beat pi05_v1 at the
  same step (0/8; caveat: old oracle), so each change speeds early learning; none separates from the
  others yet — step 4000 is their fourth checkpoint. pi05_v1 at step 10320, pi05_2arm_v2 at 1280 (first
  score pending). Scheduler: a hung ssh launch had blocked it; launches are now detached and confirmed
  by job tag; the A10 execute-horizon sweep is running and DAgger generation is at 88 episodes. Queue:
  added **pi05_v1_rec2** (continue the best two-arm model on recover2 data) at the front.
- **2026-10-16 ~07:00** pi05_v1 step 10000: bin **6/8**, pick_place_any 4/8, handover 1/8 (lift 7/8, both
  arms 4/8), two_bins **0/8 with lift 7/8**. Opened **H7: two_bins fails after the grasp** — candidate
  causes: the second arm stalls on the "hold and wait" prompt, arms collide over the shared container, or
  a cube lands on the rim/other cube. Recording 6 two_bins + 6 handover full videos of step 10000 on the
  easy-rungs H100 to decide. pi05_2arm_v2 (warm start from single-arm 6000 + recover2) step 1000:
  pick_place_any 2/12, bin 2/12 — the best step-1000 two-arm start so far.
- **2026-10-16 ~07:40** H7 closed (see above): oracle deadlock in two_bins fixed and deployed to every
  evaluator; first video-verified two_bins success. pi05_easy step 4000: **pick_place_any 6/16, bin 7/16**
  (pi05_v1 had 0/8 and 1/8 at step 4000) — strong support for H1b (curriculum); its stage 2 is queued.
- **2026-10-16 ~07:55** **H6 closed (no):** pi05_single step 6000 with execute 10/15/25/40 actions per chunk:
  strict 21/26/26/27 of 30 — re-planning more often does not help; keep execute 25. DAgger data from
  pi05_v1 at 351/600. Queued **dagger2_generate** (rollouts of pi05_2arm_v2) for the idle second A10.
  Steps: pi05_v1 11580, 2arm_v2 2560, warm2arm 4420, recover 4240, easy 4380; no errors.
- **2026-10-16 03:20** [scheduler] 132.145.199.243 was idle; launched **dagger2_generate** (H2b data, second policy: rollouts of pi05_2arm_v2 (warm start + recover2) finished by the teacher; aggregates mistakes from a different policy)
- **2026-10-16 ~08:30** **pi05_v1 step 11000 (first ladder scores with the H7 oracle fix): two_bins 3/8**
  (0/8 at every earlier step; lift 8/8), pick_place_any 5/8, bin 5/8, handover 1/8 (lift 7/8) — every
  rung now has successes, all with a lift. **pi05_2arm_v2 step 2000: pick_place_any 4/12, bin 6/12** —
  the best step-2000 result of any run (warm start from single-arm + recover2). pi05_recover step 4000
  (4th checkpoint): 1/16, 4/16, 0, 0 — behind pi05_easy at the same step but its intervals still
  overlap at equal steps and on bin vs the leader, so not clearly beaten; keep, re-check at 5000–6000.
  dagger2_generate launched on the second A10 (03:19).
- **2026-10-16 ~08:50** **pi05_warm2arm step 4000: pick_place_any 11/16 (strict), bin 9/16**, handover 1/16
  (lift 7/16), two_bins 0/16 (lift 5/16) — best step-4000 result of any run (was 3/16, 5/16 at 3000).
  **H1a now strongly supported** (warm start from the single-arm model), alongside H1b (pi05_easy 6/16, 7/16)
  and the even faster pi05_2arm_v2 (warm start + recover2: 4/12, 6/12 at step 2000). DAgger data: v1 429/600,
  v2 29/600. No errors; pi05_v1 at step 12000.
- **2026-10-16 ~09:20** pi05_v1 step 12000: pick 5/8, bin 3/8, handover 1/8, two_bins 1/8 — flat within
  noise since step 9000 (plateau). pi05_2arm_v2 step 3000: 0/12 on every rung after 4/12, 6/12 at step
  2000; reproduced on fixed seeds; opened H8.
- **2026-10-16 ~09:40** DAgger data v1 done: 532 episodes (430 teacher-successful; 68 skipped because the
  policy finished or failed before the takeover). **pi05_dagger1 unblocked** and moved to the front of the
  H100 queue (fine-tune pi05_v1 on mix + recover2 + DAgger). Queued **dagger3_generate** for the idle first
  A10 from pi05_warm2arm step 4000 (the strongest two-arm policy). dagger2 at 194/600. No errors.
- **2026-10-16 ~09:55** pi05_warm2arm step 5000: pick 5/16, bin 5/16 (from 11/16, 9/16), handover 2/16.
  Opened H9 (EMA); implemented and smoke-tested `--ema-decay`; queued pi05_2arm_ema.
- **2026-10-16 ~10:25** **pi05_v1 step 13000: pick_place_any 6/8, bin 5/8, handover 6/8 (both arms touched
  7/8), two_bins 5/8** — the two-arm tasks now succeed most of the time (handover and two_bins were 1/8 at
  step 12000); every success includes a lift and the handover successes need both grippers on the cube.
  The plateau at 9000–12000 was not the ceiling. 8-episode samples are noisy; confirm at 14000.
- **2026-10-16 04:27** [scheduler] 129.213.124.160 was idle; launched **dagger3_generate** (H2b data from the strongest two-arm policy (pi05_warm2arm step 4000: 11/16, 9/16))
- **2026-10-16 ~10:45** **pi05_recover step 5000: pick 5/16, bin 8/16, handover 4/16, two_bins 4/16** — the
  first run with successes on all four rungs by step 5000 (pi05_v1's step-5000 two-arm scores predate the
  oracle fix). It looked beaten at step 4000; not stopping it was right. Scheduler: busy check fixed;
  dagger3_generate (from pi05_warm2arm step 4000) running on the first A10; dagger2 at 269/600.
- **2026-10-16 ~11:05** pi05_2arm_v2 step 4000: pick 2/12, bin 1/12, handover 2/12 (both arms 2/12) —
  partial recovery from 0/12 at 3000 (H8: dip then recovery). Queued **v1_13000_confirm30** (30 episodes per
  rung on pi05_v1 step 13000, the two-arm demo candidate) for the first A10 after dagger3. Steps: pi05_v1
  14260, 2arm_v2 5260, warm2arm 6020, recover 5760, easy 6020; no errors. dagger2 335/600, dagger3 159/600.
- **2026-10-16 ~11:30** **pi05_v1 step 14000 confirms the breakthrough:** pick_place_any 7/8, bin 4/8,
  handover 4/8 (both arms 6/8), two_bins 4/8 (steps 13000+14000 pooled: 13/16, 9/16, 10/16, 9/16, all with
  lifts). The two-arm sim policy works on every rung roughly half to most of the time. pi05_warm2arm step
  6000: pick 10/16, bin 6/16, handover 4/16 (recovered from the step-5000 dip). pi05_easy step 6000: pick
  10/16, bin 8/16. The 30-episode confirmation of step 13000 is queued for the first A10.
- **2026-10-16 06:04** [scheduler] 129.213.124.160 was idle; launched **v1_13000_confirm30** (Confirm the two-arm breakthrough of pi05_v1 step 13000 (6/8, 5/8, 6/8, 5/8 on 8 episodes) with 30 episodes per rung — the two-arm demo candidate)
- **2026-10-16 ~12:10** pi05_v1 step 15000: pick 7/8, bin 4/8, handover 2/8 (both arms 7/8), two_bins 3/8
  (steps 13000–15000 pooled ≈ 83/54/50/50% over 24 episodes per rung — stable). pi05_2arm_v2 step 6000: 4/12,
  5/12, 2/12, 1/12 — recovered past its step-2000 peak (**H8 confirmed: dip then recovery**). pi05_easy step
  7000: pick 10/16, bin 11/16. pi05_recover step 6000: 9/16, 9/16, 4/16, 0/16. DAgger v3 done (546 episodes,
  470 successful); pi05_dagger1 now aggregates DAgger v1+v2+v3 (~1,600 episodes). The 30-episode confirmation
  of pi05_v1 step 13000 is running on the first A10.
- **2026-10-16 06:27** [scheduler] 132.145.199.243 was idle; launched **dagger4_generate** (H2b data from the best two-arm policy (pi05_v1 step 15000: 7/8, 4/8, 2/8, 3/8))
- **2026-10-16 ~12:35** **30-episode confirmation of pi05_v1 step 13000 (strict = with lift): pick_place_any
  27/30 (90%), bin 22/30 (73%), handover 17/30 (57%)**; two_bins still running. pi05_v1 step 16000: 7/8, 5/8,
  4/8, 6/8 (69% overall, the best checkpoint). **pi05_2arm_v2 step 7000: 8/12, 9/12, 3/12, 5/12 (52%)** vs
  pi05_v1's 19% at step 7000 — warm start + recover2 reaches a working two-arm policy in about half the steps.
  DAgger v2 done (570 kept, only 171 teacher-successful: generated during 2arm_v2's dip); dagger4 (from
  pi05_v1 step 15000) launched on the second A10.
- **2026-10-16 06:48** [scheduler] 129.213.124.160 was idle; launched **v1_17000_confirm30** (Confirm pi05_v1 step 17000 (8/8, 6/8, 7/8, 3/8 on 8 episodes; best checkpoint) with 30 episodes per rung)
- **2026-10-16 ~13:00** **30-episode confirmation of pi05_v1 step 13000 (strict): pick_place_any 27/30 (90%),
  bin 22/30 (73%), handover 17/30 (57%), two_bins 9/30 (30%) — 75/120 overall.** Step 17000 (8/8, 6/8, 7/8,
  3/8 on 8 episodes) is now being confirmed on 30. Opened H10 (two_bins weakest).
- **2026-10-16 07:24** [scheduler] 129.213.124.160 was idle; launched **v1_17000_twobins_video** (H10 evidence: watch how step-17000 two_bins episodes fail)
- **2026-10-16 ~11:30 UTC (07:30 EDT)** **30-episode confirmation of pi05_v1 step 17000 (strict): pick_place_any
  23/30, bin 22/30, handover 16/30, two_bins 15/30 — 76/120**, a tie with step 13000's 75/120. The two
  checkpoints trade skills on identical seeds: pick 27 → 23, two_bins 9 → 15. That is the H9 pattern
  (neighbouring checkpoints oscillate) on 30-episode evidence, not 8-episode noise. Step 17000 is the
  two-object demo model; step 13000 stays the single-object one.
  **New H9b test, no training needed:** a uniform average of pi05_v1 steps 13000–18000 (six checkpoints,
  `scripts/box/average_checkpoints.py`, built on the main box) queued as **v1_soup_confirm30** on the same
  seeds. Prediction: it matches the better checkpoint on every rung (≥ 27, 22, 17, 15). If it does, every run
  should be evaluated as a soup and EMA (pi05_2arm_ema) becomes the default. The DAgger fine-tune now also takes
  DAgger v4 (from step 15000, 340/600 done). Other runs at their latest checkpoint: warm2arm step 8000 8/16,
  5/16, 4/16, 2/16; 2arm_v2 step 8000 5/12, 5/12, 3/12, 1/12; recover step 7000 4/16, 5/16, 1/16, 1/16.
  None meets the stop rule yet (each still overlaps the leader on some rung). Steps: pi05_v1 18560, 2arm_v2
  9580, warm2arm 8420, recover 8200, easy 8640; no errors.
- **2026-10-16 ~11:30 UTC** **H10 video review (pi05_v1 step 17000, two_bins, 6 episodes): 4/6 succeed.**
  Successes are genuine: the check requires both cubes inside the bin with both grippers open; frames show the
  left arm placing its cube and the right arm placing the blue one. **Both failures share one mode:** the left
  arm places its cube, the right arm's first grasp on the blue cube misses (gripper stays open at the cube,
  t≈3 s) and the policy then hovers above the cube for the remaining ~25 s, never re-opening/re-grasping. This is
  a missed-grasp-recovery gap, not an ordering problem: the policy runs both arms in parallel. pi05_v1 never saw
  recover2's deliberate missed grasps, so pi05_dagger1 (mix + recover2 + DAgger v1–v4, resumes pi05_v1) and
  pi05_v1_rec2 directly target it. Instrumentation gap fixed: two_bins evaluations now also record
  `cube2_in_bin`, `cube2_lifted` and the first cube's `cube_in_bin`, so per-arm failure rates are measurable on
  30 episodes. pi05_2arm_v2 step 9000: 8/12, 8/12, 2/12, 1/12.
- **2026-10-16 11:35 UTC** pi05_v1 step 18000: 6/8, 7/8, 7/8, 2/8 (two_bins keeps swinging 6 → 3 → 2 over
  steps 16000–18000, the H9 pattern again). pi05_2arm_v2 step 9000: 8/12, 8/12, 2/12, 1/12 (handover and
  two_bins stuck low; the run ends at 12000). **Queue reprioritised on the H10 video evidence:** A10
  v1_soup_confirm30 now runs before twobins_mixed_generate (failures are missed-grasp recovery, not arm
  order); H100 pi05_v1_rec2 moved to second, behind pi05_dagger1 — both teach recovery from a missed grasp to
  the best model. Expected launches: pi05_dagger1 on main GPU0 when pi05_v1 hits 20000 (~45 min; DAgger v4
  will be partial, ~400/600), then pi05_v1_rec2 and pi05_2arm_ema as 2arm_v2 / warm2arm / easy end at 12000
  (~1.5–2 h). pi05_recover runs to 20000 (~8 h): 8 checkpoints, weak (best 12/16 bin, 3/16 handover) but
  not clearly beaten, so it continues under the stop rule; if its next four checkpoints stay below pi05_v1's
  lower bounds it gets stopped for a queued entry. No errors on any host; all five H100s training, A10#2
  generating DAgger v4 (handover 80/150), A10#1 about to take the soup confirmation. No extra compute needed.
- **2026-10-16 07:37** [scheduler] 129.213.124.160 was idle; launched **v1_soup_confirm30** (H9b: checkpoints trade skills (13000 vs 17000: pick 27 vs 23, two_bins 9 vs 15 of 30). A uniform average of pi05_v1 steps 13000-18000 should keep both, beating each single checkpoint on the same seeds)
- **2026-10-16 11:50 UTC** v1_soup_confirm30 (average of pi05_v1 steps 13000–18000, same seeds as the two
  single-checkpoint confirmations) half done: pick_place_any 24/30 (13000: 27, 17000: 23), bin 21/30 (22, 22) —
  within noise of both, no gain yet; handover and two_bins (the rung where 13000 and 17000 disagreed most, 9 vs
  15) decide H9b. Steps: pi05_v1 19320, 2arm_v2 10360, warm2arm 8920, recover 8640, easy 9100; no errors.
  DAgger v4 at handover 140/150 (two_bins next). Queue unchanged: 5 runnable H100 entries, 1 A10.
- **2026-10-16 12:15 UTC** **H9b refuted: the pi05_v1 checkpoint average (steps 13000–18000) scores 24/21/14/15
  of 30 = 74/120**, no better than step 13000 (75) or 17000 (76) on the same seeds. Averaging neither adds nor
  keeps skills, so the step-to-step swings are mostly sampling noise plus checkpoints failing different
  episodes. EMA is now expected to add little: pi05_2arm_ema moved to the end of the H100 queue.
  **New per-object metrics (first run with them) locate the two_bins bottleneck: the first (left-arm) cube
  reaches the bin 28/30, the second (right-arm) cube 15/30.** Of 13 second-cube-only failures, 8 never lift it
  (the hover seen on video) and 5 lift it but fail to place. A spatial check looked striking for the soup
  (second cube spawned farther from the bases, y ≥ 0.01: 5/19; nearer: 10/11) but **does not replicate** on
  steps 13000 (3/11 near vs 6/19 far) or 17000 (7/11 vs 8/19), so position is not the cause.
  **H11 (open):** the policy drives both arms at once, while every teacher demo makes the second arm wait
  above its cube until the first is home, so the right arm meets states the data never shows. Implied changes:
  (a) DAgger restricted to two_bins with later takeovers, which labels exactly those states: queued
  **dagger5_twobins** (400 episodes from step 17000, takeover 2–18 s) on A10#1 and **pi05_dagger2** (resumes
  pi05_dagger1 with complete DAgger v4 + v5); (b) if DAgger does not lift the second cube, a teacher style
  where both arms move concurrently (needs collision handling at the shared bin). Watching (a) first.
  Steps: pi05_v1 19900 (dagger1 launches next), 2arm_v2 10940, warm2arm 9220, recover 9000, easy 9460; no
  errors. DAgger v4 at two_bins 60/150.
- **2026-10-16 08:13** [scheduler] 129.213.124.160 was idle; launched **dagger5_twobins** (H11: the second arm is the two_bins bottleneck (first cube in bin 28/30, second 15/30 for the pi05_v1 soup); the policy moves both arms at once, reaching right-arm states the wait-then-go teacher never shows. DAgger on two_bins only from step 17000, takeover 2-18 s to cover second-arm failures)
- **2026-10-16 12:25 UTC** **pi05_v1 step 19000: 7/8, 5/8, 5/8, 5/8**; the run finished at 20000. dagger5_twobins
  launched on A10#1. Caught before launch: the main box still cannot render with EGL (EGLError even with
  libnvidia-gl-580-server installed), so the on-box evaluation in pi05_dagger1/pi05_dagger2 would have
  crashed. Removed it; their checkpoints are relayed from the Mac to A10#1 instead
  (`relay_checkpoints.sh` + `eval_ladder_when_ready.sh`, 8 episodes per rung, the same protocol as pi05_v1).
- **2026-10-16 08:32** [scheduler] launch of pi05_dagger1 on 192.222.52.191:0 failed: job tag SCHEDJOB_pi05_dagger1 not seen on 192.222.52.191 within 3 minutes
