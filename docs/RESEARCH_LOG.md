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
- **2026-10-16 12:35 UTC** **pi05_dagger1 is training** on main GPU0 (resumes pi05_v1 step 20000 on
  `sim_mixdag_v1`, 12,878 episodes: mix + recover2 + DAgger v1–v4; first loss 0.056). The scheduler had
  marked it failed: setup took over 3 minutes, and `train_pi05.sh` backgrounds the trainer so its tagged
  shell exits at once, so the tag check could never pass for an H100 job. Fixed `scripts/scheduler.py`: wait up
  to 15 minutes and accept a filled GPU as proof of launch; status corrected; scheduler restarted (dry run
  clean). Its checkpoints relay to A10#1. Others: 2arm_v2 11500, warm2arm 9540, recover 9280, easy 9820; no
  errors. DAgger v4 at two_bins 100/150; dagger5_twobins 60/400 (33 teacher successes of 43 kept).
- **2026-10-16 08:49** [scheduler] 132.145.199.243 was idle; launched **twobins_mixed_generate** (H10 data: two_bins demos with either arm placing first (half also with recover2 knocks), 1,500 episodes (deprioritised 11:30 UTC: the video shows failures are missed-grasp recovery, not arm order; the recover2 half of this data still helps))
- **2026-10-16 12:55 UTC** pi05_2arm_v2 finished (12000; step 11000: 8/12, 8/12, 4/12, 4/12, its best);
  pi05_easy step 10000: 14/16, 14/16 (best; H1b curriculum looks good, stage 2 queued); pi05_dagger1 at step 480
  (loss 0.031). DAgger v4 done (472 kept). **Caught before launch:** main GPU1 was free and the next entries
  would have failed there (prep ssh's to the main box from itself → host-key failure; on-box EGL eval). Fix:
  `scripts/scheduler.py` gains `avoid_hosts` and a Mac-side `mac_after_launch` hook. pi05_v1_rec2 is now
  main-box native (resumes runs/pi05_v1/step_0020000 directly; checkpoints relayed to A10#2 for 8-episode
  ladders, started by the hook); pi05_dagger2 relays to A10#1 the same way; curriculum2 / lr5e5 / 2arm_ema avoid
  the main box. Dry run picks pi05_v1_rec2 for main GPU1. Queued **v2arm_11000_confirm30** (A10): does the
  warm start match pi05_v1 on the same 30 seeds — decides the base for later DAgger rounds.
- **2026-10-16 08:58** [scheduler] 192.222.52.191:1 was idle; launched **pi05_v1_rec2** (H2 on the best model: continue pi05_v1 (6/8 and 5/8 at step 9000) on the two-arm mix plus recover2 (bumps, missed grasps) | 11:30 UTC: promoted — the step-17000 two_bins video shows missed grasps with no retry, which recover2's deliberate missed grasps teach)
- **2026-10-16 09:09** [scheduler] 132.145.199.243 was idle; launched **v2arm_11000_confirm30** (H1a/H8: does the warm-started pi05_2arm_v2 (step 11000: 8/12, 8/12, 4/12, 4/12 at 11k steps) match pi05_v1 (75-76/120 at 13-17k) on the same 30 seeds? Decides the base for later DAgger rounds)
- **2026-10-16 13:45 UTC** pi05_2arm_v2 final step 12000: 8/12, 6/12, 5/12, 5/12 (two_bins 5/12 is its best).
  twobins_mixed_generate done: clean mixed-order demos succeed 748/750 with either arm first, but the
  mixed_order_recover2 style only 356/750. **Correction to H11:** the teacher does move both arms at once
  during approach and grasp (recorded subtasks pair "Lower the left…" with "Close the right…"). Only the carry
  waits, so "the policy runs both arms in parallel" is not out of distribution by itself.
  **New H12 (data): failed scripted demos end with the hands parked until the time limit.** Measured on
  sim_mix_v1 + sim_recover2_v1 (both hands within 1 cm of their final position):

  | task | failed demos | parked > 10 s |
  | --- | --- | --- |
  | two_bins | 962 | 425 |
  | handover | 668 | 179 |
  | pick_place_any | 288 | 94 |
  | bin | 295 | 85 |

  Successful demos almost never park (≤ 7 per task). Parked tails are **13.2% of all training samples and
  20% of two_bins steps**. In two_bins recover2 timeouts the teacher's last subtask is mostly "Hold still"
  for ~25 s. The quality label marks these episodes low, but the prompt-dropout plan sometimes hides it. A
  policy that learned "park and wait" from them would produce exactly the hover in the step-17000 video.
  Change implied: sample at most 1 s of a failed episode's parked tail. Implemented as
  `PI_TRIM_STILL_TAIL_S` in `pi_model/data.py` (`trim_still_tail`, tests in `tests/test_trim.py`; on
  sim_mixrec2_v1 it trims 1,474 failed episodes). Queued **pi05_v1_rec2_trim**, an exact A/B twin of the
  running pi05_v1_rec2 (same resume step 20000, data and 8000 steps) with trimming on, first for the next
  free node. pi05_dagger2 now also takes the 1,500 mixed-order two_bins demos and **dagger6_generate** (A10,
  from pi05_2arm_v2 step 12000). Trimming will be applied to everything else only if the A/B supports it.
- **2026-10-16 13:30 UTC** pi05_warm2arm step 10000: 8/16, 10/16, 5/16, 2/16 (10 checkpoints; still overlaps
  the leader's lower bounds by a hair on every rung, ends at 12000 in ~1 h, so it runs out). v2arm_11000_confirm30
  half done: pick_place_any 17/30 (only 14 lifted) and bin 19/30, against pi05_v1 step 13000's 27 and 22 on the
  same seeds. The 12-episode ladder flattered pi05_2arm_v2, and pi05_v1 stays the base for DAgger rounds.
  pi05_dagger1 step 1000 relayed to A10#1 and is being scored; pi05_v1_rec2 at 880. dagger5_twobins 380/400. No
  errors.
- **2026-10-16 13:50 UTC** **v2arm_11000_confirm30 final (strict): 14/19/12/7 of 30 = 52/120**, against pi05_v1
  step 13000's 75/120 on the same seeds. The warm start plus recover2 reached a weaker policy, not the same one
  sooner. On 12-episode ladders it looked equal (24/48), so 12 episodes misrank runs by about this much.
  pi05_dagger1 step 1000: 4/8, 2/8, 0/8, 1/8 (7/32, down from 22/32 at its resume point; the H8 warm-start
  dip, judged after ≥ 4 checkpoints). pi05_recover step 10000: 7/16, 8/16, 4/16, 2/16 (flat; still overlaps the
  leader). pi05_easy step 11000: 12/16, 11/16. dagger5_twobins done (293 kept, 210 teacher successes). Queued
  **v1_20000_confirm30** on A10#1: the resume point of the DAgger, recover2 and trim runs, on the 30 shared seeds,
  as their baseline.
- **2026-10-16 09:50** [scheduler] 129.213.124.160 was idle; launched **v1_20000_confirm30** (Baseline for H2b/H12: pi05_v1 step 20000 (the resume point of pi05_dagger1, pi05_v1_rec2 and pi05_v1_rec2_trim) on the same 30 seeds, with per-object two_bins metrics, so their final checkpoints can be compared episode for episode)
- **2026-10-16 09:51** [scheduler] 132.145.199.243 was idle; launched **dagger6_generate** (H2b/H11 data diversity: DAgger rollouts from a second, differently trained policy (pi05_2arm_v2 final step 12000: 8/12, 6/12, 5/12, 5/12) for pi05_dagger2)
- **2026-10-16 14:05 UTC** **pi05_dagger1 is collapsing: 22/32 at its resume point → 7/32 (step 1000) → 3/32
  (step 2000: 1/8, 1/8, 1/8, 0/8), with the hand stopping 12–16 mm from the cube instead of 5–8 mm.**
  pi05_v1_rec2, resumed from the same checkpoint with the same warmup, only dipped to 9/32 at step 1000, so the
  warmup alone (H8) does not explain it. Audit of the DAgger sets it added:

  | set | episodes | teacher failed | parked share of steps |
  | --- | --- | --- | --- |
  | v1 (pi05_v1 8000) | 532 | 19% | 37% |
  | v2 (2arm_v2 in its dip) | 570 | 70% | 82% |
  | v3 (warm2arm 4000) | 546 | 14% | 29% |
  | v4 (pi05_v1 15000) | 436 | 53% | 73% |
  | all | 2,084 | 38% | **62%** |

  Two_bins continuations failed 100% (v2) and 95% (v4). So the joint-space DAgger teacher often cannot recover
  from the policy's states, and those failures then park until the time limit. With the base data's own parked
  tails (H12), roughly a quarter of pi05_dagger1's samples show still hands. **H13 (open):** the DAgger
  teacher, not DAgger itself, is the problem. Implied changes: (a) train only on teacher-successful
  continuations, with parked tails trimmed; (b) fix the takeover, e.g. re-sync the teacher's phases from state
  before it acts. Acting on (a) now: **pi05_dagger_clean** (resumes pi05_v1 step 20000; mix + recover2 +
  successful DAgger v1–v6 + successful mixed-order two_bins demos; `PI_TRIM_STILL_TAIL_S=1`) is queued second
  for nodes. pi05_dagger2 (which would resume the collapsing run) is blocked. pi05_dagger1 keeps running to ≥ 4
  checkpoints per the protocol.
- **2026-10-16 14:10 UTC** **pi05_v1 step 20000 on the 30 shared seeds (3 of 4 rungs): pick_place_any 26/30,
  bin 26/30, handover 19/30**, against step 13000's 27/22/17 and 17000's 23/22/16. The final checkpoint is the best
  pi05_v1 so far, and it is the resume point of every running continuation, so it is their baseline. two_bins
  still running. Steps: dagger1 2780, v1_rec2 2040, warm2arm 11220, recover 10940, easy 11580; DAgger v6 at bin
  60/100. No errors.
- **2026-10-16 14:55 UTC** **pi05_v1 step 20000 confirmed: 26/26/19/12 of 30 strict = 83/120, the best two-arm
  policy** (step 13000: 75, 17000: 76). two_bins: first cube in bin 27/30, second 12/30; in 10 of the 18
  failures the second cube is never lifted. pi05_recover step 11000: 12/16, 11/16, 7/16, 6/16 (36/64, its best
  by far, after a flat 8000–10000; the ≥ 4-checkpoint rule kept it alive). pi05_v1_rec2 step 2000: 4/8, 4/8,
  1/8, 1/8 (10/32; still in its post-resume dip). pi05_warm2arm step 11000: 13/16, 9/16, 6/16, 3/16.
  **H13 confirmed and fixed: the DAgger takeover was broken, not DAgger itself.** Diagnostic: the teacher
  takes over from a clean reset on 10/10 episodes for every task. When the teacher drives for T seconds and a
  fresh DAgger-style teacher then takes over (exactly dagger.py's takeover), two_bins fails 8/8 at T=4 s and
  handover fails 8/8 at T=4 s and 6 s. Bin, and the other takeover times, stay at 8/8. Cause: `ArmPickPlace` set each
  arm's rest pose to the hand position when the teacher was created, so a mid-carry takeover sent the arm
  "home" to a mid-air point and the other arm waited forever. Fix: `TaskEnv.reset` records
  `home_positions`, which the teacher uses. All takeovers now 8/8 (two_bins and handover at 2/4/6/9 s); task
  tests 5/5; new `tests/test_dagger_takeover.py`. Normal recording and the eval oracle build their teacher at
  reset, so they were never affected. This explains the 62% parked DAgger data and plausibly pi05_dagger1's
  collapse. Queued **dagger7_fixed** (A10#1, 150 per task from pi05_v1 step 20000, takeover 1–15 s);
  pi05_dagger_clean now also takes v7.
- **2026-10-16 10:42** [scheduler] 129.213.124.160 was idle; launched **dagger7_fixed** (H13 fixed: the DAgger teacher took the hand position at takeover as each arm's rest pose, deadlocking two_bins mid-carry and handover lower/raise takeovers (0/8 each; 8/8 after the fix). Regenerate DAgger from the best policy (pi05_v1 step 20000, 83/120) with the fixed teacher)
- **2026-10-16 14:50 UTC** pi05_easy finished: step 12000 pick_place_any 11/16, bin 15/16. pi05_curriculum2
  launched on its node (prep still copying the 47 GB recover2 set, so that GPU is briefly idle). **DAgger v7
  with the fixed teacher: 31 of the first 33 kept continuations succeed (94%)**, against 19–70% teacher
  failure in v1–v4. That supports H13's fix in production as well as in the diagnostic. dagger6 (old teacher) at
  two_bins 20/100. Steps: dagger1 3880, v1_rec2 3140, warm2arm 11880 (ends at 12000; its node then takes
  pi05_v1_rec2_trim), recover 11560. No errors.
- **2026-10-16 11:21** [scheduler] 209.20.158.194:0 was idle; launched **pi05_curriculum2** (H1b: easy rungs first learns fastest; stage 2 adds handover and two_bins plus recover2 data)
- **2026-10-16 15:25 UTC** **Scheduler bug fixed: H100 entries with both `train` and `eval` launched nothing.**
  `launch()` joined background steps with " ; ", giving `cmd & ; cd …`, a bash syntax error. The remote shell
  ran nothing, and the output went to /dev/null. pi05_curriculum2 (launched 14:47) sat idle for 30 min because of
  it, and pi05_v1_rec2_trim, pi05_dagger_clean, pi05_lr5e5 and pi05_2arm_ema would have hit the same bug. Fix:
  one step per line. Every H100 entry's launch string now passes `bash -n`; curriculum2 requeued and now loading
  data. pi05_warm2arm finished (step 12000: 8/16, 11/16, 4/16, 5/16); its node takes pi05_v1_rec2_trim next.
  **pi05_v1_rec2 step 3000: 4/8, 2/8, 3/8, 0/8 (9/32), the third checkpoint far below its 22/32 resume point**
  (7 → 10 → 9 of 32), and pi05_dagger1 is also low. **H14 (open):** resuming a converged policy with a fresh
  warmup to the full LR (2.5e-5) knocks it out of its basin, whatever the data. Implied change: continue at a
  lower peak LR (e.g. 5e-6). Watching 1–2 more checkpoints of both continuations (the 2arm_v2 dip recovered by
  ~6000) before queuing a low-LR continuation. Queued A10 **v1_20000_twobins_video** (6 full-video episodes) to
  check the second-arm hover on the best policy.
- **2026-10-16 15:35 UTC** **My H12 loader change broke every new training run.** I inserted the trim functions
  inside `Episode`'s class body, which pushed `Episode.frame_step` out of the class. Any trainer started after
  13:45 crashed on its first batch with `AttributeError: 'Episode' object has no attribute 'frame_step'`.
  Runs already in progress had imported the old module and were unaffected. The only casualty was
  pi05_curriculum2's first real launch. Fixed (diff against the pre-H12 file now shows only the intended
  additions), synced to all hosts, and pi05_curriculum2 relaunched by hand on 209.20.158.194 (step 20, loss
  normal). The crashed log is kept as `logs/train_pi05_curriculum2.crashed_frame_step`. Lesson: a compile check
  is not a test. Loader edits now get a short real training run before launch.
- **2026-10-16 15:36 UTC** pi05_v1_rec2_trim launched on 209.20.159.198 (prep copying the step-20000 checkpoint
  and recover2). pi05_curriculum2 is training (step 140). DAgger v6 done with the old teacher (329 kept, 157
  successful). DAgger v7 with the fixed teacher is at handover 120/150, with 243 of 273 kept continuations
  successful (89%; was 47–81% failure in v2/v4). Queued **dagger8_fixed** on A10#2 (from pi05_v1 step 15000,
  whose export is already there), which also feeds pi05_dagger_clean. Steps: dagger1 5180, v1_rec2 4460, recover
  12300. No errors.
- **2026-10-16 11:46** [scheduler] 209.20.159.198:0 was idle; launched **pi05_v1_rec2_trim** (H12: failed scripted demos end with both hands parked for 10-25 s until the time limit (44% of failed two_bins demos; 13% of all samples, 20% of two_bins steps); sampling them teaches the policy to stop, the hover seen in two_bins failures. A/B twin of pi05_v1_rec2 (same resume, data, steps) with PI_TRIM_STILL_TAIL_S=1)
- **2026-10-16 15:55 UTC** **Live interactive demo** (user request): `pi_model/live_demo.py` serves pi05_v1
  step 20000 (bf16, 5 denoise steps, oracle subtasks, 25 actions per chunk, the confirmation settings) on A10#2
  (132.145.199.243), bound to 127.0.0.1:8765 and reached through an SSH tunnel. The user drags the cube(s) and
  the destination on a top-down table map, picks one of the four tasks, and watches the three cameras live at
  real-time pace. Checks through the API: two_bins succeeded in 9.5 s, bin in 10.5 s; training-distribution
  placements gave pick_place_any 3/4 and handover 2/3, in line with the 30-episode rates, so the demo's
  placement path is faithful. A10#2 is reserved for it: dagger8_fixed is held, and the server runs under a
  SCHEDJOB tag so the scheduler leaves the GPU alone.
- **2026-10-16 15:53 UTC** **Correction to the 14:05 entry: pi05_dagger1 did not collapse for good.** Step
  5000: 7/8, 6/8, 3/8, 4/8 (20/32, back near its 22/32 resume point, with two_bins above it). pi05_v1_rec2 step
  4000: 5/8, 6/8, 1/8, 5/8 (17/32). Both continuations show the H8 dip and recovery (lows of 3/32 and 9/32 near
  steps 2000–3000). So the bad DAgger data was not the main cause of the dip, and H14 (resume LR) is weakened.
  The takeover bug (H13) was real and is fixed regardless; v7 with the fixed teacher is at 315/361 successful.
  pi05_v1_rec2_trim (the H12 A/B twin) is training with trimming active (1,474 failed episodes trimmed); its first
  checkpoints come in about an hour. pi05_curriculum2 at step 440. The live demo holds A10#2. No errors.
- **2026-10-16 16:10 UTC** **DAgger v7 (fixed teacher) done: 420 kept, 362 teacher-successful (86%)**, against
  14–70% failure per set with the broken takeover. The live demo gained an orbitable full-scene 3D view at the
  user's request (free MuJoCo camera, 1152×648; the policy's inputs are unchanged). Queued for A10#1 after the
  step-20000 video review: **dagger9_fixed** (more fixed-teacher data from step 20000, new seeds, also feeds
  pi05_dagger_clean) and **dagger1_final_confirm30** (runs once step 8000 is relayed; same 30 seeds as the
  83/120 baseline). Steps: dagger1 6120, v1_rec2 5420, rec2_trim 380, curriculum2 740, recover 12880. No errors.
- **2026-10-16 12:12** [scheduler] 129.213.124.160 was idle; launched **v1_20000_twobins_video** (H11/H12 evidence: is the second-arm hover still the main two_bins failure for the best policy (pi05_v1 step 20000: second cube 12/30, never lifted in 10 of 18 failures)?)
- **2026-10-16 12:22** [scheduler] 129.213.124.160 was idle; launched **dagger9_fixed** (H2b/H13: more fixed-teacher DAgger from the best policy (pi05_v1 step 20000), new seeds, for pi05_dagger_clean)
- **2026-10-16 16:35 UTC** **Video review of pi05_v1 step 20000 on two_bins (6 episodes, 3 succeed).** In two
  failures the left cube is placed, and the right gripper sits open about 2 cm off the blue cube for 20 s,
  prompted "move above" (the oracle's xy-near test fails, so it never says "lower"). In the third, the right arm
  holds its cube and waits, as the teacher does, while the left arm never grasps its own, so the episode
  deadlocks. All three are an arm stuck off-centre during approach and never corrected: the states the
  fixed-teacher DAgger (v7 86% and v9 teacher success) labels. Therefore **pi05_dagger_clean is now main-box
  native** (resumes step 20000 on the main box when pi05_dagger1 ends in ~45 min; evaluation relayed to
  A10#1) instead of waiting ~4 h for a node. Added **pi05_v1_cont** (H16: continue pi05_v1 on its own data with
  trimming at LR 1e-5; pi05_v1 was still improving at 20k: 75 → 76 → 83/120) for main GPU1 when pi05_v1_rec2
  ends. It is the baseline the new-data continuations must beat. pi05_dagger1 step 6000: 6/8, 6/8, 4/8, 4/8;
  pi05_v1_rec2 step 5000: 5/8, 4/8, 2/8, 3/8. Live demo on A10#2 (tunnel; public access needs a provider
  firewall rule). No errors.
- **2026-10-16 17:15 UTC** pi05_v1_rec2 step 6000: 6/8, 4/8, 7/8, 5/8 (22/32, its best; back at its resume
  point, with handover 7/8 and two_bins 5/8 above it). pi05_curriculum2 step 1000: 8/16, 6/16, 1/16, 0/16 (first
  contact with the two-arm rungs plus the resume dip). pi05_dagger1 at 7960, ending now; the scheduler hands
  main GPU0 to pi05_dagger_clean, and main GPU1 goes to pi05_v1_cont when pi05_v1_rec2 ends at 8000. Final
  checkpoints get 30-seed confirmations (dagger1_final_confirm30 queued). DAgger v9 at handover 120/150 (217/248
  successful). rec2_trim at 1500 (it shares its GPU with on-node evaluation, so it runs slower). No errors.
- **2026-10-16 13:33** [scheduler] 192.222.52.191:0 was idle; launched **pi05_dagger_clean** (H2b done right (H12 corollary): pi05_dagger1 collapsed (22/32 at resume -> 7/32 -> 3/32) on data where 38% of DAgger continuations are teacher failures and 62% of DAgger steps are parked tails. Resume pi05_v1 step 20000 on mix + recover2 + only teacher-successful DAgger (v1-v6) and mixed-order two_bins demos, with PI_TRIM_STILL_TAIL_S=1; plus DAgger v7 from the fixed teacher | 16:35 UTC: made main-box native (the step-20000 video shows the remaining two_bins failures are arms hovering off-centre without grasping, the states fixed-teacher DAgger labels); evaluation relayed to A10#1)
- **2026-10-16 17:35 UTC** pi05_dagger1 finished (step 8000 relayed to A10#1; its 30-seed confirmation runs when
  dagger9 frees that A10). **pi05_dagger_clean is training** on main GPU0: 14,170 episodes (mix 4,800 +
  recover2 5,994 + teacher-successful DAgger v1–v7/v9 2,272 + mixed-order two_bins 1,104), 1,474 failed tails
  trimmed. It resumes from pi05_v1 step 20000, and prep took ~8 min to pull 14 GB from the A10s. pi05_v1_rec2 step
  7000: 5/8 (4 lifted), 8/8, 2/8, 3/8; it ends at 8000, and main GPU1 then takes pi05_v1_cont. DAgger v9 at two_bins
  80/150 (287/336 successful). curriculum2 2080, rec2_trim 1880, recover 14400. No errors.
- **2026-10-16 17:45 UTC** **pi05_dagger1's last checkpoints are the best ladder scores yet: step 7000 8/8,
  8/8, 7/8, 4/8 (27/32), step 8000 8/8, 8/8, 4/8, 4/8 (24/32)**, against 22/32 for its resume point pi05_v1 step
  20000. So DAgger helped despite the broken-takeover data, once past the H8 dip (7 → 3 → … → 20 → 20 → 27 →
  24 of 32). Queued **dagger1_7000_confirm30** ahead of the step-8000 confirmation; both use the 30 shared seeds
  against the 83/120 baseline on A10#1. If DAgger holds up on 30 seeds, pi05_dagger_clean (fixed-teacher and
  success-only data) is the next candidate to beat it.
- **2026-10-16 13:47** [scheduler] 192.222.52.191:1 was idle; launched **pi05_v1_cont** by hand at the user's request (H16: pi05_v1 was still improving at its last step (30-seed strict 75 -> 76 -> 83/120 at 13k/17k/20k). Plain continuation)
- **2026-10-16 13:48** [scheduler] 129.213.124.160 was idle; launched **dagger1_7000_confirm30** by hand at the user's request (H2b: pi05_dagger1 step 7000 scored 27/32 on the 8-episode ladder (8, 8, 7, 4), the best yet; confirm on the 30 shared se)
- **2026-10-16 17:50 UTC** All GPUs busy (user asked to arm idle ones). pi05_v1_cont is on main GPU1 (420 of
  sim_mix_v1's failed episodes trimmed; LR 1e-5), and dagger1_7000_confirm30 is on A10#1. DAgger v9 done: 395
  kept, 334 successful (85%). pi05_v1_rec2 finished at 8000. Queue restocked: **pi05_dagger2** unblocked and
  repurposed to resume the best checkpoint (pi05_dagger1 step 7000) on pi05_dagger_clean's cleaned data. It
  avoids the main box and evaluates on its node, so dagger_clean vs dagger2 separates the effect of the resume
  point. Also queued **v1_rec2_final_confirm30** (its export is copied from A10#2 to A10#1 in prep). H100
  runnable: dagger2, lr5e5, 2arm_ema; A10: two confirmations.
- **2026-10-16 18:05 UTC** **Standing rule from the user: no GPU may idle; always keep experiments queued.**
  Changes so the queue covers every slot:
  1. `scripts/scheduler.py` checks every 60 s (was 300 s), so a freed GPU is refilled in about 2 min.
  2. Entries may carry a `main_variant` (no EGL evaluation on the main box; relay to an A10 instead), so node
     experiments can also fill main-box GPUs. pi05_dagger2, pi05_lr5e5 and pi05_2arm_ema all now run on main or
     nodes (checked: runnable and `bash -n` clean on both).
  3. Status updates re-read the queue and change one entry, so manual edits made while it sleeps are never
     overwritten, and the in-memory status blocks a second launch of the same entry in a round.
  4. A slot that is idle with nothing runnable writes `outputs/IDLE_ALERT`, which the event watcher turns into an
     immediate wake-up to write a new experiment.
  A10 confirmations copy their checkpoint to whichever A10 is free (only if missing). Queued **dagger10_fixed**
  (fixed-teacher DAgger from the best checkpoint, pi05_dagger1 step 7000). Every GPU is busy now.
- **2026-10-16 18:10 UTC** **dagger1_7000_confirm30, 3 of 4 rungs (strict, 30 shared seeds): pick_place_any 25/30
  (28 including 3 pushes), bin 28/30, handover 18/30**, against pi05_v1 step 20000's 26/26/19. Tied so far
  (71 vs 71). The ladder's 27/32 was the luck of 8 episodes. two_bins (baseline 12/30) decides it; the step-8000
  confirmation follows. All GPUs busy; scheduler on 60 s checks with no idle alerts. Steps: dagger_clean 1020,
  v1_cont 620, rec2_trim 2440, curriculum2 2620, recover 15040. No errors.
- **2026-10-16 14:15** [scheduler] 129.213.124.160 was idle; launched **dagger1_final_confirm30** (H2b: does pi05_dagger1's final checkpoint beat its resume point (pi05_v1 step 20000: 83/120) on the same 30 seeds?)
- **2026-10-16 18:30 UTC** **dagger1_7000_confirm30 final (strict, 30 shared seeds): 25/28/18/14 = 85/120**, against
  pi05_v1 step 20000's 26/26/19/12 = 83/120. That is within noise, so **DAgger round 1 (broken-takeover data) gave
  no measurable gain**. two_bins: first cube in 27/30, second 14/30; in 8 of the 16 failures the second cube is
  never lifted, the same hover as before. The open bet is the fixed-teacher, success-only data in
  pi05_dagger_clean (step 1580) and pi05_dagger2 (queued); the step-8000 confirmation is half done (pick 24 of 26
  lifted, bin 26/30). H11/H12 still open; the trim A/B (rec2_trim, step 2820) reports in a few hours. No idle GPUs.
- **2026-10-16 18:50 UTC** **pi05_v1_cont step 1000: 6/8, 5/8, 5/8, 5/8 (21/32), with no post-resume dip** (its resume
  point scored 22/32). It continues on the same data at LR 1e-5. Every LR-2.5e-5 continuation on new data dipped to
  7–12/32 at step 1000. pi05_dagger_clean step 1000: 6/8, 4/8, 1/8, 1/8 (12/32). **H14 revived:** the full-LR restart
  (and/or the new data) causes the dip. Queued **pi05_dagger_clean_lr1e5** (same as dagger_clean at LR 1e-5)
  ahead of lr5e5/ema; it runs on main or a node. H100 runnable: dagger2, dagger_clean_lr1e5, lr5e5, 2arm_ema.
- **2026-10-16 14:47** [scheduler] 129.213.124.160 was idle; launched **v1_rec2_final_confirm30** (H2: pi05_v1_rec2 final (step 8000) vs its resume point (83/120) on the 30 shared seeds)
- **2026-10-16 18:50 UTC** **dagger1_final_confirm30 (step 8000, strict): 24/26/16/14 = 80/120** (step 7000: 85; baseline
  pi05_v1 step 20000: 83). DAgger round 1 is confirmed as no gain; pi05_v1 step 20000 stays the reference and the demo
  model. v1_rec2_final_confirm30 running on A10#1. Steps: dagger_clean 2140, v1_cont 1760, rec2_trim 3140, curriculum2
  3240, recover 15760. No idle GPUs; queue has 4 H100 and 1 A10 runnable (dagger10_fixed).
- **2026-10-16 19:10 UTC** **v1_rec2_final_confirm30 (pi05_v1_rec2 step 8000), 3 of 4 rungs: pick_place_any 24/30,
  bin 20/30, handover 16/30** against the baseline's 26/26/19. Continuing the best policy on mix + recover2 at full LR
  made it worse (−11 on three rungs), despite its 22/32 ladder peak at step 6000. This fits H12 (recover2 is ~half
  failed demos with parked tails) and H14 (full-LR restart). The pi05_v1_rec2_trim A/B (step 3480) and the
  pi05_dagger_clean vs pi05_dagger_clean_lr1e5 pair test the two. pi05_v1_cont (same data, LR 1e-5) is the control
  that so far holds its level. All GPUs busy; no alerts.
- **2026-10-16 15:18** [scheduler] 129.213.124.160 was idle; launched **dagger10_fixed** (H2b next round: fixed-teacher DAgger from the best checkpoint so far (pi05_dagger1 step 7000, 27/32), whose failure states differ from pi05_v1's; data for the round after pi05_dagger2)
- **2026-10-16 19:30 UTC** **v1_rec2_final_confirm30 final: 24/20/16/13 = 73/120** (baseline 83): the recover2
  continuation is confirmed worse. Scoreboard of continuations from pi05_v1 step 20000 on the 30 shared seeds:
  dagger1 step 7000 85, step 8000 80, rec2 step 8000 73; baseline 83. Nothing has beaten the baseline yet. Pending:
  dagger_clean (3260), v1_cont (2900, low LR), rec2_trim (3840), dagger_clean_lr1e5 and dagger2 (queued).
  `scripts/scheduler.py` gained `requires_elsewhere` (wait for a file on another host), so a confirmation can
  wait for a checkpoint relayed to the other A10. Queued **v1_cont_4000_confirm30**. DAgger v10 running (40/42
  successful so far). All GPUs busy.
- **2026-10-16 19:35 UTC** pi05_dagger_clean step 2000: 3/8, 0/8, 3/8, 1/8 (7/32), the same full-LR dip as every
  other 2.5e-5 continuation (5 of 5), while pi05_v1_cont at 1e-5 stays at 21 → 19/32. Acting on H14: queue reordered so
  **pi05_dagger_clean_lr1e5** launches first, and **pi05_dagger2 now also uses LR 1e-5** (so the pair isolates
  the resume point instead of repeating a likely-harmful LR).
- **2026-10-16 16:52** [scheduler] 129.213.124.160 was idle; launched **v1_cont_4000_confirm30** (H16/H14: does the low-LR continuation of pi05_v1 on its own data (no dip, 21 and 19/32 on the ladder) improve on the 83/120 baseline by step 4000? Same 30 seeds)
- **2026-10-16 18:22** [scheduler] 192.222.52.191:0 was idle; launched **pi05_dagger_clean_lr1e5** (H14 test: pi05_v1_cont (same data, LR 1e-5) showed no post-resume dip (21/32 at step 1000), while every LR-2.5e-5 continuation fell to 7-12/32. Same as pi05_dagger_clean (resume pi05_v1 step 20000, cleaned DAgger data, trimmed) but LR 1e-5, which separates the LR effect from the new-data effect)
- **2026-10-16 18:34** [scheduler] 192.222.52.191:1 was idle; launched **pi05_dagger2** (H2b round 2: resume the best checkpoint so far (pi05_dagger1 step 7000, 27/32 on the ladder) on the cleaned data of pi05_dagger_clean (mix + recover2 + teacher-successful DAgger v1-v9 + mixed-order two_bins, tails trimmed). Compare with pi05_dagger_clean, which resumes pi05_v1 step 20000 on the same data | 19:35 UTC: switched to LR 1e-5 (every LR-2.5e-5 continuation dipped to 3-12/32; pi05_v1_cont at 1e-5 did not), so it pairs with pi05_dagger_clean_lr1e5 and isolates the resume point)
- **2026-10-16 19:05** [scheduler] 209.20.157.232:0 was idle; launched **pi05_lr5e5** (H5: the two-arm mix is under-trained at 2.5e-5; 5e-5 learns faster without destabilising)
- **2026-10-16 19:13** [scheduler] 209.20.157.232:0 was idle; launched **pi05_2arm_ema** (H8/H9: checkpoint-to-checkpoint swings (pi05_2arm_v2 4/12→0/12, pi05_warm2arm 11/16→5/16) come from raw weights; an EMA (0.999) of the 2arm_v2 recipe stabilises them | 12:15 UTC: demoted — averaging pi05_v1 steps 13000-18000 gave no gain (74/120 vs 75/76), so EMA is expected to help little)
- **2026-10-17 01:25** [scheduler] 192.222.52.191:0 was idle; launched **pi05_v1_cont2** (H14/H16: the low-LR continuation (pi05_v1_cont) never dipped and reached two_bins 7/8 at step 8000; continue it at LR 1e-5 on the cleaned DAgger data (trimmed) for 8k more steps)
- **2026-10-17 01:25** [scheduler] 209.20.159.198:0 was idle; launched **pi05_v2_mixrec2** (H12 at full scale: train from the π0.5 base on mix + recover2 with failed tails trimmed for 20k steps (pi05_recover used recover data untrimmed and reached 36/64 at best); compare with pi05_v1 and pi05_v2_clean)
- **2026-10-17 01:28** [scheduler] 209.20.158.194:0 was idle; launched **pi05_v1_cont_more** (H16: does plain low-LR training on pi05_v1's own data keep improving? Continue pi05_v1_cont step 8000 on sim_mix_v1 at LR 1e-5 (trimmed) for 8k more steps)
- **2026-10-17 01:28** [scheduler] 129.213.124.160 was idle; launched **dagger2_7000_confirm30** (Confirm pi05_dagger2 step 7000 (28/32 on the 8-episode ladder, its best) on the 30 shared seeds against the 83/120 baseline)
- **2026-10-17 01:37** [scheduler] 192.222.52.191:1 was idle; launched **pi05_v1_cont2** (H14/H16: the low-LR continuation (pi05_v1_cont) never dipped and reached two_bins 7/8 at step 8000; continue it at LR 1e-5 on the cleaned DAgger data (trimmed) for 8k more steps)
- **2026-10-17 05:45 UTC** **GPUs sat idle for hours overnight: the user's Mac slept, which stopped the scheduler
  and the watcher.** Idle from (local time): A10#1 17:19, 209.20.159.198 19:30, 209.20.158.194 21:51, main GPUs 23:08.
  Fixes: `caffeinate -i -w <scheduler pid>` keeps the Mac from idle-sleeping while the scheduler runs, and the
  scheduler now waits 20 min after a launch before calling that slot idle again (the grace period is kept across
  restarts in `outputs/last_launch.json`). pi05_2arm_ema had been double-booked onto 209.20.157.232 while
  pi05_lr5e5 loaded (OOM), and pi05_v1_cont2 onto main GPU0 next to pi05_v2_clean after a restart (OOM, relaunched
  on GPU1). Results finished while asleep (8-episode ladder, best checkpoint by total): pi05_dagger2 step 7000
  28/32, pi05_v1_cont step 8000 26/32, pi05_dagger_clean step 8000 25/32, pi05_dagger_clean_lr1e5 step 5000 24/32,
  pi05_recover finished (12000: 13/16 pick), pi05_v1_rec2_trim and pi05_curriculum2 finished. v1_cont step 4000 on
  30 seeds: 26/26/17/15 = 84/120 (baseline 83). **Now running:** pi05_v2_clean (from base, cleaned data, 20k),
  pi05_v1_cont2 (v1_cont 8000 + cleaned data at 1e-5), pi05_v2_mixrec2 (from base, mix+recover2 trimmed, 20k),
  pi05_v1_cont_more (v1_cont 8000 on its own data at 1e-5), pi05_lr5e5 (209.20.157.232), and 30-seed confirmations
  (dagger2 step 7000 first). **Queued:** pi05_dagger3 (round 3 from dagger2 step 7000 + DAgger v10, LR 1e-5),
  pi05_v1_lr5e6 (H14 dose), **pi05_twobins2x** (H18: `PI_TASK_WEIGHTS=two_bins=2` in `pi_model/data.py`, checked by
  building the real training dataset and fetching samples: two_bins share 36% → 53%); A10: three confirmations
  and dagger11_fixed.
- **2026-10-17 06:00 UTC** **dagger2_7000_confirm30 (strict, 30 shared seeds): 28/28/12/18 = 86/120** (baseline 83).
  **two_bins 18/30 is the best yet** (baseline 12, previous best 15). The second cube reaches the bin 20/30 and is
  never lifted in only 3 failures (was 8–10). This is the first evidence that the fixed-teacher/success-only DAgger
  data plus a low-LR resume fixes the second-arm hover (H11/H13). But **handover fell 19 → 12**, so it is a
  trade. pi05_dagger2 = pi05_dagger1 step 7000 resumed at LR 1e-5 on the cleaned data. Queued a two_bins +
  handover video review of it (A10) to verify the successes and see the handover failures. pi05_lr5e5 step 6000:
  6/16, 10/16, 5/16, 4/16 (recovering from ~0; from-base run at 5e-5).
- **2026-10-17 01:57** [scheduler] 129.213.124.160 was idle; launched **v1_cont_8000_confirm30** (Confirm pi05_v1_cont step 8000 (26/32 on the 8-episode ladder, its best) on the 30 shared seeds against the 83/120 baseline)
- **2026-10-17 06:00 UTC** Check: all H100s training (v2_clean 900, v1_cont2 560, v2_mixrec2 540, v1_cont_more 460,
  lr5e5 6500); A10#1 started v1_cont_8000_confirm30; A10#2 holds the demo. Removed the dead pi05_2arm_ema
  evaluation waiter on 209.20.157.232. Queue: 3 H100 runnable (dagger3, v1_lr5e6, twobins2x); A10 has 4 queued
  (two_bins/handover video of dagger2 7000, two confirmations, dagger11). Scheduler is under caffeinate; no idle
  alert; no errors.
- **2026-10-17 06:10 UTC** **Training wrapped up at the user's request.** Scheduler, watcher and relays stopped;
  all training/evaluation jobs killed (the in-flight pi05_v2_clean, pi05_v1_cont2, pi05_v2_mixrec2,
  pi05_v1_cont_more, pi05_lr5e5 and the v1_cont step-8000 confirmation are abandoned; queue entries marked
  cancelled). **Final models** (bf16, copied to the Mac's `models/`; see `docs/INFERENCE.md`):

  | model | strict success on the 30 shared seeds (pick / bin / handover / two_bins) |
  | --- | --- |
  | pi05_dagger2 step 7000 (best two-arm) | 28 / 28 / 12 / 18 = 86/120 |
  | pi05_v1 step 20000 (baseline, best handover) | 26 / 26 / 19 / 12 = 83/120 |
  | pi05_single step 6000 (single arm) | pick_place 28/30 |

  Main lessons: (1) the π0.5 pretrained backbone was what made vision work; (2) the second arm in two_bins was
  the bottleneck, and fixed-teacher, success-only DAgger data with a low-LR (1e-5) resume was the first thing that
  moved it (12 → 18/30); (3) restarting from a converged checkpoint at full LR (2.5e-5) costs ~5k steps of dip;
  (4) 8-episode ladders misrank checkpoints, and only 30-seed confirmations are trustworthy; (5) failure-heavy
  recovery data with parked tails (recover2) made the best policy worse (73/120).
- **2026-10-17 07:20 UTC** **Release to Hugging Face (private):** `KuanKuanKuan/so101-pi07-sim-models` is complete (the
  three final models in full precision, 28.1 GB). `KuanKuanKuan/so101-pi07-sim-data` stopped at 40.7 of ~132 GB:
  "Private storage limit reached" (about 70 GB on this plan). Still missing (~93 GB, staged as tar parts in
  `~/hf_release/data` on 192.222.52.191): all of sim_recover2_v1, sim_ladder_v1 parts 000–001, sim_recover_v1
  part 000, sim_twobins_mixed_v1 part 000, sim_dagger_v4. Re-running `scripts/box/hf_upload_release.py` resumes
  once there is room (public repo or a bigger plan). Mac inference verified after the MPS fix (bin 2/2,
  two_bins 1/2, live demo OK). The three H100 nodes are unreachable (released); the HF token was removed from
  both A10s.
- **2026-10-17 13:45** **Training restarted: π0.7 add-ons on the π0.5 backbone (track A) plus a data loop (track B).**
  The final models are π0.5 with π0.7's prompts and data; this round adds the π0.7 pieces π0.5 lacks, one per arm,
  each against a matched control. Every arm resumes pi05_dagger2 step 7000 (86/120) on the frozen
  sim_mixdagclean_v1 at LR 1e-5 for 6000 steps, trimmed tails, seed 0; only the add-on switch differs
  (`SET=...` → `train.py --set`). Code: `pi_model/pi05_backbone.py` (all switches off = LeRobot's own path).

  | run | host | hypothesis |
  | --- | --- | --- |
  | x_control_s0 / x_control_s1 | 192.222.52.191 GPU 0 / 1 | C0/C1: plain continuation, two seeds (run-to-run noise) |
  | x_rtc_s0 | 192.222.55.167 GPU 0 | H19: training-time RTC holds success at delays 6 and 12 where the control drops |
  | x_mem_s0 | 192.222.55.167 GPU 1 | H20: MEM history (6 frames, 1 s) helps handover and two_bins |
  | x_goal_s0 | 192.222.55.167 GPU 2 | H21: goal images (oracle: scripted teacher 2 s ahead in a twin scene, an upper bound on a world model) |
  | x_ki_s0 | 192.222.55.167 GPU 3 | H22: knowledge insulation (FAST CE trains the VLM, expert behind a stop-gradient) |
  | dagger11_generate | A10 193.122.189.119 | track B: DAgger from the best policy on handover + two_bins, seeds 1100000+ |

  Checks on the real checkpoint (`scripts/box/check_pi05_addons.py`, all pass): add-on path and sampler equal
  LeRobot's with switches off; RTC at delay 0 equals plain and keeps the committed prefix; MEM with no valid past
  frame equals π0.5's encoder; KI gives exactly zero flow gradient to the VLM. FAST CE at the start is 15.3 nats
  (dataset tokenizer, 42 tokens/chunk) and 16.1 (π0.5's universal tokenizer, 82 tokens/chunk): the fine-tuned
  VLM kept no FAST knowledge, so KI uses the dataset tokenizer. Fixes found on the way: (1) `eval_sim --delay`
  never delayed models without RTC (the new chunk ran from index 0); now every model executes the committed
  actions first. (2) OSMesa evaluation works on every box (bin 4/4 in 34 s on the main box), so the "no EGL"
  relay is unnecessary; scoring is `scripts/box/confirm30.sh` (summary in `outputs/c30_summary.txt`) triggered by
  `scripts/box/c30_when_ready.sh` at steps 3000 and 6000 on the training GPU. The baseline is re-scored under
  OSMesa at delays 0/6/12 (base_dagger2_7000). Queued: seed-1 replicates of all four add-ons (an add-on must beat
  both controls), x_data_v11 (blocked on the data), dagger12_twobins_late on the A10. The scheduler is back on the
  new hosts; a 20-minute check is armed; the Mac runs `caffeinate -i -s`.
- **2026-10-17 14:10** Check 1 after the restart. Steps: control_s0 1120, control_s1 900 (shares GPU 1 with the
  baseline re-score), rtc 200, mem 120, goal 160, ki 180; no NaN, error or OOM. Speeds (samples/s): control 14.9,
  rtc 15.0, ki 12.7, goal 11.4, mem 8.6, so mem needs ~6.2 h for 6000 steps and ki ~4.2 h. KI's FAST CE is
  falling during warmup (13.4 → 10.6 by step 180, token accuracy 0.9% → 1.8%). Memory: mem 58 GB, ki 50 GB,
  goal 44 GB, rtc 42 GB, so the step-3000 evaluations fit beside them. A10: dagger11 handover 40/400, teacher
  success 35 of 39 kept, ~100 episodes/h (≈8 h for 800). Baseline re-score still running (no summary line yet).
  Scheduler and caffeinate alive, no idle alert.
- **2026-10-17 14:03** [scheduler] 193.122.189.119 was idle; launched **dagger12_twobins_late** (Track B data, H11 follow-up: the second arm hovering in two_bins happens late, so take over at 8-22 s on two_bins only (seeds 1200000+))
- **2026-10-17 14:30** **Baseline re-scored under OSMesa (30 seeds, delay 0): pi05_dagger2 step 7000 =
  29 / 27 / 16 / 18 = 90/120** (86 on the A10 with EGL; handover 16 vs 12). Renderer or GPU differences alone move
  the total by ~4 and handover by ~4, which sets the scale: an add-on needs a clear margin over both controls
  (scored on the same boxes) to count. Delays 6 and 12 are running next. KI trend (100-step means): as FAST CE falls
  13.0 → 9.3 → 6.4 → 5.9, the flow loss rises 0.026 → 0.033 → 0.099 → 0.098 and has plateaued for ~200 steps
  (other arms ~0.027). This is the risk noted at launch: the CE gradient rewrites the VLM features that the
  insulated expert reads. Not acting yet (watch rule); its step-3000 score decides. If KI loses, the follow-up is
  H22b: the same with the CE weighted 0.1, so the VLM moves slowly enough for the expert to keep up. Steps:
  control_s0 1680, control_s1 1380, rtc 780, mem 440, goal 580, ki 660; no errors. A10: the scheduler started
  dagger12_twobins_late beside dagger11 (GPU 19.6 GB, 86%); dagger11 handover 140/400, teacher success 110/130.
- **2026-10-17 15:45** Baseline at **delay 6: 24 / 26 / 16 / 16 = 82/120** (−8 from delay 0; pick and two_bins
  lose most), so a 120 ms lag costs a non-RTC policy real success: the bar for H19 is visible. Delay 12 is
  running. KI's flow loss is coming back down (0.098 at ~600 → 0.064 at ~2400) while CE settles at 4.8 (FAST token
  accuracy 13%); still 2.5× the other arms (~0.023). control_s0 passed 3000 (its score is running). Steps: control
  3640 / 3180, rtc 2820, mem 1600, goal 2120, ki 2380; no errors. A10: dagger11 finished handover and is at
  two_bins 100/400 (kept 447, teacher success 362 = 81%); dagger12 (late takeovers) 280/400, kept 167, success
  119 (71%), 113 skipped because the policy finished or failed before the takeover. Scheduler and caffeinate alive.
- **2026-10-17 15:55** No new scores (control_s0 step 3000 and the baseline at delay 12 are being scored; rtc
  passed 3000, its d0/d6/d12 scoring starts now). Steps: control 3820 / 3360, rtc 3040, mem 1720, goal 2280, ki
  2560 (flow 0.064, flat). A10: dagger11 two_bins 140/400, dagger12 320/400. All alive.
- **2026-10-17 16:15** dagger12_twobins_late done: 400 episodes, 245 kept, teacher success 173. It joins dagger11
  in one A/B (x_data_v11 now = successful takeovers of v11 + v12; one data version, one A/B). A10 queue refilled:
  **failmine_dagger2** (60 non-test seeds 5000+ on handover and two_bins with videos, to name failure types before
  more data) then **dagger13_handover** (takeover 4-16 s, seeds 1300000+). Scores still running: control_s0 and
  control_s1 step 3000, rtc step 3000 (pick 27/30, bin 28/30 so far, success counts), baseline delay 12. KI flow
  0.063 at step 3040 (flat). Steps: control 4320 / 3800, rtc 3540, mem 2040, goal 2720, ki 3040; no errors.
- **2026-10-17 15:51** [scheduler] 193.122.189.119 was idle; launched **failmine_dagger2** (Track B step 2: mine the best policy's failures on the weak tasks (handover, two_bins) on 60 non-test seeds (5000+), with per-episode videos, to name the failure types before generating more data)
- **2026-10-17 16:40** **First control scores (30 seeds, delay 0, step 3000): control_s0 24 / 29 / 15 / 14 = 82,
  control_s1 25 / 29 / 21 / 23 = 98.** Same recipe, only the seed differs, and they are 16 apart (handover 15
  vs 21, two_bins 14 vs 23). Baseline at **delay 12: 25 / 26 / 15 / 15 = 81** (delay 6: 82, delay 0: 90).
  **Consequence for the design:** the evaluation itself has an SE of about ±4.5 on a 120-episode total, so a
  16-point gap between twins is mostly run/checkpoint variance, and a single checkpoint of a single run cannot
  detect an add-on effect below ~15 points. Two changes, no run stopped: (1) every run is now also scored at steps
  4000 and 5000 (watchers added), so each arm is judged on the mean of four checkpoints (3000-6000), which also
  shows whether the s0/s1 gap is per-run or per-checkpoint; (2) the seed-1 replicates already queued stay the rule
  (an add-on must beat both controls' means), and the data A/B (x_data_v11) will need the same: four checkpoints,
  ideally two seeds. Steps: control 4880 / 4280, rtc 4020, mem 2380, goal 3120, ki 3460 (flow 0.061, slowly
  falling); no errors. A10: failmine_dagger2 started (the scheduler launched it at 15:50 when dagger12 ended);
  dagger11 two_bins 340/400 (kept 652, success 519).
- **2026-10-17 16:26** [scheduler] 193.122.189.119 was idle; launched **dagger13_handover** (Track B: handover is the weakest task (16/30 strict at baseline) and the most variable; more teacher takeovers in its middle phase (4-16 s, around the pass) on seeds 1300000+. To be retargeted by failmine_dagger2 if it names a different failure)
- **2026-10-17 16:33** (local clock; the 15:45-16:40 entries above carry estimated times) **Failure mining done**
  (pi05_dagger2 step 7000, 60 non-test seeds 5000+, videos): handover 35/60, two_bins 41/60.
  Handover: 20 of 25 failures lift the cube to carry height (~7 cm) and then hover holding it short of or above
  the relay pad until timeout, while the receiving arm waits (videos 001, 002); 5 never lift it. two_bins: 12 of 19
  have cube 1 in the bin and not cube 2 (7 lifted it, 5 never did); 4 never lift cube 1. Both weak tasks share one
  failure mode, **hover while holding**, which is what DAgger takeovers label; dagger13_handover (takeover 4-16 s)
  is running and matches it. dagger11 done (800 episodes, 702 kept, 554 successful); v11 and v12 copied to both
  H100 hosts; **x_data_v11 queued with a seed-1 twin**, scored at 3000/4000/5000/6000 (as are the s1 add-on
  replicates). **Scheduler fix:** a GPU counted as busy at >10 GB, so a finished run's 2+ h of checkpoint scoring
  (~15 GB, light GPU load) blocked the next training. Now busy = a pi_model.train process on that GPU or >25 GB
  (the first version missed training processes because grep saw NUL bytes in /proc/PID/cmdline; fixed with
  grep -a and checked against a live run). Scheduler restarted.
- **2026-10-17 16:34** **rtc step 3000, delay 0: 27 / 28 / 16 / 19 = 90**, equal to the controls' mean at step 3000
  (82 and 98, mean 90) and to the baseline (90): RTC training costs nothing at delay 0, as H19 needs. Its delay 6
  and 12 scores are running (baseline: 82 and 81). Steps: control 5480 / 4900, rtc 4560, mem 2760, goal 3560, ki
  3960 (flow 0.054, still falling). dagger13 handover 20/400 (12 of 16 kept succeed). All alive.
- **2026-10-17 16:48** No new scores (scoring: rtc 3000 at delays 6/12, ki 3000, goal 3000 with and without
  oracle goals, control 4000s; mem just passed 3000). Steps: control 5880 / 5300 (they finish within ~15 / 40 min,
  then the scheduler starts the seed-1 replicates beside their final scoring), rtc 4900, mem 3020, goal 3860, ki
  4260 (flow 0.056). dagger13 handover 100/400, kept 84, teacher success 59 (70%). All alive.
- **2026-10-17 16:55** [scheduler] 192.222.52.191:0 was idle; launched **x_rtc_s1** (H19 replicate (seed 1): with two seeds per arm, an add-on must beat both controls to count)
- **2026-10-17 17:08** New 30-seed scores (delay 0, strict):

  | run | step | pick | bin | handover | two_bins | total |
  | --- | --- | --- | --- | --- | --- | --- |
  | control_s0 | 3000 / 4000 | 24 / 28 | 29 / 28 | 15 / 25 | 14 / 22 | **82 / 103** |
  | control_s1 | 3000 / 4000 | 25 / 29 | 29 / 28 | 21 / 18 | 23 / 26 | **98 / 101** |
  | rtc_s0 | 3000 / 4000 | 27 / 29 | 28 / 27 | 16 / 14 | 19 / 22 | 90 / 92 |
  | goal_s0 (no goal image) | 3000 | 24 | 27 | 17 | 19 | 87 |
  | baseline pi05_dagger2 7000 | — | 29 | 27 | 16 | 18 | 90 |

  The plain continuation keeps improving: both controls are at 101-103 at step 4000 (mean 102 vs baseline 90),
  with two_bins 22-26 (baseline 18). So far no add-on beats the controls at delay 0 (rtc 92 at 4000 is 10 below
  them; goal without goal images 87 at 3000 vs 90). Too early to call (watch rule; one seed per add-on, steps
  5000/6000 and the s1 replicates pending). The scheduler started **x_rtc_s1** on main GPU 0 at 16:55 beside
  control_s0's step-6000 scoring (62 GB used), as the busy-check fix intended. Steps: control_s0 done (6000),
  control_s1 5800, rtc 5340, mem 3280, goal 4220, ki 4620 (flow 0.055). dagger13 200/400 (kept 164, success 122).
- **2026-10-17 17:17** [scheduler] 192.222.52.191:1 was idle; launched **x_goal_s1** (H21 replicate (seed 1))
- **2026-10-17 17:35** **H22 (knowledge insulation) refuted as run: x_ki_s0 step 3000 = 1 / 1 / 0 / 0 = 2/120.**
  Config verified (knowledge_insulation true); the hand reaches the cube (4 mm) but never lifts it, matching its flow
  loss staying 2.4x the control's after the FAST cross-entropy rewrote the VLM in the first ~500 steps. The run
  finishes (never stop early) and its later checkpoints are scored. Its seed-1 replicate is cancelled (it would
  replicate a collapse). Follow-up **H22b: x_ki_vlm01_s0**, KI with the VLM at 0.1x learning rate. A smaller CE
  weight would not help: under KI the VLM only receives CE gradients and Adam normalises their scale, so only the
  VLM's learning rate slows the drift (new `vlm_lr_scale` config field; separate optimizer groups carry an
  `lr_scale` that the schedule multiplies; parameter split checked). **rtc_s0 step 3000 at delay 6: 25 / 28 / 13 /
  16 = 82**, the same as the baseline at delay 6 (82), so at step 3000 RTC training has not bought delay robustness
  (H19 weakening; delay 12 and step 6000 pending). Queue deepened with **H23 x_control_long_s0** (the same control
  for 12000 steps, since the controls are still climbing). The scheduler started x_goal_s1 on main GPU 1 at 17:17.
  Steps: rtc 5780, mem 3560, goal 4540, ki 5000+, rtc_s1 680. dagger13 300/400 (kept 247, success 187).
- **2026-10-17 17:41** [scheduler] 192.222.55.167:0 was idle; launched **x_mem_s1** (H20 replicate (seed 1))
- **2026-10-17 18:09** [scheduler] 193.122.189.119 was idle; launched **failmine_c1_5000** (Track B: failure types of the best checkpoint so far (x_control_s1 step 5000, 104/120 on the test seeds) on 60 non-test seeds (5000+) per weak task: does 'hover while holding' still dominate after the continuation?)
- **2026-10-17 18:12** Scores so far (30 seeds, delay 0, strict totals of 120; mean over the checkpoints scored):

  | run | 3000 | 4000 | 5000 | 6000 | mean |
  | --- | --- | --- | --- | --- | --- |
  | control_s0 | 82 | 103 | 96 | 95 | 94.0 |
  | control_s1 | 98 | 101 | **104** | 100 | **100.8** |
  | rtc_s0 | 90 | 92 | 83 | (scoring) | 88.3 |
  | goal_s0, no goal image | 87 | 96 | | | 91.5 |
  | mem_s0 | 94 | | | | 94.0 |
  | ki_s0 | 2 | | | | collapsed |
  | baseline pi05_dagger2 7000 | 90 | | | | |

  Both controls finished: the plain low-LR continuation lifts pi05_dagger2 from 90 to a 97.4 mean over eight
  checkpoints (best single: control_s1 step 5000 = 29 / 30 / 24 / 21 = 104). Checkpoint-to-checkpoint swings
  within one run reach 21 points (control_s0 82 → 103), so only the means are compared. No add-on is above the
  controls so far; rtc is below them at delay 0 (88 vs 97); goal and mem have only one or two checkpoints. On the
  A10 (idle 17:45-18:09 after dagger13 finished, because its queue was empty; my miss), the scheduler started
  **failmine_c1_5000** (failure types of the new best checkpoint), then **dagger14_from_best** and
  **dagger15_twobins_late_best** (on-policy DAgger from control_s1 step 5000, seeds 1400000+/1500000+) follow.
  dagger13 (328 kept, 254 successful) joined the data A/B batch before it started (v11 + v12 + v13).
  Steps: rtc_s1 1660, goal_s1 900 (main box); mem 4140, goal 5260, ki 5860, mem_s1 340 (4x H100); no errors.
- **2026-10-17 18:18** [scheduler] 192.222.55.167:3 was idle; launched **x_ki_vlm01_s0** (H22b: KI collapsed (2/120) because the fresh FAST cross-entropy (15 nats at the start) rewrote the VLM features faster than the insulated expert could follow (flow loss 0.026 -> 0.098 in 500 steps). A loss weight cannot slow this under Adam, a learning-rate multiplier can: KI with the VLM at 0.1x LR keeps the expert's inputs nearly fixed while the VLM learns FAST)
- **2026-10-17 18:28** All seven GPUs busy, no errors. rtc_s0 and ki_s0 finished at 6000; the scheduler started
  **x_ki_vlm01_s0** (H22b) on 4x H100 GPU 3 at 18:18: step 140, CE 13.05, flow 0.028 (still in warmup, so the
  first test of whether the slow VLM keeps the flow loss down comes over the next ~500 steps; plain KI's flow had
  doubled by step 380). Steps: mem 4380, goal 5580, mem_s1 620, rtc_s1 2120, goal_s1 1300. A10:
  failmine_c1_5000 on handover (episode 23 of 60). Separately, the write-up gained a Results tab with rollout
  videos (scripts/record_showcase.py, filmed on main GPU 1 beside training); an experiment tree is in progress.
- **2026-10-17 18:30** Two negative signals at step 3000 (one seed each; steps 4000-6000 and the s1 replicates
  still to come, so not a verdict yet):
  - **H19 (RTC):** rtc_s0 at delays 0 / 6 / 12 = **90 / 82 / 75**, against the non-RTC baseline's 90 / 82 / 81.
    Training with committed prefixes did not make the policy more robust to lag, and at 12 steps it is 6 worse.
    In this sim the naive switch (old chunk's committed actions, then the new chunk from step d) costs little
    beyond the stale observation, so RTC's smoothing has little to fix; its real case is jerky hardware with
    latency. Step 6000 at delays 0/6/12 decides.
  - **H21 (goal images):** goal_s0 step 3000 **with** oracle goals = 23 / 24 / 15 / 14 = **76**, **without** = 87.
    Showing the model where the scene should be in 2 s made it worse. Candidate reasons: goals were present in only
    25% of training samples (the paper's dropout), so the model rarely learned to use them; and the teacher's
    continuation from the policy's own state differs from the demonstrations' futures. If step 6000 agrees, the
    world-model idea is not worth building in this form. ki_s0 step 4000: 0/120 (H22 collapse confirmed).
  Steps: mem 4400, goal 5580, ki_vlm01 140, mem_s1 620, rtc_s1 2140, goal_s1 1320; no errors.
- **2026-10-17 18:37** [scheduler] 193.122.189.119 was idle; launched **dagger14_from_best** (Track B batch 2 (on-policy DAgger): takeovers from the current best policy (x_control_s1 step 5000) label the states it reaches now rather than those of pi05_dagger2; handover + two_bins, takeover 2-16 s, seeds 1400000+)
- **2026-10-17 18:50** New: **control_s0 step 6000 at delay 6 = 29 / 28 / 21 / 21 = 99** (95 at delay 0). The
  continued control loses nothing to a 6-step lag, which leaves RTC (H19) even less to fix in this sim.
  **goal_s0 step 5000 without goals = 28 / 28 / 24 / 22 = 102** (87, 96, 102 so far; mean 95, the controls'
  mean is 97.4); its with-goal score at 6000 is pending. **H22b looks right so far:** x_ki_vlm01_s0 at step 500
  has flow 0.029 (plain KI 0.099 at the same point) while CE falls more slowly (11.4). **failmine_c1_5000** (best
  checkpoint, 60 non-test seeds): handover 44/60 (pi05_dagger2: 35/60), two_bins 37/60 (41/60). The failure mode
  is unchanged: handover 14 of 16 failures lift and then stall holding the cube; two_bins 18 of 23 have cube 1 in
  the bin and cube 2 not (12 lifted, 6 never). The continuation fixed handover more than two_bins, so the data
  loop's second-arm takeovers (dagger12, dagger15) target the right gap. dagger14 (on-policy from the best
  checkpoint) handover 40/300, success 28 of 36 kept. Steps: mem 4660, goal 5960, ki_vlm01 500, mem_s1 900,
  rtc_s1 2640, goal_s1 1720; no errors.
- **2026-10-17 18:52** [scheduler] 192.222.55.167:2 was idle; launched **x_data_v11** (Track B batch 1: successful teacher takeovers from DAgger rounds 11 (handover + two_bins, takeover 1-15 s) and 12 (two_bins, takeover 8-22 s), both from pi05_dagger2 step 7000 on seeds 1100000+, added to sim_mixdagclean_v1, beat the control. Blocked until both are copied to the host. Unblocked 16:55: v11 (702 episodes, 554 successful) and v12 (245, 173) are on both H100 hosts. 18:05: v13 (handover takeovers 4-16 s; 328 kept, 254 successful) joins this batch before it started: batch 1 = successful takeovers of v11 + v12 + v13 (~980 episodes), all from pi05_dagger2 step 7000 with the fixed teacher)
- **2026-10-17 19:10** New: **rtc_s0 step 6000, delay 0 = 28 / 28 / 21 / 24 = 101** (90, 92, 83, 101: mean
  91.5); **mem_s0 step 4000 = 28 / 27 / 16 / 19 = 90** (94, 90: mean 92). Means so far at delay 0: controls 97.4
  (8 checkpoints), goal without goals 95 (3), mem 92 (2), rtc 91.5 (4). No add-on is ahead of the controls.
  The scheduler started **x_data_v11** (batch v11+v12+v13) on 4x H100 GPU 2 at 18:51. ki_vlm01 step 860: flow
  0.035 (0.028 → 0.029 → 0.035, creeping up slowly; plain KI was at ~0.099 here), CE 9.1. Queue deepened with
  **H24 x_control_r2_s0** (a second low-LR round from the best checkpoint; copied to the 4x H100 host). dagger14
  handover 160/300 (kept 135, success 106). No errors; scheduler and caffeinate alive.
- **2026-10-17 19:30** No new scores. **H22b weakening:** x_ki_vlm01_s0 flow loss 0.028 → 0.029 → 0.035 → **0.070**
  at steps 140 / 500 / 860 / 1220 while CE falls 13.1 → 7.2. The 0.1x VLM learning rate delayed the drift rather
  than preventing it: the flow loss took off once warmup ended (step 1000) and the VLM reached its full 1e-6.
  Third check in a row with the same direction; the run continues to 6000 and its step-3000 score decides. If it
  collapses too, KI is closed for this setup (the FAST objective has to be learned from pretraining, as in π0.5,
  not bolted on during fine-tuning). Queue: rtc_s0, goal_s0, ki_s0 marked done with results. Steps: mem 5240,
  data_v11 820, mem_s1 1460, rtc_s1 3560, goal_s1 2580. dagger14 handover 260/300 (kept 225, success 173). All
  seven GPUs busy; scheduler and caffeinate alive.
- **2026-10-17 19:50** Step-6000 scores (30 seeds, strict):

  | run, step 6000 | delay 0 | delay 6 | delay 12 |
  | --- | --- | --- | --- |
  | control_s0 | 95 | 99 | 82 |
  | rtc_s0 | 101 | 89 | (scoring) |
  | baseline pi05_dagger2 7000 | 90 | 82 | 81 |

  **H19 (RTC) refuted on seed 0:** at delay 6 RTC drops 12 points from its own delay-0 score while the matched
  control gains 4 (89 vs 99); at step 3000 the picture was the same (82 / 75 vs the baseline's 82 / 81). rtc_s1
  (step 4040) is the replicate; unless it reverses this, RTC adds nothing in this simulator.
  **goal_s0 step 6000 without goals = 29 / 29 / 23 / 24 = 105**, the best single checkpoint so far (87, 96, 102,
  105: mean 97.5, the same as the controls' 97.4). So training with goal images in 25% of samples costs nothing,
  while feeding oracle goals at test time hurt at step 3000 (76); its with-goal score at 6000 is next. ki_vlm01
  flow 0.085 at step 1580 (still rising). Steps: mem 5500, data_v11 1320, mem_s1 1760, rtc_s1 4040, goal_s1 3000.
  dagger14 handover done (kept 225+, success 173+), now two_bins 80/300. All alive.
- **2026-10-17 20:10** New: **rtc_s1 step 3000 = 26 / 27 / 15 / 18 = 86**; **mem_s0 step 5000 = 26 / 29 / 18 / 19 =
  92** (mem: 94, 90, 92, mean 92, below the controls' 97.4 on all three); **ki_s0 steps 5000 / 6000 = 2 / 3** (KI
  collapsed at every checkpoint: 2, 0, 2, 3). control_s0 step 6000 at delay 12 = 24 / 27 / 15 / 16 = 82 (the
  control's delay curve: 95 / 99 / 82). ki_vlm01 flow 0.079 at step 2020 (0.085 at 1580: levelling off, still ~3x
  the controls), CE 6.2. Steps: mem 5780, data_v11 1840, mem_s1 2040, rtc_s1 4540, goal_s1 3400. dagger14 two_bins
  200/300 (total kept 425, success 330). All seven GPUs busy; scheduler and caffeinate alive.
- **2026-10-17 20:24** [scheduler] 192.222.55.167:1 was idle; launched **x_data_v11_s1** (Seed-1 twin of x_data_v11 (the control twins differ by 16 points at one checkpoint, so a data version needs two seeds))
- **2026-10-17 20:28** [scheduler] 193.122.189.119 was idle; launched **dagger15_twobins_late_best** (Track B batch 2: second-arm hover in two_bins from the best policy, late takeovers 8-22 s (as v12, which kept 61% of episodes), seeds 1500000+)
- **2026-10-17 20:30** mem_s0 finished (scored 94 / 90 / 92 at 3000-5000; 6000 scoring); the scheduler started
  **x_data_v11_s1** on 4x H100 GPU 1 at 20:24. dagger14 done (600 episodes, 507 kept, 398 successful); the
  scheduler started **dagger15_twobins_late_best** at 20:28. Queues refilled: **H21b x_goal75_s0** (goal images in
  75% of training samples instead of the paper's 25%, scored with and without oracle goals; new `subgoal_prob`
  config field, default 0.25 so earlier runs are unchanged), **dagger16_twobins_mid_best** (A10: two_bins
  takeovers at 3-14 s from the best checkpoint), and **x_data_b2** (blocked: on-policy batch v14 + v15 on top of
  batch 1, resumed from the best checkpoint, against x_control_r2_s0). ki_vlm01 flow 0.080 at 2520 (flat), CE 6.1;
  its step-3000 score comes next. Steps: data_v11 2340, mem_s1 2320, rtc_s1 5020, goal_s1 3780. All busy.
- **2026-10-17 20:50** rtc_s0 step 6000 at delay 12 = 23 / 23 / 16 / 14 = **76**, completing the step-6000 delay
  curves: **rtc 101 / 89 / 76 vs control 95 / 99 / 82** at delays 0 / 6 / 12. RTC is better at no delay by 6 (inside
  the run-to-run spread) and worse under every delay, the opposite of H19. goal_s1 step 3000 = 26 / 27 / 18 / 18 =
  89 (goal_s0 had 87 there). ki_vlm01 flow 0.068 at 2980 (drifting down slowly), CE 6.0, FAST accuracy 5%; its
  step-3000 score is next. Steps: data_v11 2840, data_v11_s1 500, mem_s1 2600, rtc_s1 5520, goal_s1 4140.
  dagger15 two_bins 80/300 (kept 44, success 31; 36 skipped: the policy finished or failed before the late
  takeover). All seven GPUs busy; scheduler and caffeinate alive.
- **2026-10-17 21:10** **H21 (goal images, paper's 25% share) refuted on seed 0:** goal_s0 step 6000 = **105 without
  goals, 90 with oracle goals** (28 / 27 / 12 / 23; handover collapses from 23 to 12). Same direction as step 3000
  (87 vs 76). Training with goal images costs nothing (its without-goal mean 97.5 equals the controls'), but giving
  the model a goal image at test time hurts, so a world model built to supply them would make this policy worse.
  H21b (x_goal75_s0, goals in 75% of samples) is queued as the last test of this idea. rtc_s1 step 3000 at delays
  0 / 6 = 86 / 84. rtc_s1 finished; the scheduler started **x_control_long_s0** (H23, 12000 steps) on main GPU 0
  at 21:09, a minute after the GPU dropped to scoring only. ki_vlm01 flow 0.070 at 3400 (step-3000 score running).
  Steps: data_v11 3360, data_v11_s1 960, mem_s1 2920, goal_s1 4540. dagger15 180/300 (kept 98, success 70).
- **2026-10-17 21:09** [scheduler] 192.222.52.191:0 was idle; launched **x_control_long_s0** (H23: the plain continuation is still climbing (controls 82/98 at step 3000, 103/101 at 4000 vs baseline 90), so the 6000-step budget may be too short: the same recipe over 12000 steps (LR decays over twice as long) ends higher. Also the right budget for the phase-2 combination run)
- **2026-10-17 21:30** **mem_s0 step 6000 = 28 / 30 / 20 / 22 = 100** (94, 90, 92, 100: mean 94.0, controls 97.4;
  two_bins 19-22 vs the controls' 19-26, so no sign yet that history helps the second arm). Seed-0 tallies at delay
  0, mean over steps 3000-6000: controls 97.4 (two seeds), goal without goals 97.5, mem 94.0, rtc 91.5, KI 1.8.
  Only the s1 replicates (mem_s1 3220, goal_s1 4920; rtc_s1 done, scoring) can still move this. x_control_long_s0
  at 420, ki_vlm01 3820 (flow 0.072; step-3000 score still running), data_v11 3860, data_v11_s1 1500. dagger15
  260/300 (kept 145, success 102); when it ends v14 + v15 go to both hosts and x_data_b2 is unblocked.
- **2026-10-17 21:35** [scheduler] 193.122.189.119 was idle; launched **dagger16_twobins_mid_best** (Track B: two_bins is now the weakest task for the best checkpoint (37/60 non-test; 18 of 23 failures leave cube 2 undelivered); takeovers at 3-14 s cover the second arm's approach and grasp rather than only its late hover (v15 covers 8-22 s). Seeds 1600000+)
- **2026-10-17 21:50** **x_data_v11 (batch 1: v11 + v12 + v13) step 3000 = 28 / 29 / 20 / 17 = 94** (controls at
  3000: 82 and 98). dagger15 done (300 episodes, 165 kept, 116 successful); dagger16 started 21:35 (two_bins
  60/300, success 40 of 49 kept). v14 (507 kept, 398 successful) and v15 (165, 116) copied to both H100 hosts;
  **x_data_b2 unblocked** as a clean A/B: sim_mixdagclean_v1 + successful v14/v15 only, resumed from the best
  checkpoint, against x_control_r2_s0 (same resume, no new data). A10 queue: **dagger17_handover_best** (handover
  stall window 4-16 s, seeds 1700000+). mem_s0 and rtc_s1 marked done. ki_vlm01 flow 0.068 at 4240 (its step-3000
  score has not landed yet: scoring shares the GPU with training). Steps: control_long 920, data_v11 4340,
  data_v11_s1 2060, mem_s1 3480, goal_s1 5320. All seven GPUs busy; scheduler and caffeinate alive.
- **2026-10-17 22:10** **H22b refuted: x_ki_vlm01_s0 step 3000 = 0 / 0 / 0 / (two_bins scoring) out of 30**, the
  hand mostly never closes on the cube (lift rates 0-7%). Slowing the VLM 10x kept the flow loss at 0.07 (plain
  KI 0.10) but the policy still collapsed. **Knowledge insulation is closed for this setup:** bolting a fresh FAST
  objective onto a VLM that was fine-tuned through the flow loss rewrites the features the frozen-gradient expert
  depends on, whatever the VLM's learning rate; π0.5 learns FAST in pretraining instead. The run finishes to 6000
  (~40 min) and frees GPU 3. New: **rtc_s1 step 3000 at delays 0 / 6 / 12 = 86 / 84 / 78** (rtc_s0: 90 / 82 / 75;
  the baseline: 90 / 82 / 81); **mem_s1 step 3000 = 27 / 28 / 23 / 20 = 98** (mem_s0 had 94). Steps:
  control_long 1420, goal_s1 5720, data_v11 4840, data_v11_s1 2620, mem_s1 3760, ki_vlm01 4660. dagger16 two_bins
  180/300 (success 119 of 145 kept). Queue: control_r2, goal75, data_b2 for the next three free GPUs.
- **2026-10-17 22:25** [scheduler] 192.222.52.191:1 was idle; launched **x_control_r2_s0** (H24: low-LR continuation compounds. The first round lifted pi05_dagger2 from 90 to a 97.4 mean (best 104); a second identical round started from that best checkpoint (x_control_s1 step 5000) ends higher again, or shows the plateau. Scored at 3000-6000)
- **2026-10-17 22:30** [scheduler] 193.122.189.119 was idle; launched **dagger17_handover_best** (Track B: handover still fails by stalling while holding the cube (14 of 16 failures of the best checkpoint on non-test seeds); more on-policy takeovers in the stall window (4-16 s) from the best checkpoint, seeds 1700000+)
- **2026-10-17 22:32** **goal_s1 step 3000 with oracle goals = 26 / 22 / 13 / 14 = 75, without = 89**: the second
  seed repeats goal_s0 (76 vs 87), so H21 is refuted on both seeds at the paper's 25% goal share (H21b, 75% share,
  queued). **ki_vlm01 step 3000 = 0/120** (H22b closed). dagger16 done (300 episodes, 236 kept, 193 successful);
  the scheduler started dagger17_handover_best at 22:30 and x_control_r2_s0 (H24) on main GPU 1 at 22:24. A10
  queue: **H25 robust_c1_5000** (the best checkpoint under visual randomization on the 30 test seeds per task:
  how brittle is it before hardware?). Steps: control_long 1920, control_r2 40, data_v11 5340, data_v11_s1 3160,
  mem_s1 4060, ki_vlm01 5080. All seven GPUs busy; scheduler and caffeinate alive.
- **2026-10-17 22:50** **x_data_v11 step 4000 = 27 / 28 / 18 / 19 = 92** (94, 92 so far; the controls had 103 / 101
  at 4000), so batch 1 (DAgger from pi05_dagger2) is not ahead yet; its seed-1 twin (3660) and steps 5000/6000
  decide. **rtc_s1 step 4000 = 24 / 27 / 16 / 22 = 89** (86, 89). Steps: control_long 2420, control_r2 520,
  data_v11 5840, data_v11_s1 3660, mem_s1 4340, ki_vlm01 5480. dagger17 handover 80/300 (success 53 of 66 kept).
  All seven GPUs busy; scheduler and caffeinate alive.
- **2026-10-17 22:56** [scheduler] 192.222.55.167:2 was idle; launched **x_goal75_s0** (H21b: oracle goal images hurt goal_s0 (76 with vs 87 without at step 3000) because goals were present in only 25% of training samples, so the model never learned to rely on them. With goals in 75% of samples the with-goal score beats the without-goal score and the controls. If it does not, goal conditioning is closed here)
- **2026-10-17 23:10** No new scores. x_data_v11 finished; the scheduler started **x_goal75_s0** (H21b) on 4x H100
  GPU 2 at 22:56. Queue deepened for the next frees (ki_vlm01 ends in minutes and takes x_data_b2): **H18
  x_twobins2x_r2** (two_bins sampled twice as often in a second low-LR round from the best checkpoint; never run
  before) and **x_control_r2_s1** (the second-round reference needs two seeds, given the 16-point twin spread).
  Steps: control_long 2900, control_r2 1000, data_v11_s1 4160, mem_s1 4600, ki_vlm01 5900, goal75 180. dagger17
  handover 180/300 (success 116 of 152 kept). All seven GPUs busy; scheduler and caffeinate alive.
- **2026-10-17 23:15** [scheduler] 192.222.55.167:3 was idle; launched **x_data_b2** (Track B batch 2 (on-policy): successful takeovers from DAgger rounds 14 (handover + two_bins, takeover 2-16 s; 398 successful) and 15 (two_bins, 8-22 s; 116 successful), both rolled out from the best checkpoint x_control_s1 step 5000, added to sim_mixdagclean_v1 and resumed from that checkpoint. Clean A/B against x_control_r2_s0 (same resume and recipe, no new data))
- **2026-10-17 23:30** New: **x_data_v11 step 5000 = 28 / 28 / 15 / 20 = 91** (94, 92, 91), **x_data_v11_s1 step
  3000 = 91**: batch 1 (DAgger rounds 11-13 from pi05_dagger2) trails the controls at every matched step (controls
  at 3000-5000: 82 / 103 / 96 and 98 / 101 / 104, mean 97.3; batch 1 mean 92.3). Leaning negative, as earlier
  failure-heavy data did (recover2: 72); handover in particular does not improve (20, 18, 15) although batch 1 was
  aimed at it. **goal_s1 step 4000 = 98** (89, 98); **mem_s1 step 4000 = 24 / 27 / 21 / 18 = 90** (98, 90).
  ki_vlm01 finished; the scheduler started **x_data_b2** (on-policy batch 2) on 4x H100 GPU 3 at 23:14.
  Steps: control_long 3400, control_r2 1520, data_v11_s1 4660, mem_s1 4880, goal75 560, data_b2 280. dagger17
  280/300 (success 179 of 237 kept). All seven GPUs busy; scheduler and caffeinate alive.
- **2026-10-17 23:31** [scheduler] 193.122.189.119 was idle; launched **robust_c1_5000** (H25: the best policy (x_control_s1 step 5000, 104/120) is brittle to visual change: with randomized lighting, colours and textures (eval_sim --visual-randomization) on the same 30 seeds per task it drops far below 104. A large drop means visually randomized data must come before hardware; a small one means the sim-to-real gap is mostly elsewhere)
