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

## Wake-up entries

- **2026-10-15 ~22:00** pi05_single step 1000: 3/10 (genuine grasps, one in 194 steps). pi05_v1
  step 2000: 0/8 on pick_place_any (61 mm) and bin (76 mm). Opened H1–H3. Started a per-checkpoint
  evaluator for pi05_single on GPU1.
