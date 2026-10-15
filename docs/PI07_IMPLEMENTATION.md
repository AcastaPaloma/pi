# π₀.₇ implementation notes

This repository reimplements the π₀.₇ model from *π0.7: a Steerable Generalist Robotic
Foundation Model with Emergent Capabilities* (arXiv 2604.15483v2) in PyTorch under
`pi_model/`, with the paper's dimensions and a short list of explicit assumptions
where the paper is silent. The target robot is two SO-101 followers with three
Arducam IMX708 USB cameras, simulated in MuJoCo by `pi_sim/`.

## Dimensions

| Block | Paper | Here (`Pi07Config.full()`) |
| --- | --- | --- |
| VLM backbone | Gemma 3 4B | Gemma 3 4B decoder: 34 layers, hidden 2560, 8 query heads, 4 KV heads, head dim 256, MLP 10240 GeGLU, vocab 262208, sliding window 1024 with every 6th layer global, local RoPE base 10⁴, global base 10⁶ with linear factor 8, QK-norm, pre and post RMSNorm, embeddings scaled by √2560, tied output head |
| Vision encoder | 400M, initialised from Gemma 3, MEM-style | SigLIP So400m/14: 27 layers, hidden 1152, 16 heads, MLP 4304, patch 14; images 448×448 → 32×32 patches; Gemma 3 projector pools 2×2 to **256 tokens** per image, soft-token RMSNorm, linear to 2560 |
| History | up to 6 frames at 1 s stride, compressed to one frame's tokens | MEM space-time attention in every 4th SigLIP layer (layers 4, 8, …, 24); past frames dropped after layer 24 |
| Action expert | 860M, flow matching, adaptive RMSNorm, 50 action tokens | 34 layers (one per backbone layer), hidden 1216, MLP 4608, same head layout as the backbone so it attends to the backbone's keys/values; adaptive RMSNorm on both pre-norms from a 256-d timestep embedding |
| Totals | 4B VLM (incl. 0.4B vision) + 0.86B expert ≈ 5B | 3.88B + 0.42B + 0.87B = **5.17B** (`tests/test_model.py` checks these) |
| Actions | 50-step chunk | 50 × 32 (12 used: two arms × 5 joints + gripper, degrees and 0–100) |
| Control | 50 Hz | 50 Hz in `pi_sim` (`RobotEnv(control_hz=50)`) |
| Cameras | front, two wrists, optional rear | overhead, left wrist, right wrist (overhead takes the base-view role) |
| Subgoals | up to 3 views, same encoder | 3 views, same encoder, K = 1 |

`Pi07Config.small()` (≈0.3B, vocab 4096) and `Pi07Config.tiny()` keep the exact
wiring for experiments without Gemma weights and for CPU tests.

## Prompt and sequence

```
<obs tokens: V cameras × 256> <subgoal tokens: G views × 256>
Task: Place the red cube on the green disk. Subtask: Lift the object off the table.
Speed: 500. Quality: 5. Mistake: false. Control Mode: joint. Action:
<state tokens: K frames> [<FAST tokens> <eos>]     ← training only
                                                     ↓ action expert: 50 flow tokens
```

Attention (paper Fig. 19): observation tokens are bidirectional among themselves;
subgoal tokens are bidirectional and attend observations; text, state and FAST
tokens are causal; the 50 flow tokens attend the whole prefix except the FAST
tokens and are bidirectional among themselves. FAST tokens never see flow tokens.
Local layers add a 1024-token window to all of this. See `attention_mask` in
`pi_model/model.py` and `tests/test_model.py::test_attention_rules`.

Proprioception follows MEM: each history state is a linear projection into the
backbone width, one token per frame, masked when the frame is dropped. The current
state is always the last state token, so it is the position that predicts the
first FAST token.

Dropout (Sec. V-E, VI-B): subgoal images on 25% of examples; subtask dropped 30%
of the time when subgoals are present; metadata dropped entirely 15% and each
field 5%; history dropped 30%; control mode never dropped. Subgoal frames are the
end of the current segment with probability 0.25, otherwise 0–4 s ahead (Sec. VI-C).

Episode metadata: speed = episode length rounded to 500 steps; quality 1–5; mistake
per segment. At run time (Sec. VII): speed = 15th percentile of the task's training
episode lengths, quality 5, mistake false, 5 denoising steps, execute 15 or 25 of 50,
optional classifier-free guidance on the metadata with v = v_c + β (v_c − v_u).

## Training objectives

- **Knowledge insulation.** The VLM is trained only by cross-entropy on the FAST
  tokens (and, for the high-level policy, subtask text). The expert attends to the
  backbone's per-layer keys and values through a stop-gradient. The test
  `test_losses_and_knowledge_insulation` asserts that the flow loss produces zero
  gradient on every VLM parameter.
- **Flow matching.** x_τ = τ·a + (1−τ)·ε with τ = 1 clean; target velocity a − ε;
  τ sampled with π0's Beta(1.5, 1) emphasis on noisy steps; Euler sampling.
- **Training-time RTC** (Black et al. 2025): a delay d ∈ {0…12} is sampled per
  example; the first d action tokens are the clean ground truth with τ = 1 and no
  loss; at inference the already-committed actions fill those slots.
- **FAST** (Pertsch et al. 2025): per-dimension quantile scaling to [−1, 1], DCT-II
  along time, ×10 and rounding, frequency-major flattening, BPE with a 1024-symbol
  vocabulary fitted per dataset; FAST id k maps to text id 262208 − 1 − 128 − k.

## High-level policy and world model

`--objective subtask` trains the same architecture as the high-level policy:
observation + task + previous subtasks → next subtask text, decoded greedily by
`Pi07Policy.next_subtask`. The paper's BAGEL-based world model for generated
subgoals is **not** reimplemented; training uses real future frames, and the
policy runs with or without subgoals. Supplying generated subgoals later only
requires filling the `subgoals` input.

## Assumptions

1. Space-time attention: a query at (frame k, patch p) attends every patch of frame
   k plus patch p in valid earlier frames; one softmax over both sets; the
   layer's own projections; a sinusoidal e(t) with e(0) = 0 added before the
   temporal keys/values. Periods 0.2–100 s. Past frames are dropped right after
   the last temporal layer.
2. Expert width 1216 and MLP 4608 reach 0.87B; the paper gives only the total.
3. The 2×2 pooling at 448 px matches Gemma 3's 256 tokens per image.
4. Images are resized to 448×448 without padding, as the paper states.
5. The expert obeys the same sliding window as the backbone on local layers.
6. The prompt ends with `Action:` before the state tokens; the FAST stream ends
   with Gemma's EOS.
7. The control rate is 50 Hz on the SO-101 as on the paper's robots; the
   simulator's per-second speeds and dwell times are unchanged from the earlier
   20 Hz playground.
8. Scripted-demo quality labels: 5 clean, 4 slow, 3 sloppy, 2 slow and sloppy,
   minus one per retried grasp, 1 for failures; a retried grasp marks the
   segment as a mistake.

## Running it

```bash
# record scripted episodes with subtask segments, mistakes and visual randomization
python -m pi_sim.record --out data/sim_v1 --episodes 800 --image-size 448
python -m pi_model.data prepare --root data/sim_v1 --out data/sim_v1/pi07_meta

# paper size, initialised from Gemma 3 4B (gated; accept the license, then hf download)
torchrun --nproc_per_node 2 -m pi_model.train --data data/sim_v1 --config full \
    --gemma weights/gemma-3-4b-pt --tokenizer weights/gemma-3-4b-pt/tokenizer.model \
    --out runs/full_v1 --batch-size 4 --grad-accum 4 --bf16 --gradient-checkpointing

# closed loop in MuJoCo with the paper's runtime prompting; the paper always supplies a
# subtask at run time (high-level policy or human), so evaluate with one: `oracle` derives
# the scripted phase from simulator state, or train the high-level policy (--objective subtask)
python -m pi_model.eval_sim --checkpoint runs/full_v1/step_0010000 --episodes 40 --execute 25 --delay 6 --subtask oracle
```

Training prompts contain a subtask about 92% of the time (it is dropped only in 30% of the
25% of examples that carry subgoal images), so running without one is out of distribution.

`scripts/box/` holds the launch scripts used on the two-H100 machine. Tests:
`python -m unittest tests.test_model tests.test_pipeline`.
