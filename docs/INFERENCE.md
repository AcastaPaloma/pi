# Running the trained SO-101 policies (no GPU box needed)

Training is finished. The final models are bf16 copies (about 8.3 GB each) on the Mac in `models/`. That folder is
gitignored, so back it up yourself (or push it to a private Hugging Face repo, see the end).

| Folder | What it is | 30-episode strict success (pick / bin / handover / two_bins) |
| --- | --- | --- |
| `models/pi05_dagger2_step7000` | **Best two-arm policy.** π0.5 backbone + π0.7 recipe, DAgger round 2 | 28 / 28 / 12 / 18 of 30 = 86/120 |
| `models/pi05_v1_step20000` | Two-arm baseline, best at handover (the live-demo model) | 26 / 26 / 19 / 12 of 30 = 83/120 |
| `models/pi05_single_step6000` | Single-arm pick-and-place only | pick_place 28/30 |
| `models/tokenizer/tokenizer.model` | PaliGemma tokenizer (also copied to `weights/paligemma/`, the path the checkpoints expect) | |

"Strict" means success with the object actually lifted (pushes do not count). The scores come from the same 30
seeds per task; the raw per-episode results and the review videos are in `outputs/final_evals/`, and the full
history is in `docs/RESULTS.md` and `docs/RESEARCH_LOG.md`.

## One-time setup (Apple Silicon Mac)

```bash
cd ~/Me/code/_learn/pi
uv venv -p 3.12 .venv-infer
VIRTUAL_ENV=.venv-infer uv pip install "lerobot[pi]==0.6.1" mujoco tokenizers sentencepiece av pyarrow scipy einops
VIRTUAL_ENV=.venv-infer uv pip install --no-deps -e .
```

The first model load downloads the small `lerobot/pi05_base` config from Hugging Face (not the weights). The
model needs about 9 GB of memory in bf16, so run one at a time on a 24 GB Mac.

## Evaluate a model on a task (with videos)

```bash
.venv-infer/bin/python -m pi_model.eval_sim --checkpoint models/pi05_dagger2_step7000 \
    --task two_bins --episodes 5 --episode-seconds 30 --execute 25 --delay 0 \
    --device mps --dtype bf16 --subtask oracle --seed 4000 \
    --out outputs/mac_two_bins --video --full-video 2
```

Tasks: `pick_place_any`, `bin`, `handover`, `two_bins` (two-arm models), or `pick_place` (single-arm model).
`--seed 4000 --episodes 30` reproduces the confirmation runs. Videos land in `--out` as
`episode_XXX_success.mp4` or `episode_XXX_fail.mp4`, with the three cameras and the prompted subtask in the caption.

## Interactive demo in the browser

```bash
.venv-infer/bin/python -m pi_model.live_demo --checkpoint models/pi05_dagger2_step7000 --device mps --port 8765
# open http://localhost:8765
```

Drag the cube(s) and the destination on the table map, press Run, and watch an orbitable 3D view plus the
policy's three cameras. Placements far outside the shaded training regions are a test of generalisation.

## Copies on Hugging Face

- `KuanKuanKuan/so101-pi07-sim-models` (private): the same three models in **full precision** (as saved by training).
- `KuanKuanKuan/so101-pi07-sim-data` (**public**, complete, 140.7 GB): every simulation dataset as ~10 GB tar parts (22 parts; restore with
  `cat NAME/NAME.tar.part-* | tar xf -`), the training-mix manifests and the normalisation statistics.
  The models repo is private; make it public on the Hub if you want to share it.

```bash
.venv-infer/bin/hf download KuanKuanKuan/so101-pi07-sim-models pi05_dagger2_step7000/model.pt --local-dir hf_models
```

## Note on Apple Silicon

Inference on MPS was silently wrong until `pi_model/data.py:to_device` stopped using non-blocking copies off
CUDA. A temporary CPU tensor could be freed before the async copy finished, which truncated the prompt mask to
11 of 175 tokens. The arms then went to the wrong places. Now MPS matches CUDA (checked action by action
on the same scene), and bin succeeds 2/2 and two_bins 1/2 on the Mac.
