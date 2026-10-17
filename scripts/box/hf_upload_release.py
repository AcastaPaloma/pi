"""Pack the simulation datasets and the full-precision final checkpoints and upload them to private Hugging Face repos.

    ~/lerobot-venv/bin/python scripts/box/hf_upload_release.py --user <hf-user>   # run on the training box, from ~/pi

Datasets are millions of small JPEG frames, which the Hub handles badly, so each dataset folder becomes ~10 GB tar
parts (`cat NAME.tar.part-* | tar xf -` restores it). Mixes are symlink folders; they are uploaded as text manifests.
The upload uses the token in ~/.cache/huggingface/token and `upload_large_folder` (resumable: re-run to continue).
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi

DATASETS = ["sim_v1", "sim_ladder_v1", "sim_recover_v1", "sim_recover2_v1", "sim_twobins_mixed_v1"] + \
           [f"sim_dagger_v{v}" for v in (1, 2, 3, 4, 5, 6, 7, 9, 10)]
WHAT = {"sim_v1": "single-arm pick_place, scripted (first dataset)",
        "sim_ladder_v1": "four-task ladder, clean scripted teacher",
        "sim_recover_v1": "ladder with target knocks (recover style)",
        "sim_recover2_v1": "knocks, object bumps and deliberate missed grasps, mistake-labelled (recover2)",
        "sim_twobins_mixed_v1": "two_bins with either arm first (mixed_order and mixed_order_recover2)",
        "sim_dagger_v1": "DAgger from pi05_v1 step 8000 (broken takeover)",
        "sim_dagger_v2": "DAgger from pi05_2arm_v2 (broken takeover)",
        "sim_dagger_v3": "DAgger from pi05_warm2arm step 4000 (broken takeover)",
        "sim_dagger_v4": "DAgger from pi05_v1 step 15000 (broken takeover)",
        "sim_dagger_v5": "DAgger, two_bins only, from pi05_v1 step 17000 (broken takeover)",
        "sim_dagger_v6": "DAgger from pi05_2arm_v2 step 12000 (broken takeover)",
        "sim_dagger_v7": "DAgger from pi05_v1 step 20000, fixed takeover",
        "sim_dagger_v9": "DAgger from pi05_v1 step 20000, fixed takeover, new seeds",
        "sim_dagger_v10": "DAgger from pi05_dagger1 step 7000, fixed takeover"}
MIXES = ["sim_mix_v1", "sim_mixrec2_v1", "sim_mixdag_v1", "sim_mixdagclean_v1"]
MODELS = {"pi05_dagger2_step7000": "runs/pi05_dagger2/step_0007000",
          "pi05_v1_step20000": "runs/pi05_v1/step_0020000",
          "pi05_single_step6000": "runs/pi05_single/step_0006000"}

DATA_CARD = """---
license: apache-2.0
tags: [robotics, simulation, mujoco, so-101, bimanual, imitation-learning]
---
# SO-101 bimanual MuJoCo demonstrations (π0.7 recipe experiments)

Scripted-teacher and DAgger demonstrations for two SO-101 arms in MuJoCo, used to train the π0.7-recipe policies in
`{user}/so101-pi07-sim-models`. Three cameras (overhead, left wrist, right wrist), 50 Hz joint control.

**Tasks**: `pick_place_any` (one arm, cube to a disk), `bin`, `handover` (pass over a pad, then bin), `two_bins`
(one cube per arm into a shared bin), plus the older single-arm `pick_place` (`sim_v1`).

| folder | episodes | what |
| --- | --- | --- |
{rows}

**Restore** a folder: `cat NAME/NAME.tar.part-* | tar xf -` → `NAME/episode_*/{{meta.json, steps.npz, frames/<camera>/<step>.jpg}}`.
`meta.json` holds the instruction, task, style, success, quality (1–5) and the subtask names. `steps.npz` holds
per-step state, action (commanded joint targets, degrees and gripper 0–100), subtask index, mistake flags, and
object, goal and end-effector poses. Frames are 448 px JPEGs every `frame_stride` (5) steps. Failed episodes are
kept on purpose (π0.7 conditions on episode quality); `PI_TRIM_STILL_TAIL_S=1` in the training code trims their
parked tails. Training mixes are symlink folders: `mixes/*.txt` lists their members, and `sim_mix_v1_pi07_meta/` holds
the normalisation statistics and FAST tokenizer used by every model. The DAgger sets v1–v6 came from a teacher
takeover that was broken (fixed before v7; see the research log), so filter on `success` for those.
"""

MODEL_CARD = """---
license: apache-2.0
tags: [robotics, vla, pi0.5, pi0.7, so-101, mujoco]
---
# SO-101 bimanual policies (π0.7 recipe on the π0.5 backbone), full precision

Checkpoints as saved by training (`model.pt` state dict plus `config.json`, `stats.json`, `extra.json`), for
the `pi_model` code in the project repo (`docs/INFERENCE.md`). The backbone is LeRobot's `PI05Pytorch`
(`lerobot/pi05_base`); the PaliGemma tokenizer is not included (get it from `google/paligemma-3b-pt-224`).

| folder | strict success on 30 fixed seeds per task (pick / bin / handover / two_bins) |
| --- | --- |
| `pi05_dagger2_step7000` | 28 / 28 / 12 / 18 = 86/120 (best two-arm, best two_bins) |
| `pi05_v1_step20000` | 26 / 26 / 19 / 12 = 83/120 (best handover) |
| `pi05_single_step6000` | single-arm pick_place 28/30 |

Evaluated closed loop in the matching MuJoCo scene with oracle subtask prompts, 25 actions per chunk, no delay.
"""


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--stage", default=os.path.expanduser("~/hf_release"))
    args = ap.parse_args()
    stage = Path(args.stage)
    data_dir, model_dir = stage / "data", stage / "models"
    (data_dir / "mixes").mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in DATASETS:
        src = Path("data") / name
        n = sum(1 for p in src.iterdir() if p.name.startswith("episode_"))
        ok = 0
        for p in src.iterdir():
            meta = p / "meta.json"
            if p.name.startswith("episode_") and meta.exists():
                ok += bool(json.loads(meta.read_text()).get("success"))
        rows.append(f"| `{name}` | {n} ({ok} successful) | {WHAT.get(name, '')} |")
        out = data_dir / name
        if not (out / ".done").exists():
            out.mkdir(exist_ok=True)
            sh(f"tar cf - -C data {name} | split -b 10G -d -a 3 - {out}/{name}.tar.part-")
            (out / ".done").write_text("ok")
    for mix in MIXES:
        links = sorted(f"{p.name} -> {os.readlink(p)}" for p in (Path("data") / mix).iterdir() if p.is_symlink())
        (data_dir / "mixes" / f"{mix}.txt").write_text("\n".join(links) + "\n")
    if not (data_dir / "sim_mix_v1_pi07_meta").exists():
        shutil.copytree("data/sim_mix_v1/pi07_meta", data_dir / "sim_mix_v1_pi07_meta")
    (data_dir / "README.md").write_text(DATA_CARD.format(user=args.user, rows="\n".join(rows)))
    for name, src in MODELS.items():
        dst = model_dir / name
        dst.mkdir(exist_ok=True)
        for f in ("model.pt", "config.json", "stats.json", "extra.json"):
            if (Path(src) / f).exists() and not (dst / f).exists():
                os.link(Path(src) / f, dst / f)  # hard link: no extra disk
    (model_dir / "README.md").write_text(MODEL_CARD)

    api = HfApi()
    for repo, kind, folder in ((f"{args.user}/so101-pi07-sim-models", "model", model_dir),
                               (f"{args.user}/so101-pi07-sim-data", "dataset", data_dir)):
        api.create_repo(repo, repo_type=kind, private=True, exist_ok=True)
        print(f"uploading {folder} -> {repo} ({kind}, private)", flush=True)
        api.upload_large_folder(repo_id=repo, repo_type=kind, folder_path=str(folder),
                                ignore_patterns=["**/.done", ".done"])
        print(f"done {repo}", flush=True)


if __name__ == "__main__":
    main()
