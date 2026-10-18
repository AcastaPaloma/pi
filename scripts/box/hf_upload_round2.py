"""Upload the 2026-10-17/18 round (π0.7 add-ons on π0.5) to the project's Hugging Face repos.

    ~/lerobot-venv/bin/python scripts/box/hf_upload_round2.py data sim_dagger_v11 sim_dagger_v12 ...
    ~/lerobot-venv/bin/python scripts/box/hf_upload_round2.py models x_data_b2:6000 x_control_s1:5000 ...

Run from ~/pi on the box that holds the files. Data folders become ~10 GB tar parts under round2/<name>/ in
KuanKuanKuan/so101-pi07-sim-data (restore: cat <name>.tar.part-* | tar xf -). Checkpoints (model.pt, config.json,
stats.json, extra.json, run.json) go under round2/<run>_step<step>/ in KuanKuanKuan/so101-pi07-sim-models, in the
order given (best first, so the most important land before any storage limit). upload_large_folder is resumable:
re-run the same command to continue. Uses the token in ~/.cache/huggingface/token.
"""
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

USER = "KuanKuanKuan"
STAGE = Path.home() / "hf_round2"


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


def stage_data(names):
    root = STAGE / "data"
    for name in names:
        out = root / "round2" / name
        if (out / ".done").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        sh(f"tar cf - -C data {name} | split -b 10G -d -a 3 - {out}/{name}.tar.part-")
        (out / ".done").write_text("ok")
    return root, f"{USER}/so101-pi07-sim-data", "dataset"


def stage_models(specs):
    root = STAGE / "models"
    for spec in specs:
        run, step = spec.split(":")
        src = Path("runs") / run / f"step_{int(step):07d}"
        dst = root / "round2" / f"{run}_step{step}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("model.pt", "config.json", "stats.json", "extra.json"):
            if (src / f).exists() and not (dst / f).exists():
                os.link(src / f, dst / f)  # hard link: no extra disk
        run_json = Path("runs") / run / "run.json"
        if run_json.exists() and not (dst / "run.json").exists():
            (dst / "run.json").write_text(run_json.read_text())
    return root, f"{USER}/so101-pi07-sim-models", "model"


def main():
    kind, items = sys.argv[1], sys.argv[2:]
    root, repo, repo_type = stage_data(items) if kind == "data" else stage_models(items)
    api = HfApi()
    print(f"uploading {root} -> {repo}", flush=True)
    api.upload_large_folder(repo_id=repo, repo_type=repo_type, folder_path=str(root),
                            ignore_patterns=["**/.done", ".done"])
    print(f"done {repo}", flush=True)


if __name__ == "__main__":
    main()
