#!/usr/bin/env bash
# 30-seed confirmation of one checkpoint on the four ladder tasks (seeds 4000+, oracle subtasks, execute 25).
#   GPU=1 DELAY=0 bash scripts/box/confirm30.sh runs/x_control_s0/step_0006000 control_s0_6000
# Software rendering (OSMesa) works on every box. Results: outputs/c30_<name>_d<delay>_<task>/results.json.
set -euo pipefail
CKPT=${1:?checkpoint dir}; NAME=${2:?name}; DELAY=${DELAY:-0}; GPU=${GPU:-0}
PY=${PY:-$HOME/lerobot-venv/bin/python}
export MUJOCO_GL=osmesa OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs
for T in ${TASKS:-pick_place_any bin handover two_bins}; do
  $PY -m pi_model.eval_sim --checkpoint "$CKPT" --task "$T" --episodes ${EPISODES:-30} --episode-seconds 30 \
      --execute 25 --delay "$DELAY" --device cuda --subtask oracle --seed 4000 \
      --tokenizer weights/paligemma/tokenizer.model --out "outputs/c30_${NAME}_d${DELAY}_$T" ${EXTRA_ARGS:-}
done
$PY - "$NAME" "$DELAY" <<'PY'
import json, sys
from pathlib import Path
name, delay = sys.argv[1], sys.argv[2]
row = []
for t in ("pick_place_any", "bin", "handover", "two_bins"):
    f = Path(f"outputs/c30_{name}_d{delay}_{t}/results.json")
    if not f.exists():
        row.append(f"{t}=missing"); continue
    r = json.loads(f.read_text())["results"]
    strict = sum(1 for e in r if e.get("success") and e.get("lifted"))
    row.append(f"{t}={strict}/{len(r)}")
total = sum(int(x.split("=")[1].split("/")[0]) for x in row if "missing" not in x)
line = f"C30 {name} delay={delay} " + " ".join(row) + f" total={total}"
print(line)
with open("outputs/c30_summary.txt", "a") as fh:
    fh.write(line + "\n")
PY
