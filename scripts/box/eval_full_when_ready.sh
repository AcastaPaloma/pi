#!/usr/bin/env bash
# Evaluate paper-size checkpoints on the CPU (the GPUs are full while training), as they appear.
set -u
export MUJOCO_GL=osmesa LP_NUM_THREADS=4 OMP_NUM_THREADS=40
for step in "$@"; do
  ckpt=$(printf "runs/full_v1/step_%07d" "$step")
  until [ -f "$ckpt/model.pt" ]; do sleep 120; done
  sleep 60
  echo "=== $ckpt $(date) ==="
  ~/pi-venv/bin/python -m pi_model.eval_sim --checkpoint "$ckpt" --episodes 6 --execute 25 --delay 6 \
      --device cpu --subtask oracle --out "outputs/eval_full_v1_step$step" --placement wide 2>&1 | grep -E "^episode|success rate|Traceback|Error"
done
