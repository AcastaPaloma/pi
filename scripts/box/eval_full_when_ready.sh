#!/usr/bin/env bash
# Evaluate paper-size checkpoints on the CPU (the GPUs are full while training), as they appear.
#   nohup bash scripts/box/eval_full_when_ready.sh 1000 2000 3000 > logs/eval_full_v1.log 2>&1 &
# One 20 s episode costs about 14 min on 24 Xeon threads (40 policy calls of ~20 s), so six
# episodes take ~85 min; more threads oversubscribe the cores the training data loaders use.
set -u
RUN=${RUN:-runs/full_v1}; EPISODES=${EPISODES:-6}; THREADS=${THREADS:-24}
export MUJOCO_GL=osmesa LP_NUM_THREADS=4 OMP_NUM_THREADS=$THREADS
for step in "$@"; do
  ckpt=$(printf "%s/step_%07d" "$RUN" "$step")
  until [ -f "$ckpt/model.pt" ]; do sleep 120; done
  sleep 60
  echo "=== $ckpt $(date) ==="
  ~/pi-venv/bin/python -m pi_model.eval_sim --checkpoint "$ckpt" --episodes "$EPISODES" --execute 25 --delay 6 \
      --device cpu --subtask oracle --out "outputs/eval_$(basename "$RUN")_step$step" --placement wide 2>&1 \
      | grep --line-buffered -E "^episode|success rate|Traceback|Error"
done
