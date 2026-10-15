#!/usr/bin/env bash
# On the evaluation GPU: evaluate each relayed bf16 checkpoint as it lands, in order.
#   nohup bash scripts/box/eval_a10_when_ready.sh 500 1000 1500 > logs/eval_a10.log 2>&1 &
# Checkpoints arrive in exports/<run>/step_<N>/ via scripts/relay_checkpoints.sh (model.pt appears last).
set -u
RUN=${RUN:-full_v1}; EPISODES=${EPISODES:-20}
# Mesa matches how the training data was rendered; NVIDIA EGL is 200x faster but shifts overhead
# lighting gradients and antialiasing by a few levels. Rendering is ~25 s of each episode either way.
export MUJOCO_GL=${MUJOCO_GL:-osmesa} LP_NUM_THREADS=8
for step in "$@"; do
  ckpt=$(printf "exports/%s/step_%07d" "$RUN" "$step")
  until [ -f "$ckpt/model.pt" ]; do sleep 60; done
  while pgrep -f "pi_model.eval_sim" >/dev/null; do sleep 30; done  # one model at a time on the 24 GB GPU
  echo "=== $ckpt $(date) ==="
  ~/pi-venv/bin/python -m pi_model.eval_sim --checkpoint "$ckpt" --episodes "$EPISODES" --execute 25 --delay 6 \
      --device cuda --dtype bf16 --subtask oracle --out "outputs/eval_${RUN}_step$step" --placement wide 2>&1 \
      | grep --line-buffered -E "^episode|success rate|Traceback|Error"
done
