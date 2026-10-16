#!/usr/bin/env bash
# Evaluate each checkpoint of a run as it appears, on the training box (spare GPU memory beside training).
#   GPU=1 PY=~/lerobot-venv/bin/python nohup bash scripts/box/eval_when_ready.sh runs/pi05_single 2000 3000 > logs/eval_pi05_single.log 2>&1 &
# Prints one "EVAL <run> <step> success=<x> lift=<y>" line per checkpoint.
set -u
RUN=$1; shift
PY=${PY:-$HOME/lerobot-venv/bin/python}; GPU=${GPU:-1}; EPISODES=${EPISODES:-10}; TASK=${TASK:-pick_place}; DELAY=${DELAY:-0}
export MUJOCO_GL=osmesa LP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$GPU
for step in "$@"; do
  ckpt=$(printf "%s/step_%07d" "$RUN" "$step")
  until [ -f "$ckpt/model.pt" ]; do sleep 60; done
  sleep 60
  out="outputs/eval_$(basename "$RUN")_step${step}_${TASK}"
  $PY -m pi_model.eval_sim --checkpoint "$ckpt" --task "$TASK" --episodes "$EPISODES" --execute 25 --delay "$DELAY" \
      --device cuda:0 --subtask oracle --out "$out" > "$out.log" 2>&1
  python3 -c "
import json
r = json.load(open('$out/results.json'))
print(f\"EVAL $(basename "$RUN") $step $TASK success={r['success_rate']:.2f} lift={r['lift_rate']:.2f}\", flush=True)" \
    || { echo "EVAL $(basename "$RUN") $step FAILED"; tail -3 "$out.log"; }
done
