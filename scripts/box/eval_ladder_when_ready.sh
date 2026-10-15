#!/usr/bin/env bash
# On the evaluation GPU: score each relayed checkpoint on every ladder rung, in order.
#   RUN=full_v2 nohup bash scripts/box/eval_ladder_when_ready.sh 500 1000 1500 > logs/eval_ladder_full_v2.log 2>&1 &
# Prints one "RUNG <step> <rung> success=<x> lift=<y>" line per rung so trends are easy to grep.
set -u
RUN=${RUN:-full_v2}; EPISODES=${EPISODES:-8}; RUNGS=${RUNGS:-pick_place_any bin handover two_bins}
# NVIDIA EGL: the ladder episodes (most of the training data) were rendered with it, and it is ~100x
# faster than Mesa, which leaves the ~1 s policy calls as the main cost (~60 s per failed 30 s episode).
export MUJOCO_GL=${MUJOCO_GL:-egl} LP_NUM_THREADS=8
for step in "$@"; do
  ckpt=$(printf "exports/%s/step_%07d" "$RUN" "$step")
  until [ -f "$ckpt/model.pt" ]; do sleep 60; done
  while pgrep -f "[p]i_model.eval_sim" >/dev/null; do sleep 30; done  # one model at a time on the 24 GB GPU
  echo "=== $ckpt $(date) ==="
  for rung in $RUNGS; do
    out="outputs/eval_${RUN}_step${step}_${rung}"
    ~/pi-venv/bin/python -m pi_model.eval_sim --checkpoint "$ckpt" --task "$rung" --episodes "$EPISODES" \
        --episode-seconds 30 --execute 25 --delay 6 --device cuda --dtype bf16 --subtask oracle --out "$out" 2>&1 \
        | grep --line-buffered -E "Traceback|Error"
    python3 -c "
import json, statistics as s
r = json.load(open('$out/results.json'))
res = r['results']
print(f\"RUNG $step $rung success={r['success_rate']:.2f} lift={r['lift_rate']:.2f} \"
      f\"closest_mm={1000 * s.median(x['closest_hand_to_cube_m'] for x in res):.0f} \"
      f\"both_arms_touched={sum(all((x.get('touched') or {}).values()) for x in res)}/{len(res)}\", flush=True)"
  done
done
