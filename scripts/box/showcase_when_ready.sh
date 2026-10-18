#!/usr/bin/env bash
# Film a run's checkpoints for the write-up's Results tab as soon as they are saved, so every model is on video
# before it is discarded. Standard layouts: handover seed 4002 and two_bins seed 4003 (pick_place seed 4000 for
# single-arm runs); goal-conditioned runs are filmed once more with oracle goal images.
#   nohup bash scripts/box/showcase_when_ready.sh x_mem_s1 6000 > logs/showcase_x_mem_s1.log 2>&1 &
# Runs on the GPU with the most free memory, only when it has at least 20 GB free (re-checked every minute), and
# one recording at a time per box (flock), so training and scoring are never squeezed. Already filmed clips are
# skipped, so the script can be restarted freely. Clips: outputs/showcase/<task>_<run>_<step>_s<seed>[_goal].*
set -u
RUN=${1:?run name}; shift
PY=${PY:-$HOME/lerobot-venv/bin/python}
MIN_FREE_MB=${MIN_FREE_MB:-20000}
mkdir -p outputs/showcase logs
for S in "$@"; do
  D=runs/$RUN/step_$(printf %07d "$S")
  until grep -qs "saved $D" "logs/train_$RUN.log"; do sleep 60; done
  SPECS="handover:4002 two_bins:4003"
  "$PY" -c "import json,sys; c=json.load(open('runs/$RUN/config.json')); sys.exit(0 if c.get('max_subgoals',0)>0 else 1)" 2>/dev/null \
    && SPECS="$SPECS handover:4002:goal two_bins:4003:goal"
  for SPEC in $SPECS; do
    IFS=: read -r TASK SEED GOAL <<< "$SPEC"
    OUT=outputs/showcase/${TASK}_${RUN}_${S}_s${SEED}${GOAL:+_goal}
    [ -s "$OUT.json" ] && continue
    exec 9>/tmp/showcase.lock
    flock 9
    while :; do
      read -r GPU FREE < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ',' | sort -k2 -n -r | head -1)
      [ "${FREE:-0}" -ge "$MIN_FREE_MB" ] && break
      sleep 60
    done
    echo "$(date +%H:%M) filming $OUT on GPU $GPU (${FREE} MB free)"
    MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 "$PY" scripts/record_showcase.py --checkpoint "$D" \
        --task "$TASK" --seed "$SEED" --out "$OUT" ${GOAL:+--goal oracle} 2>&1 | grep -v -i warn | tail -2
    flock -u 9
  done
done
echo "$(date +%H:%M) showcase_when_ready done for $RUN"
