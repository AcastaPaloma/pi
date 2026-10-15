#!/usr/bin/env bash
# Parallel scripted-episode generation on the training box, then statistics + FAST fitting.
#   bash scripts/box/generate_data.sh data/sim_v1 800 24
set -euo pipefail
OUT=${1:-data/sim_v1}; TOTAL=${2:-800}; WORKERS=${3:-24}
PY=${PY:-$HOME/pi-venv/bin/python}
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
mkdir -p "$OUT" logs
PER=$(( (TOTAL + WORKERS - 1) / WORKERS ))
for w in $(seq 0 $((WORKERS - 1))); do
  START=$((w * PER)); COUNT=$(( TOTAL - START < PER ? TOTAL - START : PER ))
  [ "$COUNT" -le 0 ] && break
  nohup $PY -m pi_sim.record --out "$OUT" --episodes "$COUNT" --start-index "$START" --seed 0 \
      --styles clean,clean,slow,sloppy,slow_sloppy --colors red,green,blue,yellow --shapes cube,cylinder \
      --image-size 448 --placement wide > "logs/record_$w.log" 2>&1 &
done
wait
ls -d "$OUT"/episode_* | wc -l
$PY -m pi_model.data prepare --root "$OUT" --out "$OUT/pi07_meta" --config full
