#!/usr/bin/env bash
# From-scratch "small" π0.7 on two GPUs (no Gemma weights needed). Batch size is per GPU.
#   bash scripts/box/train_small.sh data/sim_v1 runs/small_v1 20000
set -euo pipefail
DATA=${1:-data/sim_v1}; OUT=${2:-runs/small_v1}; STEPS=${3:-20000}
PY=${PY:-$HOME/pi-venv/bin/python}
export MUJOCO_GL=osmesa OMP_NUM_THREADS=4
mkdir -p logs
nohup $PY -m torch.distributed.run --nproc_per_node 2 -m pi_model.train --data "$DATA" --config small \
    --out "$OUT" --steps "$STEPS" --batch-size 32 --lr 3e-4 --warmup 500 --bf16 --num-workers 12 \
    --log-every 20 --save-every 1000 > "logs/train_$(basename "$OUT").log" 2>&1 &
echo "launched; tail -f logs/train_$(basename "$OUT").log"
