#!/usr/bin/env bash
# Paper-size π0.7 initialised from Gemma 3 4B. Needs the gated weights on disk:
#   hf download google/gemma-3-4b-pt --local-dir weights/gemma-3-4b-pt   (after accepting the license)
#   bash scripts/box/train_full.sh data/sim_v1 runs/full_v1 30000
set -euo pipefail
DATA=${1:-data/sim_v1}; OUT=${2:-runs/full_v1}; STEPS=${3:-30000}
GEMMA=${GEMMA:-weights/gemma-3-4b-pt}
PY=${PY:-$HOME/pi-venv/bin/python}
export MUJOCO_GL=osmesa OMP_NUM_THREADS=4
mkdir -p logs
nohup $PY -m torch.distributed.run --nproc_per_node 2 -m pi_model.train --data "$DATA" --config full \
    --gemma "$GEMMA" --tokenizer "$GEMMA/tokenizer.model" --out "$OUT" --steps "$STEPS" --batch-size 4 \
    --grad-accum 4 --lr 5e-5 --warmup 1000 --fsdp --gradient-checkpointing --num-workers 12 \
    --log-every 10 --save-every 500 > "logs/train_$(basename "$OUT").log" 2>&1 &
echo "launched; tail -f logs/train_$(basename "$OUT").log"
