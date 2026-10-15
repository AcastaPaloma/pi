#!/usr/bin/env bash
# Paper-size π0.7. From Gemma 3 4B (needs the gated weights on disk):
#   hf download google/gemma-3-4b-pt --local-dir weights/gemma-3-4b-pt   (after accepting the license)
#   bash scripts/box/train_full.sh data/sim_v1 runs/full_v1 30000
# Warm start from an earlier run's weights, re-initialising the state/action input-output layers
# (use when the action statistics change, e.g. adding two-arm data):
#   RESUME=runs/full_v1/step_0005500 bash scripts/box/train_full.sh data/sim_mix_v1 runs/full_v2 8000
set -euo pipefail
DATA=${1:-data/sim_v1}; OUT=${2:-runs/full_v1}; STEPS=${3:-10000}
GEMMA=${GEMMA:-weights/gemma-3-4b-pt}
PY=${PY:-$HOME/pi-venv/bin/python}
export MUJOCO_GL=osmesa OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "${RESUME:-}" ]; then
  INIT=(--resume "$RESUME" --reset-action-io); WARMUP=${WARMUP:-500}
else
  INIT=(--gemma "$GEMMA"); WARMUP=${WARMUP:-1000}
fi
mkdir -p logs
nohup $PY -m torch.distributed.run --nproc_per_node 2 -m pi_model.train --data "$DATA" --config full \
    "${INIT[@]}" --tokenizer "$GEMMA/tokenizer.model" --out "$OUT" --steps "$STEPS" --batch-size 4 \
    --grad-accum 4 --lr 5e-5 --warmup "$WARMUP" --fsdp --gradient-checkpointing --num-workers 12 \
    --log-every 10 --save-every 500 > "logs/train_$(basename "$OUT").log" 2>&1 &
echo "launched; tail -f logs/train_$(basename "$OUT").log"
