#!/usr/bin/env bash
# π0.7 data and prompting on π0.5's pretrained VLM + action expert (pi_model/pi05_backbone.py).
# Needs a LeRobot venv with this repo installed (see docs/HARDWARE_AND_TRAINING.md):
#   uv venv -p 3.12 ~/lerobot-venv && VIRTUAL_ENV=~/lerobot-venv uv pip install "lerobot[pi]==0.6.1"
#   VIRTUAL_ENV=~/lerobot-venv uv pip install --no-deps -e . && ... mujoco tokenizers sentencepiece av pyarrow scipy einops
# and the PaliGemma tokenizer at weights/paligemma/tokenizer.model (google/paligemma-3b-pt-224, gated).
#   GPUS=0 bash scripts/box/train_pi05.sh data/sim_mix_v1 runs/pi05_v1 20000
# Options: RESUME=<checkpoint dir> warm start; META=<prepare() dir> reuse statistics (default <data>/pi07_meta).
set -euo pipefail
DATA=${1:-data/sim_mix_v1}; OUT=${2:-runs/pi05_v1}; STEPS=${3:-20000}
PY=${PY:-$HOME/lerobot-venv/bin/python}; GPUS=${GPUS:-0}
export MUJOCO_GL=osmesa OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPUS
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
mkdir -p logs
if [ "$NPROC" -gt 1 ]; then LAUNCH=($PY -m torch.distributed.run --nproc_per_node "$NPROC"); else LAUNCH=($PY); fi
EXTRA=()
[ -n "${RESUME:-}" ] && EXTRA+=(--resume "$RESUME")
[ -n "${META:-}" ] && EXTRA+=(--meta "$META")
nohup "${LAUNCH[@]}" -m pi_model.train --data "$DATA" --config pi05 --tokenizer weights/paligemma/tokenizer.model "${EXTRA[@]}" \
    --out "$OUT" --steps "$STEPS" --batch-size ${BATCH:-32} --grad-accum 1 --lr ${LR:-2.5e-5} --warmup 1000 --bf16 \
    --gradient-checkpointing --device cuda:0 --num-workers 16 --log-every 20 --save-every ${SAVE_EVERY:-1000} \
    > "logs/train_$(basename "$OUT").log" 2>&1 &
echo "launched; tail -f logs/train_$(basename "$OUT").log"
