#!/usr/bin/env bash
# Closed-loop evaluation of a checkpoint in MuJoCo (headless).
#   bash scripts/box/eval.sh runs/small_v1/step_0020000 40
set -euo pipefail
CKPT=${1:?checkpoint dir}; EPISODES=${2:-40}; shift 2 || true
PY=${PY:-$HOME/pi-venv/bin/python}
export MUJOCO_GL=osmesa
$PY -m pi_model.eval_sim --checkpoint "$CKPT" --episodes "$EPISODES" --execute 25 --device cuda:0 \
    --out "outputs/eval_$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")" --placement wide "$@"
