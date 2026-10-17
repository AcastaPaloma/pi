#!/usr/bin/env bash
# Run 30-seed confirmations on a training run's checkpoints as they are saved (on the same GPU, beside training).
#   GPU=0 VARIANTS="d0 d6 d12" nohup bash scripts/box/c30_when_ready.sh runs/x_rtc_s0 3000 6000 &
# VARIANTS: dN = delay N steps; goal = delay 0 with oracle goal images (goal-conditioned runs).
set -u
RUN=${1:?run dir}; shift
NAME=$(basename "$RUN")
for S in "$@"; do
  D=$RUN/step_$(printf %07d "$S")
  until grep -qs "saved $D" "logs/train_$NAME.log"; do sleep 60; done
  for V in ${VARIANTS:-d0}; do
    case $V in
      goal) DELAY=0 EXTRA_ARGS="--goal oracle" bash scripts/box/confirm30.sh "$D" "${NAME}_${S}_goal" ;;
      d*) DELAY=${V#d} bash scripts/box/confirm30.sh "$D" "${NAME}_${S}" ;;
    esac
  done
done
echo "c30_when_ready done for $NAME"
