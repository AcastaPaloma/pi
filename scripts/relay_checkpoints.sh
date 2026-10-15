#!/usr/bin/env bash
# Run on a machine that can ssh to both boxes: wait for each training checkpoint, export it as bf16
# on the training box, and stream it to the evaluation GPU (the boxes have no direct access to each other).
#   nohup bash scripts/relay_checkpoints.sh 500 1000 1500 > relay.log 2>&1 &
set -u
TRAIN=${TRAIN:-ubuntu@192.222.52.191}; EVAL=${EVAL:-ubuntu@129.213.124.160}; RUN=${RUN:-full_v1}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30"
for step in "$@"; do
  name=$(printf "step_%07d" "$step")
  if $SSH "$EVAL" "test -f ~/pi/exports/$RUN/$name/model.pt"; then echo "$name already on the eval box"; continue; fi
  until $SSH "$TRAIN" "test -f ~/pi/runs/$RUN/$name/model.pt"; do sleep 120; done
  sleep 90  # let the trainer finish writing
  for attempt in 1 2 3; do
    $SSH "$TRAIN" "cd ~/pi && nice -n 10 ~/pi-venv/bin/python scripts/box/export_bf16.py runs/$RUN/$name exports/$RUN/$name >/dev/null" || { sleep 60; continue; }
    start=$(date +%s)
    if $SSH "$TRAIN" "cd ~/pi/exports/$RUN && tar cf - $name" | \
       $SSH "$EVAL" "mkdir -p ~/pi/exports/$RUN/.incoming && cd ~/pi/exports/$RUN/.incoming && rm -rf $name && tar xf - && mv $name .. "; then
      echo "$(date +%H:%M) relayed $name in $(( $(date +%s) - start )) s"
      $SSH "$TRAIN" "rm -rf ~/pi/exports/$RUN/$name"
      break
    fi
    echo "$(date +%H:%M) relay of $name failed (attempt $attempt)"; sleep 60
  done
done
