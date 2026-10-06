#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
exec conda run --no-capture-output -n "${PI_SIM_ENV:-mini-vla}" python -m pi_sim "$@"
