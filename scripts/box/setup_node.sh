#!/usr/bin/env bash
# One-time setup of a fresh Lambda GPU node for this repo (run on the node from ~/pi):
# uv, a LeRobot π0.5 venv with this repo, NVIDIA EGL + Mesa for MuJoCo rendering.
set -eu
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
BR=${DRV%%.*}
sudo -n apt-get update -qq >/dev/null 2>&1 || true
sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libosmesa6 libgl1 libglvnd0 >/dev/null 2>&1 || true
PKG=$(apt-cache policy "libnvidia-gl-$BR-server" 2>/dev/null | awk '/Candidate/{print $2}')
[ -n "$PKG" ] && sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "libnvidia-gl-$BR-server=$PKG" >/dev/null 2>&1 || true
[ -d ~/lerobot-venv ] || uv venv -q -p 3.12 ~/lerobot-venv
VIRTUAL_ENV=~/lerobot-venv uv pip install -q "lerobot[pi]==0.6.1"
VIRTUAL_ENV=~/lerobot-venv uv pip install -q --no-deps -e .
VIRTUAL_ENV=~/lerobot-venv uv pip install -q mujoco==3.15.0 tokenizers safetensors sentencepiece av pyarrow scipy einops pillow gymnasium==1.4.0
MUJOCO_GL=egl ~/lerobot-venv/bin/python -c "
from pi_sim.env import RobotEnv
e = RobotEnv(task='pick_place', control_mode='joint'); e.reset(seed=0); e.render_camera('overhead')
import torch, lerobot, pi_model; import time; t = time.time(); [e.render_camera('overhead') for _ in range(10)]; dt = (time.time() - t) / 10
print('node ready:', torch.__version__, torch.cuda.get_device_name(0), f'render {dt*1000:.1f} ms', 'NVIDIA EGL' if dt < 0.05 else 'SOFTWARE RENDERING - check libnvidia-gl')"
