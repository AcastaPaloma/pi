"""Time one policy call on the CPU in fp32 and bf16: python scripts/box/bench_cpu_inference.py runs/full_v1/step_0000500 40"""
import sys, time
import numpy as np, torch
from pi_model.policy import Pi07Policy, HistoryBuffer
from pi_model.prompt import PromptContext

ckpt, threads = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40
torch.set_num_threads(threads)
t = time.time(); policy = Pi07Policy(ckpt, device="cpu"); print(f"load {time.time()-t:.0f}s threads={torch.get_num_threads()}", flush=True)
S = policy.recorded_image_size
hist = HistoryBuffer(policy.cameras, hz=policy.hz)
rng = np.random.default_rng(0)
state_dim = len(policy.stats["state_q01"])
for k in range(policy.cfg.max_history):
    hist.append(k * policy.stride, {c: rng.integers(0, 255, (S, S, 3), dtype=np.uint8) for c in policy.cameras}, np.zeros(state_dim, np.float32))
ctx = PromptContext(task="put the red cube on the goal", subtask="reach the red cube", speed_steps=600, quality=5)
gen = torch.Generator()

def run(label):
    gen.manual_seed(0)
    with torch.no_grad():
        t = time.time(); a = policy.act(hist, ctx, generator=gen); dt = time.time() - t
    print(f"{label}: {dt:.1f}s first action {np.round(a[0, :6], 2)}", flush=True); return a

a32 = run("fp32"); run("fp32 again")
policy.model.to(torch.bfloat16)
a16 = run("bf16"); run("bf16 again")
print(f"max |fp32-bf16| over chunk: {np.abs(a32 - a16).max():.3f} deg", flush=True)
from torch.profiler import profile, ProfilerActivity
gen.manual_seed(0)
with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as prof:
    policy.act(hist, ctx, generator=gen)
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12), flush=True)
