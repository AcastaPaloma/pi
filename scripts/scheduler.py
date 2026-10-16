"""Keep every GPU busy: launch the next queued experiment whenever a machine goes idle.

    nohup python3 scripts/scheduler.py > outputs/scheduler.log 2>&1 &

Every INTERVAL seconds it checks each H100 host for a running `pi_model.train` process and each
A10 host for a running `pi_model.eval_sim` process. A host idle on two consecutive checks gets the
first queued entry of its class from experiments/queue.json (preferring entries whose prefer_host
matches and whose requires_on_host paths exist there). It runs the entry's prep (on the main box),
setup, train and eval commands detached on the target, marks the entry running, and appends a line
to docs/RESEARCH_LOG.md (committed and pushed). Crashed or stopped runs free their GPU the same way.
"""
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "experiments" / "queue.json"
LOG = ROOT / "docs" / "RESEARCH_LOG.md"
MAIN = "192.222.52.191"
H100 = ["192.222.52.191:0", "192.222.52.191:1", "209.20.159.198:0", "209.20.157.232:0", "209.20.158.194:0"]
A10 = ["129.213.124.160", "132.145.199.243"]
INTERVAL = int(os.environ.get("INTERVAL", 300))
DRY_RUN = os.environ.get("DRY_RUN") == "1"   # print decisions only; one pass with no idle wait


def ssh(host, cmd, timeout=600):
    r = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", f"ubuntu@{host}", cmd],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip() + ("\n" + r.stderr.strip() if r.stderr.strip() else "")


def gpu_busy(slot):
    host, gpu = slot.split(":")
    # A training process on that GPU, or more than 10 GB of that GPU's memory in use.
    code, out = ssh(host, f"nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1=={gpu} {{print $2}}'")
    try:
        return int(out.split()[0]) > 10000
    except (ValueError, IndexError):
        return True  # unreachable host: never launch blindly


def a10_busy(host):
    """Busy if a scheduler job already runs there, or the GPU is too full for one more model (~9 GB).
    Checkpoint evaluations may run alongside a scheduler job."""
    code, out = ssh(host, "pgrep -f '[S]CHEDJOB_' >/dev/null && echo busy; "
                          "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1")
    if "busy" in out:
        return True
    try:
        return int(out.split()[-1]) > 13000
    except (ValueError, IndexError):
        return True


def resolve(text, host, run):
    text = text.replace("{HOST}", host).replace("{RUN}", run)
    for m in re.finditer(r"\{LATEST:([^}]+)\}", text):
        _, out = ssh(host, f"ls -d ~/pi/{m.group(1)}/step_* 2>/dev/null | sort | tail -1")
        latest = out.splitlines()[0].replace(str(Path.home()), "~") if out else ""
        latest = re.sub(r"^/home/ubuntu/pi/", "", latest)
        text = text.replace(m.group(0), latest)
    return text


def requirements_met(entry, host):
    for path in entry.get("requires_on_host", []):
        code, out = ssh(host, f"test -e ~/pi/{path} && echo ok")
        if "ok" not in out:
            return False
    return True


def log_line(text):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with LOG.open("a") as f:
        f.write(f"- **{stamp}** [scheduler] {text}\n")
    subprocess.run(["git", "-C", str(ROOT), "add", str(LOG), str(QUEUE)], capture_output=True)
    subprocess.run(["git", "-C", str(ROOT), "commit", "-q", "-m", f"Scheduler: {text[:60]}",
                    "-m", "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"], capture_output=True)
    subprocess.run(["git", "-C", str(ROOT), "push", "-q", "origin", "main"], capture_output=True)


def launch(entry, host, gpu=None):
    run = entry["name"]
    if entry.get("prep_on_main"):
        ssh(MAIN, f"cd ~/pi && {resolve(entry['prep_on_main'], host, run)}", timeout=7200)
    steps = []
    if entry.get("setup"):
        steps.append(f"( {resolve(entry['setup'], host, run)} )")
    for key in ("train", "eval"):
        if entry.get(key):
            cmd = resolve(entry[key], host, run)
            if gpu is not None:
                cmd = cmd.replace("GPUS=0", f"GPUS={gpu}").replace("GPU=0", f"GPU={gpu}")
            log = f"logs/{key}_{run}.log" if key == "train" else f"logs/eval_{run}.log"
            # shlex.quote keeps $VARS for the job's own shell (json.dumps let the login shell expand them).
            tagged = f": SCHEDJOB_{run}; {cmd}"
            steps.append(f"setsid nohup bash -c {shlex.quote(tagged)} < /dev/null > {log} 2>&1 &")
    code, out = ssh(host, "cd ~/pi && mkdir -p logs outputs && " + " ; cd ~/pi && ".join(steps), timeout=3600)
    return code, out


def main():
    idle_counts = {}
    print(f"{datetime.now():%H:%M} scheduler started", flush=True)
    while True:
        queue = json.loads(QUEUE.read_text())
        for cls, slots, busy in (("h100", H100, gpu_busy), ("a10", A10, a10_busy)):
            for slot in slots:
                try:
                    is_busy = busy(slot)
                except Exception as exc:  # network hiccup: skip this round
                    print(f"{datetime.now():%H:%M} {slot} check failed: {exc}", flush=True)
                    continue
                idle_counts[slot] = 0 if is_busy else idle_counts.get(slot, 0) + (2 if DRY_RUN else 1)
                if DRY_RUN:
                    print(f"{slot}: {'busy' if is_busy else 'IDLE'}", flush=True)
                if idle_counts[slot] < 2:
                    continue
                host, gpu = (slot.split(":") + [None])[:2]
                candidates = [e for e in queue[cls] if e.get("status") == "queued"]
                candidates.sort(key=lambda e: 0 if e.get("prefer_host") == host else 1)
                for entry in candidates:
                    if not requirements_met(entry, host):
                        continue
                    print(f"{datetime.now():%H:%M} launching {entry['name']} on {slot}", flush=True)
                    if DRY_RUN:
                        print("   would run:", resolve(entry.get("train") or entry.get("eval"), host, entry["name"])[:160], flush=True)
                        break
                    entry["status"] = "running"
                    entry["host"] = slot
                    entry["started"] = datetime.now().isoformat(timespec="minutes")
                    QUEUE.write_text(json.dumps(queue, indent=2) + "\n")
                    code, out = launch(entry, host, gpu if cls == "h100" else None)
                    if code != 0:
                        entry["status"] = "failed"
                        entry["error"] = out[-500:]
                        QUEUE.write_text(json.dumps(queue, indent=2) + "\n")
                        log_line(f"launch of {entry['name']} on {slot} failed: {out[-200:]}")
                    else:
                        log_line(f"{slot} was idle; launched **{entry['name']}** ({entry['hypothesis']})")
                    idle_counts[slot] = 0
                    break
                else:
                    if idle_counts[slot] == 2:
                        print(f"{datetime.now():%H:%M} {slot} idle but no runnable {cls} entry", flush=True)
        if DRY_RUN:
            return
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
